"""Causal-prefix, public-only tool SFT examples; never reuse fixed traces as GRPO rollouts."""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from collections import Counter
from .atlas import StateAtlas
from .contracts import Budget,ContractError,Sample,Scope
from .io import atomic_json,append_jsonl,canonical,digest,read_jsonl
from .rollout import manager_messages
from .tools import ToolExecutor,parse_tool,STATE_OPS,CONTEXT_OPS
from .public_data import assert_public_text,TERMINAL_PUBLIC_TEXT


def build_examples(public_file,output,allowed_tasks):
    target=Path(output)
    if target.exists():raise ContractError("Refusing to overwrite warmup examples")
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name("."+target.name+"."+uuid.uuid4().hex+".tmp")
    try:
        report=_build_examples(public_file,temp,allowed_tasks)
        os.replace(temp,target)
        return report
    finally:
        if temp.exists():temp.unlink()


def _build_examples(public_file,output,allowed_tasks):
    allowed=set(map(str,allowed_tasks));count=0
    filtered=Counter()
    target=Path(output)
    if target.exists():raise ContractError("Refusing to overwrite warmup examples")
    for index,row in enumerate(read_jsonl(public_file)):
        task=str(row["task_id"])
        if task not in allowed:raise ContractError("Warmup task outside designated training/validation partition")
        if "atlas" in row:
            evidence=[json.loads(r[1]) for r in row["atlas"]["tables"]["evidence"]]
            evidence.sort(key=lambda r:r["revision"])
        else:evidence=row["evidence"]
        requirements=[r for r in evidence if r["kind"]=="requirement"]
        if not requirements:raise ContractError("Public requirement missing")
        if not evidence or evidence[0]["kind"]!="requirement":raise ContractError("Requirement must precede observations")
        goal=requirements[0]["text"]
        atlas=StateAtlas(Scope("warmup",task,f"prefix{index}"))
        tools=ToolExecutor()
        try:
            for raw in evidence:
                assert_public_text(raw["text"])
                if raw.get("terminal") or raw["text"]==TERMINAL_PUBLIC_TEXT:break
                if raw.get("scope",{}).get("task",task)!=task:raise ContractError("Public evidence task scope mismatch")
                record=atlas.append_public(raw["text"],raw["kind"],raw.get("entity","task"),clickables=raw.get("clickables",[]))
                if raw["kind"]=="requirement":goal=raw["text"]
                ref={"id":record["id"],"start":0,"end":min(120,len(record["text"]))}
                if ref["end"]==0:continue
                proposals=[("state",{"operation":"PROPOSE_STATE_UPDATE","entity":record["entity"],
                          "predicate":"quoted_text","value":record["text"][:ref["end"]],"support_status":"supported","evidence_refs":[ref]}),
                           ("context",{"operation":"RETRIEVE","query":record["id"],"top_k":4,"max_hops":0}),
                           ("context",{"operation":"SUMMARIZE","items":[{"text":record["text"][:ref["end"]],"evidence_ref":ref}]}),
                           ("state",{"operation":"FINISH"}),("context",{"operation":"FINISH"})]
                if record["kind"]=="action_receipt":
                    proposals.append(("state",{"operation":"PROPOSE_STATE_UPDATE","entity":record["entity"],
                        "predicate":"attempted_action","value":record["text"],"support_status":"supported",
                        "evidence_refs":[{"id":record["id"],"start":0,"end":len(record["text"])}]}))
                if record["kind"]=="observation" and record["entity"]!="task":
                    from .validation import PRICE
                    price_matches=PRICE.findall(record["text"][:ref["end"]])
                    if price_matches:
                        first=price_matches[0]
                        proposals.append(("state",{"operation":"PROPOSE_STATE_UPDATE","entity":record["entity"],
                            "predicate":"price","value":float(first[0] or first[1]),"support_status":"supported","evidence_refs":[ref]}))
                    for color in record.get("clickables",[]):
                        if color in {"黑色","白色","红色","蓝色","灰色","绿色","银色","金色","black","white","red","blue"}:
                            at=record["text"].find(color)
                            if at<0:continue
                            color_ref={"id":record["id"],"start":at,"end":at+len(color)}
                            for predicate,status in (("available_option","supported"),("selected_color","pending")):
                                proposals.append(("state",{"operation":"PROPOSE_STATE_UPDATE","entity":record["entity"],
                                    "predicate":predicate,"value":color,"support_status":status,"evidence_refs":[color_ref]}))
                            break
                for phase,proposal in proposals:
                    proposal["expected_revision"]=atlas.revision
                    clone=StateAtlas.restore(atlas.export())
                    try:tools.execute(clone,{},proposal,phase,Budget(),goal)
                    except ContractError as exc:
                        filtered[str(exc)]+=1
                        continue
                    finally:clone.close()
                    messages=manager_messages(atlas,record["text"],{},goal,phase,Budget())
                    append_jsonl(target,{"task_id":task,"prefix_evidence_id":record["id"],"phase":phase,
                                 "messages":messages,"completion":canonical(proposal),"source":"checked_public_prefix_rule"})
                    count+=1
        finally:atlas.close()
    if not count:raise ContractError("No checked examples produced")
    return {"examples":count,"source":"public_prefix_rules","shopping_success_rate":None,"filtered_proposals":dict(filtered)}


def warmup(policy,examples,optimizers,output,receipt_hash,epochs=1,batch_size=1,allowed_tasks=(),resume=None,stop_after=None,checkpoint_every=100):
    from .checkpoint import save_checkpoint,load_checkpoint
    rows=read_jsonl(examples)
    allowed=set(map(str,allowed_tasks));valid=[]
    for row in rows:
        if str(row["task_id"]) not in allowed:raise ContractError("SFT split leakage")
        if row["phase"] not in {"state","context"}:raise ContractError("Unknown SFT management phase")
        operation=parse_tool(row["completion"]).get("operation")
        if operation not in (STATE_OPS if row["phase"]=="state" else CONTEXT_OPS):raise ContractError("SFT label is not a tool in this phase")
        ids=policy.encode(row["messages"])
        tokens=policy.tokenizer.encode(row["completion"],add_special_tokens=False)
        if policy.tokenizer.eos_token_id is not None:tokens.append(policy.tokenizer.eos_token_id)
        if len(ids)>2048 or len(tokens)>512:continue
        valid.append(Sample("manager",row["phase"],ids,tokens,[0.]*len(tokens),row["completion"],
                            policy.versions["manager"],digest(row)))
    if not valid:raise ContractError("No warmup samples fit the configured token budgets")
    updates=0
    batches=(len(valid)+batch_size-1)//batch_size
    total_updates=epochs*batches
    if resume:
        saved=load_checkpoint(resume,policy,optimizers,receipt_hash)
        if saved["extra"].get("stage")!="manager_sft":raise ContractError("Not a manager SFT checkpoint")
        updates=saved["step"]
        journal=Path(output)/"warmup_metrics.jsonl"
        if journal.exists():
            from .io import atomic_jsonl
            metrics_rows=read_jsonl(journal)
            uncommitted=[row for row in metrics_rows if row["update"]>updates]
            if uncommitted:
                atomic_jsonl(Path(output)/("uncommitted_warmup_metrics_"+digest(uncommitted)[:12]+".jsonl"),uncommitted)
                atomic_jsonl(journal,[row for row in metrics_rows if row["update"]<=updates])
    else:
        save_checkpoint(Path(output)/"checkpoints",0,policy,optimizers,receipt_hash,{"stage":"manager_sft"})
    before=policy.fingerprint("manager");actor=policy.fingerprint("actor")
    completed=updates
    for epoch in range(epochs):
        for batch_index,start in enumerate(range(0,len(valid),batch_size)):
            target_step=epoch*batches+batch_index+1
            if target_step<=updates:continue
            if stop_after is not None and target_step>stop_after:break
            metrics=policy.sft_update(valid[start:start+batch_size],optimizers["manager"])
            completed=target_step
            append_jsonl(Path(output)/"warmup_metrics.jsonl",{"epoch":epoch,"update":completed,**metrics})
            final=completed==total_updates
            stopping=stop_after is not None and completed==stop_after
            if final:policy.reset_reference("manager")
            if final or stopping or completed%checkpoint_every==0:
                path=save_checkpoint(Path(output)/"checkpoints",completed,policy,optimizers,receipt_hash,
                                    {"stage":"manager_sft","total_updates":total_updates,"epoch":epoch})
            atomic_json(Path(output)/"status.json",{"phase":"warmup","completed_updates":completed,"expected_updates":total_updates})
    if policy.fingerprint("actor")!=actor:raise ContractError("Actor changed during manager SFT")
    latest=json.loads((Path(output)/"checkpoints/latest.json").read_text())
    path=Path(output)/"checkpoints"/latest["path"]
    summary={"updates":completed,"examples":len(valid),"filtered_over_budget":len(rows)-len(valid),
             "manager_changed":before!=policy.fingerprint("manager"),"actor_unchanged":True,
             "checkpoint":str(path),"shopping_success_rate":None,"complete":completed==total_updates}
    atomic_json(Path(output)/"summary.json",summary)
    atomic_json(Path(output)/"status.json",{"phase":"complete" if summary["complete"] else "stopped_at_update_boundary",
                "completed_updates":completed,"expected_updates":total_updates})
    return summary

