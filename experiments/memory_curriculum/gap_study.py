"""Audit missing menus and train a read-only evidence-gap diagnostic selector."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import re

from scope.atlas import StateAtlas
from scope.candidate_actions import candidate_actions
from scope.contracts import Budget, ContractError, Scope
from scope.data import SplitRegistry
from scope.environments import ShopActions
from scope.io import atomic_json, atomic_jsonl, canonical, digest, file_sha, read_jsonl
from scope.memory_gaps import GAP_TYPES, diagnose, diagnosis_messages
from scope.teacher import LocalTeacher
from scope.tools import ToolExecutor, parse_tool
from scope.validation import public_prices


def history_item(record, focus=None):
    text = record["text"]
    start = text.find(focus) if focus else -1
    start = max(0, start-80) if start >= 0 else 0
    return {"id":record["id"],"entity":record["entity"],"kind":record["kind"],"start":start,"text":text[start:start+450]}


def old_failures(path):
    output = []
    for row in read_jsonl(path):
        manager = iter(row["manager_operations"])
        spent, state_used, previous = 0, False, None
        for event in row["history"]:
            if event["phase"] == "execute":
                previous, state_used = event, False
                continue
            call = next(manager)
            result = call["result"]
            if result.get("operation") in {"RETRIEVE", "FETCH_EVIDENCE"} and result["accepted"]:
                spent += 1
                state_used = state_used or call["phase"] == "state"
            if result.get("reason") != "no_public_candidate":
                continue
            phase = call["phase"]
            evidence = [e for e in row["public_evidence"] if e["revision"] <= call["revision"] and not e.get("terminal")]
            observations = [e for e in evidence if e["kind"] == "observation"]
            goal = next(e["text"] for e in evidence if e["kind"] == "requirement")
            current = observations[-1]
            entity = current["entity"]
            product_id = entity.split(":",1)[1] if entity.startswith("product:") else None
            prior = next((e for e in reversed(observations[:-1]) if product_id and e["entity"]=="task" and
                          product_id in e.get("clickables",[]) and product_id in e["text"]), None)
            target, expected, detail, witness = None, "undetermined", "need_exact_pre_action_graph", None
            # Only classify the specific target identified at this boundary.
            if previous and previous.get("loop") and (spent >= 6 or (phase=="state" and state_used)):
                target, expected = "retrieve prior loop evidence", "budget_limited"
                detail = "episode_retrieval_budget" if spent >= 6 else "state_lookup_used_this_turn"
            elif phase=="context" and prior and previous and previous.get("command")==f"click[{product_id}]":
                atlas = StateAtlas(Scope("audit",row["task_id"],str(call["revision"])))
                try:
                    atlas.append_public(goal,"requirement")
                    for source in (prior,current):
                        atlas.append_public(source["text"],"observation",source["entity"],clickables=source.get("clickables",[]))
                    menu = candidate_actions(atlas,{"state_graph":True,"current_entity":entity},goal,"context",
                                             Budget(retrievals=max(0,6-spent)),ToolExecutor(adapter=ShopActions()))
                finally:
                    atlas.close()
                if any(option["operation"]=="SUMMARIZE" for option in menu.values()):
                    target, expected, detail, witness = "preserve the search result and newly opened product", "menu_gap", "shadow_summary_validated", prior
            elif "预算不超过" in goal and entity.startswith("product:"):
                own = [e for e in observations if e["entity"]==entity]
                prices = [m.group(0) for e in own for m in public_prices(e["text"])
                          if re.search(r"(?:^|\n|\[SEP\])\s*$", e["text"][:m.start()])]
                if prices and not any(any(unit in p for unit in ("元", "人民币", "￥")) for p in prices):
                    target, expected, detail = "explicit currency for the current product price", "missing_evidence", "displayed_price_has_no_currency"
            selected = ([witness] if witness else []) + observations[-2:]
            prefix = {"target":target,"phase":phase,"public_goal":goal,"remaining_retrievals":max(0,6-spent),
                      "state_lookup_used":state_used,"public_prefix":[history_item(e,product_id) for e in selected],
                      "projected_fields":(["search-result listing and current product identity"] if witness else []),
                      "menu_operations":["FINISH"],"audit_projection_complete":False}
            secondary = (["explicit color requirement used wording 颜色要, omitted by the old goal extractor"]
                         if re.search(r"颜色\s*要\s*(黑色|白色|红色|蓝色|灰色|绿色|银色|金色|透明)", goal) else [])
            output.append({"task_id":row["task_id"],"revision":call["revision"],"prefix":prefix,"gold":expected,
                           "detail":detail,"witness_id":witness and witness["id"],
                           "witness_token":product_id if witness else None,"secondary_findings":secondary})
    return output


def controlled(task, category, variant):
    identity = digest([task, category, variant])
    index = int(identity[12],16) % 3
    number = 1+int(identity[10],16)
    entities = ("product:"+str(int(identity[:10],16)),f"alf:book {number}",f"alf:cabinet {number}")
    target, proof, incomplete = (
        ("selected_color", "已选颜色：蓝色", "可选颜色：蓝色、黑色"),
        ("held", f"You pick up the book {number} from the desk 1.", f"You see a book {number} on the desk 1."),
        ("container_open", f"You open the cabinet {number}.", f"You see a cabinet {number}."),
    )[index]
    available = category != "missing_evidence"
    projected = category in {"menu_gap", "budget_limited"}
    allowed = category != "budget_limited"
    named = target if category != "undetermined" else None
    prefix = {"target":named,"phase":"context","remaining_retrievals":4 if allowed else 0,"state_lookup_used":False,
              "public_prefix":[{"id":f"obs_000007_{identity[:10]}","entity":entities[index],"kind":"observation",
                                "start":0,"text":proof if available else incomplete}],
              "projected_fields":[target] if projected else [],"menu_operations":["FINISH"],"audit_projection_complete":True}
    gold = diagnose(target=named,public_witness=available,extracted=projected,menu_covers=False,allowed=allowed)
    if gold!=category:raise ContractError("Controlled diagnosis label mismatch")
    return {"task_id":task,"revision":variant,"prefix":prefix,"gold":gold,
            "witness_id":prefix["public_prefix"][0]["id"] if category in {"menu_gap","extractor_gap"} else None,
            "witness_token":proof if category in {"menu_gap","extractor_gap"} else None}


def labeled(row):
    messages, options = diagnosis_messages(row["prefix"], [row["task_id"],row["revision"],row["prefix"]])
    correct = next(k for k,v in options.items() if v["diagnosis"]==row["gold"])
    return {**row,"phase":"state","messages":messages,"completion":correct,"source":"state_graph_gap_choice"}


def teacher_audit(teacher, row):
    request = [dict(r) for r in row["messages"]]
    request[0]["content"] = request[0]["content"].rsplit("Answer",1)[0] + (
        'Return only JSON {"choice":"letter","evidence_id":null,"quote":null}. '
        'For a projection or menu gap, cite a public evidence ID and exact substring. '
        'For missing evidence, budget and unknown, evidence_id and quote may be null.')
    try:
        answer = parse_tool(teacher(request))
        menu = json.loads(row["messages"][-1]["content"])["verified_options"]
        choice = answer["choice"]
        kind = menu[choice]["diagnosis"]
        quote, eid = answer.get("quote"), answer.get("evidence_id")
        exact = isinstance(quote,str) and bool(quote) and any(e["id"]==eid and quote in e["text"] for e in row["prefix"]["public_prefix"])
        supported = kind not in {"menu_gap", "extractor_gap"} or (exact and eid==row["witness_id"] and
                                                                 bool(row["witness_token"] and row["witness_token"] in quote))
        accepted = bool(choice==row["completion"] and supported)
        return {"teacher_class":kind,"citation_valid":bool(exact),"accepted":accepted,
                "evidence_id":eid if exact and accepted else None,"quote":quote if exact and accepted else None}
    except (KeyError,TypeError,ValueError):
        return {"teacher_class":None,"citation_valid":False,"accepted":False}


def teacher_contrast(row):
    audit = row.get("teacher_audit") or {}
    if row["gold"] not in {"extractor_gap", "menu_gap"} or not audit.get("accepted") or not audit.get("quote"):
        return None
    prefix = copy.deepcopy(row["prefix"])
    # Hold the exact teacher-verified observation fixed and vary only projection.
    prefix["projected_fields"] = [] if row["gold"] == "menu_gap" else [prefix["target"]]
    gold = diagnose(target=prefix["target"],public_witness=True,extracted=bool(prefix["projected_fields"]),
                    menu_covers=False,allowed=True)
    return labeled({"task_id":row["task_id"],"revision":row["revision"],"prefix":prefix,"gold":gold,
                    "witness_id":audit["evidence_id"],"witness_token":audit["quote"],
                    "label_origin":"teacher_verified_projection_contrast"})


def build(args, config):
    out=args.output
    registry=SplitRegistry(config["data_root"],config.get("split_seed",233))
    source=args.previous
    old=json.loads((source/"study_receipt.json").read_text())
    excluded=set(old["excluded_previous_dev"])|set(old["dev_tasks"])
    ids=sorted(registry.ids("rl_train"),key=lambda t:digest(["gap_train",t]))[:20]
    dev=sorted(set(registry.ids("internal_dev"))-excluded,key=lambda t:digest(["gap_dev",t]))[:6]
    if len(dev)!=6 or set(ids)&set(dev):raise ContractError("Need six unseen diagnostic dev tasks")
    train=[labeled(controlled(t,k,i)) for i,t in enumerate(ids) for k in GAP_TYPES]
    evaluation=[labeled(controlled(t,k,i+30)) for i,t in enumerate(dev) for k in GAP_TYPES]
    real=[labeled(r) for r in old_failures(source/"live_after"/"results.jsonl")]
    if len(real)!=13:raise ContractError("Prior missing-candidate audit does not match the 13 recorded events")
    teacher = LocalTeacher(args.teacher_model,"cuda") if args.teacher_model else None
    if teacher:
        for row in [*train[:10], *real]:row["teacher_audit"]=teacher_audit(teacher,row)
    contrasts = [twin for row in train if (twin:=teacher_contrast(row)) is not None]
    train.extend(contrasts)
    random.Random(233).shuffle(train)
    atomic_jsonl(out/"warmup.jsonl",train)
    atomic_jsonl(out/"controlled_dev.jsonl",evaluation)
    atomic_jsonl(out/"historical_audit.jsonl",real)
    report={"train_tasks":ids,"dev_tasks":dev,"historical_dev_used_for_training":False,
            "warmup_sha256":file_sha(out/"warmup.jsonl"),"controlled_dev_sha256":file_sha(out/"controlled_dev.jsonl"),
            "historical_causes":dict(Counter(r["gold"] for r in real)),
            "teacher_historical_accepted":sum((r.get("teacher_audit") or {}).get("accepted",False) for r in real),
            "teacher_train_accepted":sum((r.get("teacher_audit") or {}).get("accepted",False) for r in train),
            "teacher_derived_contrasts":len(contrasts),
            "teacher_identity":teacher.identity if teacher else None}
    atomic_json(out/"study_receipt.json",report)
    print(canonical(report),flush=True)


def evaluate(args, config):
    import torch
    from scope.checkpoint import load_checkpoint
    from scope.policy import HFPolicy
    output=args.output
    report={}
    policy=HFPolicy(config["model"]["path"])
    all_rows=[]
    for condition,path in (("before",args.before),("after",args.after)):
        load_checkpoint(path,policy,restore_random=False)
        initial=policy.fingerprint()
        for dataset,file,reps in (("controlled","controlled_dev.jsonl",2),("historical","historical_audit.jsonl",1)):
            stats=Counter()
            for i,row in enumerate(read_jsonl(output/file)):
                options=json.loads(row["messages"][-1]["content"])["verified_options"]
                for j in range(reps):
                    seed=1600+i*2+j
                    torch.manual_seed(seed)
                    sample=policy.sample_choice(row["messages"],"state",digest([row["task_id"],i]),list(options),2048,greedy=dataset=="historical")
                    predicted=options[sample.text]["diagnosis"]
                    correct=predicted==row["gold"]
                    stats["n"]+=1; stats["correct"]+=correct
                    stats["n:"+row["gold"]]+=1;stats["correct:"+row["gold"]]+=correct
                    all_rows.append({"condition":condition,"dataset":dataset,"task_id":row["task_id"],"revision":row["revision"],
                                     "seed":seed,"gold":row["gold"],"predicted":predicted,"correct":correct})
            report[condition+"_"+dataset]=dict(stats)
        if policy.fingerprint()!=initial:raise ContractError("Diagnostic eval modified the checkpoint")
    atomic_jsonl(output/"diagnostic_results.jsonl",all_rows)
    atomic_json(output/"diagnostic_summary.json",report)
    print(canonical(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("build","evaluate"))
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--previous",type=Path)
    p.add_argument("--teacher-model")
    p.add_argument("--before",type=Path)
    p.add_argument("--after",type=Path)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    (build if args.command=="build" else evaluate)(args,cfg)
