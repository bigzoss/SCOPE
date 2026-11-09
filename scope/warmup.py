"""Causal-prefix, public-only tool SFT examples; never reuse fixed traces as GRPO rollouts."""
from __future__ import annotations
import copy
import json
import os
import uuid
from pathlib import Path
from collections import Counter
from .atlas import StateAtlas
from .candidate_actions import candidate_actions, choice_messages
from .contracts import Budget,ContractError,Sample,Scope
from .decision_evidence import review
from .io import atomic_json,append_jsonl,canonical,digest,read_jsonl
from .rollout import manager_messages
from .tools import ToolExecutor,context_view,parse_tool,STATE_OPS,CONTEXT_OPS,GRAPH_STATE_OPS
from .memory_gaps import confirmation_status
from .public_data import assert_public_text,TERMINAL_PUBLIC_TEXT
from .state_graph import apply_observation, compression_due, core_claims, historical_candidate_evidence, management_reason
from .interaction import record_progress
from .recovery import prepare_recovery, record_recovery


def build_examples(public_file,output,allowed_tasks,config=None,teacher=None):
    target=Path(output)
    if target.exists():raise ContractError("Refusing to overwrite warmup examples")
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name("."+target.name+"."+uuid.uuid4().hex+".tmp")
    try:
        graph = config and config["runtime"].get("state_graph")
        if teacher and not graph:
            raise ContractError("Teacher distillation requires runtime.state_graph")
        report=(_build_graph_examples(public_file,temp,allowed_tasks,config,teacher) if graph
                else _build_examples(public_file,temp,allowed_tasks))
        os.replace(temp,target)
        return report
    finally:
        if temp.exists():temp.unlink()


def graph_adapter(kind):
    if kind == "alfworld_http":
        from .alfworld import ALFWorldActions
        return ALFWorldActions()
    from .environments import ShopActions
    return ShopActions()


def public_evidence(row):
    if "atlas" in row:
        return sorted((json.loads(r[1]) for r in row["atlas"]["tables"]["evidence"]), key=lambda r: r["revision"])
    return row["evidence"]


def _decision_example(atlas, working, goal, phase, budget, tools, observation, future, teacher):
    menu = candidate_actions(atlas,working,goal,phase,budget,tools)
    if len(menu) < 2 and not (teacher and working.get("freeform_manager")):
        return None
    slots = review(atlas,working,goal)
    option, category, checked = None, None, {}
    if phase == "state":
        lookup = [(label,p) for label,p in menu.items() if p["operation"]=="RETRIEVE"]
        rank = {(s["entity"],s["predicate"]):i for i,s in enumerate(slots)}
        lookup.sort(key=lambda pair:rank.get((pair[1].get("entity"),pair[1].get("predicate")),len(rank)))
        pending = [(label,p) for label,p in menu.items() if p["operation"]=="PROPOSE_STATE_UPDATE"
                   and p["support_status"]=="pending"]
        primary = slots[0] if slots else None
        visible = {working.get("observation_id"), *working.get("selected_ids", [])}
        if primary and primary["status"]=="supported" and any(ref["id"] in visible for ref in primary["evidence_refs"]):
            option = next(k for k,p in menu.items() if p["operation"]=="FINISH")
            category = "evidence_visible"
        elif working.get("recovery_slots") and primary and primary["status"] == "pending" and not lookup:
            option = next(k for k,p in menu.items() if p["operation"]=="FINISH")
            category = "evidence_pending"
        elif lookup:
            option,proposal = lookup[0]
            category = "historical_evidence"
            if teacher and teacher.counts["decision_witness_calls"] < teacher.max_calls and proposal.get("predicate") and proposal.get("source") != "raw":
                spans = []
                target = next(s for s in slots if (s["entity"],s["predicate"])==(proposal["entity"],proposal["predicate"]))
                for label,candidate in lookup:
                    if candidate.get("source") == "raw":
                        continue
                    record = atlas.evidence(candidate["query"])
                    refs = [ref for s in slots if (s["entity"],s["predicate"])==(candidate.get("entity"),candidate.get("predicate"))
                            for ref in s.get("evidence_refs",[]) if ref["id"]==record["id"]]
                    if refs:
                        ref = refs[0]
                        witness = {"id":record["id"],"quote":record["text"][ref["start"]:ref["end"]]}
                        if witness not in spans:
                            spans.append(witness)
                if spans:
                    teacher.counts["decision_witness_calls"] += 1
                    chosen = teacher.generate.select_witness(
                        {k:target[k] for k in ("entity","predicate","value")},spans)
                    admissible = {(ref["id"],atlas.evidence(ref["id"])["text"][ref["start"]:ref["end"]])
                                  for ref in target["evidence_refs"]}
                    if chosen and chosen in spans and (chosen["id"],chosen["quote"]) in admissible:
                        checked = {"teacher_evidence_id":chosen["id"],"teacher_quote":chosen["quote"]}
                        teacher.counts["decision_witness_accepted"] += 1
        elif pending or slots:
            option = pending[0][0] if pending else next(k for k,p in menu.items() if p["operation"]=="FINISH")
            category = "evidence_pending" if pending or any(s["status"]!="supported" for s in slots) else "evidence_visible"
    elif future:
        next_command = future["text"]
        for label,proposal in menu.items():
            if proposal["operation"] != "SUMMARIZE":
                continue
            adapter = tools.validator.adapter
            command = adapter.normalize(next_command)
            binding = adapter.bind(command,working["public_state"])
            if binding.status not in {"bound","not_required","unchecked"}:
                continue
            shadow, shadow_working = StateAtlas.restore(atlas.export()), copy.deepcopy(working)
            try:
                before = context_view(shadow,shadow_working,goal)
                tools.execute(shadow,shadow_working,proposal,phase,copy.deepcopy(budget),goal)
                after = context_view(shadow,shadow_working,goal)
                if (before["goal"],before["claims"],before["constraint_checklist"],before.get("decision_evidence")) != (
                    after["goal"],after["claims"],after["constraint_checklist"],after.get("decision_evidence")):
                    raise ContractError("Summary removed a decision-relevant state field")
                if after["summary"]:
                    option, category = label,"compression_before_logged_action"
                    checked = {"following_logged_action":next_command,"preserved_claim_count":len(after["claims"])}
            finally:
                shadow.close()
            break
    if option is None and not teacher:
        return None
    if option is None:
        option = next((label for label, p in menu.items() if p["operation"] == "FINISH"), None)
        if option is None:
            return None
        category, checked = "teacher_reviewed_prefix", {}
    messages = choice_messages(manager_messages(atlas,observation,working,goal,phase,budget),menu)
    if working.get("decision_recovery") and teacher:
        base_messages = manager_messages(atlas,observation,working,goal,phase,budget)
        learned = teacher.annotate(atlas, working, goal, phase, budget, base_messages, tools, menu[option])
        if learned:
            if working.get("freeform_manager"):
                proposal = learned.pop("proposal")
                return {"messages":base_messages, "completion":canonical(proposal),
                        "source":"teacher_state_graph_public_prefix", "decision_case":"teacher_reviewed_prefix",
                        "operation":proposal["operation"], **learned}
            selected = next((label for label, p in menu.items() if p == learned["proposal"]), None)
            if selected is not None:
                option = selected
                checked.update(teacher_case=learned["teacher_case"], teacher_trigger=learned["teacher_trigger"],
                               verified_target_ids=learned["verified_target_ids"], teacher_probe=learned["teacher_probe"])
            else:
                teacher.counts["online_menu_gap"] += 1
                # Do not train a categorical label for an action absent from the
                # deployed menu, or silently report this as teacher supervision.
                checked["teacher_menu_gap"] = True
    if working.get("freeform_manager"):
        return {"messages":manager_messages(atlas,observation,working,goal,phase,budget),
                "completion":canonical(menu[option]), "source":"state_graph_public_prefix",
                "decision_case":"freeform_rule_prefix", "operation":menu[option]["operation"], **checked}
    return {"messages":messages,"completion":option,"source":"state_graph_candidate_choice",
            "decision_case":category,"operation":menu[option]["operation"],**checked}


def _graph_teacher(atlas, working, goal, phase, budget):
    action = {"operation": "FINISH", "expected_revision": atlas.revision}
    reason = management_reason(atlas, working, goal, phase)
    entity = working["current_entity"]
    claims = [c for c in core_claims(atlas, working) if c.get("state_slot") and c["entity"] != "task"
              and c["support_status"] == "supported"]
    if phase == "state" and reason == "historical_candidate_evidence" and budget.retrievals:
        evidence = historical_candidate_evidence(atlas, working)
        return {**action, "operation":"RETRIEVE", "query":evidence["id"], "entity":"task", "top_k":1}
    if phase == "state" and reason == "missing_evidence" and entity.startswith("product:"):
        current = atlas.all_evidence()[-1]
        known = {c["value"] for c in atlas.nodes("Claim") if c["entity"] == entity and c["predicate"] == "available_option"}
        option = next((label for label in current.get("clickables", []) if label and label in goal
                       and label not in known and label in current["text"]), None)
        if option:
            start = current["text"].find(option)
            return {**action,"operation":"PROPOSE_STATE_UPDATE","entity":entity,"predicate":"available_option",
                    "value":option,"support_status":"supported",
                    "evidence_refs":[{"id":current["id"],"start":start,"end":start+len(option)}]}
    if phase == "context" and compression_due(atlas, working):
        recent_ids = {e["id"] for e in atlas.all_evidence()[-2:]}
        refs = {ref["id"]: ref for claim in claims for ref in claim["evidence_refs"] if ref["id"] not in recent_ids}
        items = [{"text": atlas.evidence(ref["id"])["text"][ref["start"]:ref["end"]], "evidence_ref": ref}
                 for ref in list(refs.values())[:3]]
        if not items:
            attempts = [e for e in atlas.all_evidence()[:-2] if e["kind"] == "action_receipt" and e["entity"] == entity]
            repeated = Counter(e["text"] for e in attempts)
            record = next((e for e in reversed(attempts) if repeated[e["text"]] > 1), None)
            if record:
                items = [{"text": record["text"], "evidence_ref": {"id": record["id"], "start": 0, "end": len(record["text"])}}]
        if not items and entity.startswith("product:"):
            identifier = entity.split(":", 1)[1]
            observations = [e for e in atlas.all_evidence() if e["kind"] == "observation"]
            for record in reversed(observations[:-1]):
                start = record["text"].casefold().find(identifier.casefold())
                if start >= 0 and identifier.casefold() in {label.casefold() for label in record.get("clickables", [])}:
                    end = min(start + 100, len(record["text"]))
                    items = [{"text":record["text"][start:end],"evidence_ref":{"id":record["id"],"start":start,"end":end}}]
                    break
        if items:
            return {**action, "operation": "SUMMARIZE", "items": items}
    if reason in {"missing_evidence", "conflicting_evidence", "stalled_or_invalid_action"} and budget.retrievals:
        pending = [c for c in core_claims(atlas, working) if c["support_status"] == "pending" or c.get("conflicted")]
        for claim in pending:
            matching = [c for c in atlas.nodes("Claim") if c["entity"] == claim["entity"]
                        and c["predicate"] == claim["predicate"] and c["support_status"] == "supported"]
            if matching:
                return {**action, "operation": "RETRIEVE", "query": claim["predicate"], "predicate": claim["predicate"],
                        "entity": claim["entity"], "top_k": 2}
        feedback = working.get("interaction_feedback") or {}
        if feedback.get("evidence_ids"):
            return {**action, "operation": "RETRIEVE", "query": feedback["evidence_ids"][0], "top_k": 2}
    return action


def _build_graph_examples(public_file, output, allowed_tasks, config, teacher=None):
    adapter = graph_adapter(config["environment"].get("kind", "shopsim"))
    experience = None
    if teacher and config.get("experience_bank"):
        from .experience import ExperienceBank
        experience = ExperienceBank(config["experience_bank"], allowed_tasks)
    tools = ToolExecutor(adapter=adapter, experience=experience)
    runtime = config["runtime"]
    counts = Counter()
    allowed = set(map(str, allowed_tasks))
    for index, row in enumerate(read_jsonl(public_file)):
        task = str(row["task_id"])
        if task not in allowed:
            raise ContractError("Warmup task outside designated training partition")
        evidence = public_evidence(row)
        if not evidence or evidence[0]["kind"] != "requirement":
            raise ContractError("Requirement must precede observations")
        atlas = StateAtlas(Scope("warmup", task, f"prefix{index}"))
        goal = evidence[0]["text"]
        working = {"state_graph": True, "current_entity": "task", "public_state": {},
                   "compression_after": runtime.get("compression_after", 6),
                   "compression_min_chars": runtime.get("compression_min_chars", 1800),
                   "domain_predicates": sorted(tools.validator.domain_predicates)}
        decision = runtime.get("decision_evidence", False)
        if decision:
            working.update(decision_evidence=True, shared_memory=True)
            working["freeform_manager"] = not runtime.get("memory_choice", False)
        if runtime.get("decision_recovery"):
            working["decision_recovery"] = True
        budget = Budget(runtime.get("max_turns", 12), runtime.get("manager_episode_tokens", 1536),
                        runtime.get("retrievals", 6), runtime.get("total_tokens", 100000))
        command, previous_id, window, receipt = None, None, [], None
        try:
            for offset, raw in enumerate(evidence):
                text = assert_public_text(raw["text"])
                if raw.get("terminal") or text == TERMINAL_PUBLIC_TEXT:
                    break
                if raw.get("scope", {}).get("task", task) != task:
                    raise ContractError("Public evidence task scope mismatch")
                entity = raw.get("entity", "task")
                if entity != working["current_entity"]:
                    working.update(current_entity=entity, candidate_since=atlas.revision + 1)
                    working.pop("confirmation",None)
                record = atlas.append_public(text, raw["kind"], entity, clickables=raw.get("clickables", []))
                if record["kind"] == "action_receipt":
                    command = text
                    receipt = record
                    budget.turns = max(0, budget.turns - 1)
                    continue
                if record["kind"] == "requirement":
                    tools.execute(atlas, working, {"operation": "PROPOSE_STATE_UPDATE", "expected_revision": atlas.revision,
                        "entity": "task", "predicate": "requirement", "value": goal, "support_status": "supported",
                        "evidence_refs": [{"id": record["id"], "start": 0, "end": len(goal)}]}, "state", budget, goal)
                    if len(evidence) > 1 and evidence[1]["kind"] == "observation":
                        continue
                before = working["public_state"]
                accepted = command is not None and not text.strip().casefold().startswith("nothing happens")
                initial = previous_id is None
                after = adapter.observe(text, before, command, record.get("clickables", [])) if accepted or initial else before
                working["public_state"] = after
                progress, loop = record_progress(window, digest(adapter.progress_key(before)), digest(adapter.progress_key(after)),
                                           command, previous_id, accepted, False)
                working["interaction_feedback"] = ({"message": "Public state loop. Recheck evidence or choose a different action.",
                    "evidence_ids": loop["evidence_ids"]} if loop and runtime.get("loop_policy") == "state" else None)
                previous_id, command = record["id"], None
                working["state_lookup_used"] = False
                if accepted or initial:
                    apply_observation(tools, atlas, working, record, budget, goal)
                if decision:
                    working["observation_id"] = record["id"]
                    attempted = receipt["text"] if receipt else None
                    if accepted and receipt:
                        target = adapter.confirmation_target(attempted,entity)
                        if target:
                            subject,predicate,value = target
                            working["confirmation"] = {"entity":subject,"predicate":predicate,"value":value,
                                "attempt_id":receipt["id"],"attempt_revision":receipt["revision"]}
                        if working.get("confirmation"):
                            working["confirmation"] = confirmation_status(atlas,tools,working["confirmation"])
                    working["recovery_slots"] = list(adapter.recovery_slots(attempted)) if attempted and not accepted else []
                    receipt = None
                    if working.get("decision_recovery"):
                        if attempted:
                            record_recovery(working, {"normalized_command":attempted,
                                "action_status":"accepted" if accepted else "environment_invalid", "progress":progress,
                                "public_state_before":digest(adapter.progress_key(before)), "observation_before_id":record["id"]})
                        prepare_recovery(adapter, atlas, working, goal)
                    future = evidence[offset+1:offset+3]
                    followup = (future[0] if len(future)==2 and future[0]["kind"]=="action_receipt"
                                and future[1]["kind"]=="observation" and not future[1].get("terminal")
                                and not future[1]["text"].strip().casefold().startswith("nothing happens") else None)
                    for phase in ("state","context"):
                        example = _decision_example(atlas,working,goal,phase,budget,tools,text,followup,teacher)
                        if example:
                            counts[example["operation"]] += 1
                            append_jsonl(output,{"task_id":task,"prefix_evidence_id":record["id"],"phase":phase,**example})
                            if phase=="state" and example["decision_case"]=="historical_evidence":
                                menu = json.loads(example["messages"][-1]["content"])["verified_options"]
                                selected = menu[example["completion"]]
                                if selected.get("predicate"):
                                    presented, remaining = copy.deepcopy(working), copy.deepcopy(budget)
                                    presented["last_tool_result"] = tools.execute(atlas,presented,selected,"context",remaining,goal)
                                    variant = _decision_example(atlas,presented,goal,phase,remaining,tools,text,None,None)
                                    if variant and variant["decision_case"]=="evidence_visible":
                                        counts[variant["operation"]] += 1
                                        append_jsonl(output,{"task_id":task,"prefix_evidence_id":record["id"],"phase":phase,
                                            **variant,"controlled_visibility":"retrieved_evidence_already_in_view"})
                    continue
                for phase in ("state", "context"):
                    proposal = _graph_teacher(atlas, working, goal, phase, budget)
                    for _ in range(2 if phase == "state" else 1):
                        messages = manager_messages(atlas, text, working, goal, phase, budget)
                        annotation = teacher.annotate(atlas, working, goal, phase, budget, messages, tools,
                                                      proposal) if teacher else None
                        emit = True
                        if teacher and not annotation:
                            try:
                                teacher.verify(atlas, working, goal, phase, budget, messages, tools, proposal)
                            except ContractError:
                                teacher.counts["unobservable_rule_skipped"] += 1
                                emit = False
                        result = tools.execute(atlas, working, proposal, phase, budget, goal)
                        working["last_tool_result"] = result
                        # Teacher corrections are counterfactual labels. Keep the replayed prefix on its own path.
                        if emit:
                            label = annotation.pop("proposal") if annotation else proposal
                            counts[label["operation"]] += 1
                            append_jsonl(output, {"task_id":task, "prefix_evidence_id":record["id"], "phase":phase,
                                "messages":messages, "completion":canonical(label),
                                "source":"teacher_state_graph_public_prefix" if annotation else "state_graph_public_prefix",
                                **(annotation or {})})
                        if phase != "state" or proposal["operation"] not in {"RETRIEVE", "FETCH_EVIDENCE"}:
                            break
                        proposal = {"operation": "FINISH", "expected_revision": atlas.revision}
        finally:
            atlas.close()
    if not counts:
        raise ContractError("No checked state-graph examples produced")
    report = {"examples": sum(counts.values()), "source": "state_graph_public_prefix", "operations": dict(counts),
              "shopping_success_rate": None}
    if teacher:
        report["teacher"] = {"counts": dict(teacher.counts), "identity": getattr(teacher.generate, "identity", None)}
        if experience:
            experience.assert_unchanged()
    return report


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


def warmup(policy,examples,optimizers,output,receipt_hash,epochs=1,batch_size=1,allowed_tasks=(),resume=None,stop_after=None,
           checkpoint_every=100,graph_token_budget=(2048,192),memory_choice=False,role="manager"):
    from .checkpoint import save_checkpoint,load_checkpoint
    if role not in {"manager", "actor"}:
        raise ContractError("Unknown warmup role")
    other = "actor" if role == "manager" else "manager"
    stage = role + "_sft"
    rows=read_jsonl(examples)
    allowed=set(map(str,allowed_tasks));valid=[]
    for row in rows:
        if str(row["task_id"]) not in allowed:raise ContractError("SFT split leakage")
        if row["phase"] not in ({"state","context"} if role == "manager" else {"execute"}):
            raise ContractError("Unknown SFT role/phase")
        ids=policy.encode(row["messages"])
        choices = []
        if role == "actor":
            if row.get("source") != "teacher_actor_public_prefix" or row.get("environment_accepted") is not True or row.get("task_success") != 1:
                raise ContractError("Actor SFT requires executed demonstrations from successful training episodes")
            tokens=policy.tokenizer.encode(row["completion"],add_special_tokens=False)
            if policy.tokenizer.eos_token_id is not None:tokens.append(policy.tokenizer.eos_token_id)
            input_limit, output_limit = graph_token_budget
        elif row.get("source") in {"state_graph_candidate_choice", "state_graph_gap_choice"}:
            from .candidate_actions import LABELS
            menu = json.loads(row["messages"][-1]["content"])["verified_options"]
            labels = list(menu)
            if not memory_choice or len(labels) < 2 or labels != list(LABELS[:len(labels)]) or row["completion"] not in menu:
                raise ContractError("Invalid versioned choice SFT example")
            choices = policy.choice_ids(labels)
            tokens = [choices[labels.index(row["completion"])]]
            input_limit, output_limit = graph_token_budget
        else:
            if memory_choice:
                raise ContractError("Choice SFT requires categorical action labels")
            operation=parse_tool(row["completion"]).get("operation")
            graph_source = row.get("source") in {"state_graph_public_prefix", "teacher_state_graph_public_prefix"}
            state_ops = GRAPH_STATE_OPS if graph_source else STATE_OPS
            if operation not in (state_ops if row["phase"]=="state" else CONTEXT_OPS):raise ContractError("SFT label is not a tool in this phase")
            tokens=policy.tokenizer.encode(row["completion"],add_special_tokens=False)
            if policy.tokenizer.eos_token_id is not None:tokens.append(policy.tokenizer.eos_token_id)
            input_limit, output_limit = graph_token_budget if graph_source else (2048, 512)
        if len(ids)>input_limit or len(tokens)>output_limit:continue
        valid.append(Sample(role,row["phase"],ids,tokens,[0.]*len(tokens),row["completion"],
                            policy.versions[role],digest(row),choice_token_ids=choices,origin="demonstration"))
    if not valid:raise ContractError("No warmup samples fit the configured token budgets")
    updates=0
    batches=(len(valid)+batch_size-1)//batch_size
    total_updates=epochs*batches
    if resume:
        saved=load_checkpoint(resume,policy,optimizers,receipt_hash)
        if saved["extra"].get("stage")!=stage:raise ContractError("Warmup checkpoint role mismatch")
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
        save_checkpoint(Path(output)/"checkpoints",0,policy,optimizers,receipt_hash,{"stage":stage})
    before=policy.fingerprint(role);frozen=policy.fingerprint(other)
    completed=updates
    for epoch in range(epochs):
        for batch_index,start in enumerate(range(0,len(valid),batch_size)):
            target_step=epoch*batches+batch_index+1
            if target_step<=updates:continue
            if stop_after is not None and target_step>stop_after:break
            metrics=policy.sft_update(valid[start:start+batch_size],optimizers[role],role=role)
            completed=target_step
            append_jsonl(Path(output)/"warmup_metrics.jsonl",{"epoch":epoch,"update":completed,**metrics})
            final=completed==total_updates
            stopping=stop_after is not None and completed==stop_after
            if final:policy.reset_reference(role)
            if final or stopping or completed%checkpoint_every==0:
                path=save_checkpoint(Path(output)/"checkpoints",completed,policy,optimizers,receipt_hash,
                                    {"stage":stage,"total_updates":total_updates,"epoch":epoch})
            atomic_json(Path(output)/"status.json",{"phase":"warmup","completed_updates":completed,"expected_updates":total_updates})
    if policy.fingerprint(other)!=frozen:raise ContractError("Frozen role changed during SFT")
    latest=json.loads((Path(output)/"checkpoints/latest.json").read_text())
    path=Path(output)/"checkpoints"/latest["path"]
    summary={"updates":completed,"examples":len(valid),"filtered_over_budget":len(rows)-len(valid),
             "role":role,role+"_changed":before!=policy.fingerprint(role),other+"_unchanged":True,
             "checkpoint":str(path),"shopping_success_rate":None,"complete":completed==total_updates}
    atomic_json(Path(output)/"summary.json",summary)
    atomic_json(Path(output)/"status.json",{"phase":"complete" if summary["complete"] else "stopped_at_update_boundary",
                "completed_updates":completed,"expected_updates":total_updates})
    return summary

