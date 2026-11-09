"""Frozen paired evaluation, atomic sequence units and task/receipt identity checks."""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from .contracts import ContractError
from .io import atomic_json,atomic_jsonl,append_jsonl,canonical,digest,file_sha,read_jsonl
from .rollout import EpisodeRunner
from .config import DEFAULT_RUNTIME
from .interaction import TRACE_VERSION

_BUDGET_FIELDS=("max_turns","actor_max_input","actor_max_new","manager_max_input","manager_max_new",
                "manager_episode_tokens","total_tokens","retrievals")


def evaluation_contract(config,split,sequences):
    return {"split":split,"sequences_sha256":digest(sequences),"information_protocol":"full_instruction",
            "benchmark":config.get("environment",{}).get("kind","shopsim"),
            "public_interaction":{"trace_version":TRACE_VERSION, **{
                k:config["runtime"].get(k,DEFAULT_RUNTIME[k]) for k in ("action_mode","loop_policy")}},
            "generation_budget":{k:config["runtime"].get(k,DEFAULT_RUNTIME[k]) for k in _BUDGET_FIELDS}}


def _check_records(records,tasks,sequence_id):
    if [r.get("task_id") for r in records]!=tasks or any(r.get("sequence_id")!=sequence_id for r in records):
        raise ContractError("Cached evaluation sequence records differ from their task unit")
    for record in records:
        m=record["metrics"]
        if any(not math.isfinite(float(m[k])) or not 0<=float(m[k])<=1 for k in ("success","r_loose","r_strict","purchased")):
            raise ContractError("Invalid evaluation metrics")
        if m["success"] not in (0,1):raise ContractError("Success must be binary")


def evaluate(policy,env,registry,config,output,split="dev",experience=None,limit=None):
    output=Path(output)
    sequences=registry.evaluation_sequences(split,config.get("evaluation",{}).get("sequence_length",2),
                                           config.get("evaluation",{}).get("seed",233))
    if limit is not None:
        # Reduced evaluation is explicitly a subset, never labelled a full 300-task result.
        chosen=set(registry.ids(split)[:limit])
        sequences=[[t for t in sequence if t in chosen] for sequence in sequences]
        sequences=[s for s in sequences if s]
    expected=[task for sequence in sequences for task in sequence]
    if not expected:raise ContractError("Evaluation task set is empty")
    if len(expected)!=len(set(expected)):raise ContractError("Duplicate IDs in evaluation receipt")
    fixed_hash=policy.fingerprint()
    base_before=policy.frozen_base_signature() if hasattr(policy,"frozen_base_signature") else None
    contract=evaluation_contract(config,split,sequences)
    runner=EpisodeRunner(policy,env,config["runtime"],split,experience)
    records=[];sequence_ids=[]
    for index,tasks in enumerate(sequences):
        path=output/"sequences"/f"sequence_{index:05d}.json"
        if path.exists():
            saved=json.loads(path.read_text())
            if saved["tasks"]!=tasks or saved["parameter_hash"]!=fixed_hash or saved.get("contract_sha256")!=digest(contract):
                raise ContractError("Evaluation resume identity mismatch")
            unit=saved["records"]
            if digest(unit)!=saved.get("records_sha256"):raise ContractError("Cached evaluation records checksum mismatch")
            _check_records(unit,tasks,index)
        else:
            unit=[]
            for task in tasks:
                outcome=runner.run(task,f"eval:{index}:{task}",greedy=True)
                actor=[s for s in outcome.samples if s.role=="actor"]
                steps=[e for e in outcome.events if e["phase"]=="execute"]
                unit.append({"task_id":task,"sequence_id":index,"metrics":outcome.metrics,"costs":outcome.costs,
                    "actor_turns":len(actor),"length_terminations":sum(s.termination=="length" for s in actor),
                    "illegal_actions":sum(bool(e["is_error"]) for e in steps),
                    "termination_reason":outcome.termination_reason,
                    "history":[e if e["phase"]=="execute" else {"phase":e["phase"],"command":None,
                               "is_error":None,"view_hash":None} for e in outcome.events],
                    "public_evidence":[json.loads(r[1]) for r in outcome.final_atlas["tables"]["evidence"]],
                    "manager_operations":[e for e in outcome.events if e["phase"]!="execute"],
                    "initial_memory":"empty_task_facts_plus_frozen_training_experience",
                    "final_atlas_hash":digest(outcome.final_atlas)})
            _check_records(unit,tasks,index)
            atomic_json(path,{"tasks":tasks,"parameter_hash":fixed_hash,"records":unit,
                            "records_sha256":digest(unit),"contract_sha256":digest(contract)})
        records.extend(unit)
        atomic_json(output/"status.json",{"phase":"evaluating","completed":len(records),"expected":len(expected)})
    if policy.fingerprint()!=fixed_hash:raise ContractError("Parameters changed during evaluation")
    if base_before is not None and policy.frozen_base_signature()!=base_before:raise ContractError("Frozen base parameters changed during evaluation")
    if experience:experience.assert_unchanged()
    ids=[r["task_id"] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ContractError("Evaluation task completeness mismatch")
    result=output/"results.jsonl"
    atomic_jsonl(result,records)
    turns=sum(r["actor_turns"] for r in records)
    steps=[e for r in records for e in r["history"] if e["phase"]=="execute"]
    action_counts=Counter(e["action_status"] for e in steps)
    progress_counts=Counter(e["progress"] for e in steps)
    n=len(records)
    summary={"complete":True,"n":n,"expected":len(expected),"split":split,"subset":limit is not None,
             "success_count":sum(int(r["metrics"]["success"]==1) for r in records),
             "strict_success_rate":sum(r["metrics"]["success"] for r in records)/max(1,n),
             "mean_loose_reward":sum(r["metrics"]["r_loose"] for r in records)/max(1,n),
             "mean_strict_reward":sum(r["metrics"]["r_strict"] for r in records)/max(1,n),
             "wrong_purchase_rate":sum(r["metrics"]["purchased"] and not r["metrics"]["success"] for r in records)/max(1,n),
             "illegal_action_rate_per_turn":sum(r["illegal_actions"] for r in records)/max(1,turns),
             "length_termination_rate_per_turn":sum(r["length_terminations"] for r in records)/max(1,turns),
             "public_interaction":{"format_errors":action_counts["format_error"],
                 "binding_failures":action_counts["binding_failed"],"environment_invalid":action_counts["environment_invalid"],
                 "legal_no_progress":progress_counts["legal_no_progress"],"public_state_loops":sum(bool(e["loop"]) for e in steps),
                 "episode_termination_reasons":dict(Counter(r["termination_reason"] for r in records))},
             "mean_total_tokens":sum(r["costs"]["charged_tokens"] for r in records)/max(1,n),
             "parameters_unchanged_during_eval":True,"memory_unchanged":True,
             "frozen_base_parameters_unchanged":True if base_before is not None else None,"evaluation_contract":contract,
             "results_sha256":file_sha(result),"task_ids_sha256":digest(sorted(ids))}
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"status.json",{"phase":"complete","completed":n,"expected":n})
    return summary


def paired_comparison(left,right,repeats=20000,seed=233,allow_interaction_contrast=False):
    if repeats<1:raise ContractError("Bootstrap repeats must be positive")
    def read(path):
        path=Path(path);summary=json.loads((path/"summary.json").read_text())
        rows=read_jsonl(path/"results.jsonl");ids=[r["task_id"] for r in rows]
        if len(ids)!=len(set(ids)):raise ContractError("Duplicate task IDs in paired results")
        if not rows or not summary.get("complete") or summary["n"]!=len(rows) or summary.get("expected",len(rows))!=len(rows):
            raise ContractError("Incomplete paired evaluation")
        if summary["results_sha256"]!=file_sha(path/"results.jsonl"):raise ContractError("Paired results checksum mismatch")
        if any(type(r["metrics"]["success"]) not in (int,float) or r["metrics"]["success"] not in (0,1) for r in rows):
            raise ContractError("Paired success must be binary")
        contract=summary.get("evaluation_contract")
        if contract is None and (path/"receipt.json").exists():
            old=json.loads((path/"receipt.json").read_text())
            contract={"split":summary["split"],"information_protocol":old["information_protocol"],
                      "generation_budget":{k:old["config"]["runtime"].get(k) for k in _BUDGET_FIELDS},
                      "benchmark":old["config"]["environment"].get("kind","shopsim")}
        if contract is None:raise ContractError("Evaluation protocol identity is missing")
        return {r["task_id"]:r for r in rows},contract
    a,ac=read(left);b,bc=read(right)
    contrast = None
    if ac != bc:
        left_interaction, right_interaction = ac.get("public_interaction", {}), bc.get("public_interaction", {})
        modes = {"action_mode", "loop_policy"}
        same_other_fields = {k: v for k, v in ac.items() if k != "public_interaction"} == {
            k: v for k, v in bc.items() if k != "public_interaction"}
        same_trace = left_interaction.get("trace_version") == right_interaction.get("trace_version") == TRACE_VERSION
        if not (allow_interaction_contrast and same_other_fields and same_trace
                and {k: v for k, v in left_interaction.items() if k not in modes}
                == {k: v for k, v in right_interaction.items() if k not in modes}):
            raise ContractError("Paired evaluation protocols/budgets differ")
        contrast = {"left": left_interaction, "right": right_interaction}
    if set(a)!=set(b):raise ContractError("Paired evaluations have different task IDs")
    clusters={}
    for task,row in a.items():
        if row["sequence_id"]!=b[task]["sequence_id"]:raise ContractError("Different evaluation sequence assignment")
        clusters.setdefault(row["sequence_id"],[]).append(b[task]["metrics"]["success"]-row["metrics"]["success"])
    sums=np.asarray([sum(v) for v in clusters.values()]);sizes=np.asarray([len(v) for v in clusters.values()])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        selected=rng.integers(0,len(sums),len(sums))
        values.append(sums[selected].sum()/sizes[selected].sum())
    result = {"n":len(a),"clusters":len(clusters),"difference":float(sums.sum()/sizes.sum()),
            "ci95":np.quantile(values,[.025,.975]).tolist(),"bootstrap_repeats":repeats,
            "interpretation":"task-sequence uncertainty; training-seed variation must be reported separately"}
    if contrast:
        result["interaction_contrast"] = contrast
    return result
