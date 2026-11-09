"""B4 global, B5 window-local, B6 phase-conditional GRPO with real continuations."""
from __future__ import annotations
import copy
import time
from dataclasses import asdict
from pathlib import Path
from .checkpoint import save_checkpoint,load_checkpoint
from .contracts import ContractError
from .credit import global_samples,local_samples
from .io import atomic_json,append_jsonl,digest,read_jsonl
from .rollout import EpisodeRunner


def journal_outcome(path,outcome,identity):
    # Full provenance is retained; native private snapshot contents never leave the service.
    append_jsonl(path,{"identity":identity,"task_id":outcome.task_id,"metrics":outcome.metrics,
        "reward":outcome.reward,"samples":[asdict(s) for s in outcome.samples],"events":outcome.events,
        "termination_reason":outcome.termination_reason,
        "costs":outcome.costs,"atlas":outcome.final_atlas,
        "boundary_hashes":[b.fingerprint() for b in outcome.boundaries]})


def local_rollouts(runner,boundary,continuation_tasks,group_size,mode):
    candidates=[]
    # Each query creates a new fact scope; frozen experience is read-only. Independent
    # future queries cannot measure the causal utility of this operation.
    if runner.cfg.get("sequence_memory_scope","task")!="task":
        raise ContractError("Cross-query local credit requires a sequence-scoped memory implementation")
    for index in range(group_size):
        original=runner.run(boundary.scope.task,boundary.scope.episode,boundary=boundary)
        selected=[]
        for sample in original.samples:
            if sample.role=="actor":break
            if mode=="window" or sample.phase==boundary.phase:selected.append(sample)
        candidates.append({"boundary_hash":boundary.fingerprint(),"return":original.reward,
                           "selected":selected,"outcomes":[original],"credit_horizon":"current_query_only",
                           "excluded_independent_future_queries":len(continuation_tasks)})
    return candidates


def train(policy,env,registry,config,output,receipt,experience=None,resume=None,stop_after=None):
    import torch
    output=Path(output)
    cfg=config["training"];runtime=config["runtime"]
    role_mode=cfg.get("role","manager")
    optimizers={role:torch.optim.AdamW(policy.parameters(role),lr=cfg.get("lr",1e-5),weight_decay=0.)
                for role in ("manager","actor")}
    receipt_hash=digest(receipt)
    start=0
    if resume:
        saved=load_checkpoint(resume,policy,optimizers,receipt_hash)
        start=saved["step"]
        recorded=read_jsonl(output/"train_metrics.jsonl") if (output/"train_metrics.jsonl").exists() else []
        if start and start not in {r["step"] for r in recorded}:
            append_jsonl(output/"train_metrics.jsonl",saved["extra"]["metrics"])
    elif config.get("warmstart"):
        load_checkpoint(config["warmstart"],policy,restore_random=False)
        policy.reset_reference("manager");policy.reset_reference("actor")
        policy.versions={"manager":0,"actor":0}
    if not resume:
        save_checkpoint(output/"checkpoints",0,policy,optimizers,receipt_hash)
    runner=EpisodeRunner(policy,env,runtime,"train",experience)
    updates=cfg.get("updates",48)
    mode=cfg.get("method","conditional")
    if mode not in {"global","window","conditional"}:raise ContractError("Unknown GRPO method")
    group_size=cfg.get("group_size",4)
    if group_size<2:raise ContractError("GRPO needs group_size >= 2")
    schedule=cfg.get("curriculum",[{"through":16,"length":1},{"through":32,"length":2},{"through":48,"length":4}])
    probe_updates=set(cfg.get("probe_updates",[4,8,12,16,20,24,28,32,36,40,44,48]))
    for step in range(start+1,updates+1):
        if stop_after is not None and step>stop_after:break
        begin=time.perf_counter()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        role=("manager" if ((step-1)//cfg.get("alternating_window",8))%2==0 else "actor") if role_mode=="alternating" else role_mode
        if role not in optimizers:raise ContractError("Role must be manager/actor/alternating")
        length=next((p["length"] for p in schedule if step<=p["through"]),schedule[-1]["length"])
        tasks=registry.training_sequence(length)
        probe=mode!="global" and step in probe_updates and role=="manager"
        phase=("window" if mode=="window" else ("state" if sorted(probe_updates).index(step)%2==0 else "context")) if probe else None
        atomic_json(output/"status.json",{"phase":"training","step":step,"updates":updates,"role":role,"tasks":tasks})
        branches=[];boundary=None;remaining=[]
        try:
            for branch in range(group_size):
                outcomes=[]
                for index,task in enumerate(tasks):
                    outcome=runner.run(task,f"step{step}:task{index}",capture_phase=phase if branch==0 and index==0 else None,
                                       capture_turn=cfg.get("probe_turn",1))
                    outcomes.append(outcome)
                    for saved_boundary in outcome.boundaries:
                        saved_boundary.continuation_tasks=tasks[index+1:]
                    journal_outcome(output/"trajectories.jsonl",outcome,{"step":step,"branch":branch,"query":index,"kind":"global"})
                    if outcome.boundaries and boundary is None:
                        boundary=outcome.boundaries[0];remaining=tasks[index+1:]
                branches.append(outcomes)
            global_weighted,global_info=global_samples(branches,role)
            local_weighted=[];local_info=None;probe_cost={"queries":0,"tokens":0,"actor_turns":0}
            if boundary:
                candidates=local_rollouts(runner,boundary,remaining,cfg.get("local_group_size",4),mode)
                for i,candidate in enumerate(candidates):
                    for j,outcome in enumerate(candidate["outcomes"]):
                        journal_outcome(output/"trajectories.jsonl",outcome,{"step":step,"branch":i,"query":j,"kind":"local"})
                        probe_cost["queries"]+=1;probe_cost["tokens"]+=outcome.costs["charged_tokens"]
                        probe_cost["actor_turns"]+=outcome.costs["actor_turns"]
                local_weighted,local_info=local_samples(boundary,candidates,mode=mode)
            elif probe:
                local_info={"skipped":"selected_boundary_not_reached_or_manager_budget_exhausted"}
            before=policy.fingerprint(role);other="actor" if role=="manager" else "manager"
            frozen=policy.fingerprint(other)
            weighted=global_weighted+[(sample,weight*cfg.get("local_weight",.25)) for sample,weight in local_weighted]
            metrics=policy.grpo_update(weighted,role,optimizers[role],cfg.get("clip",.2),cfg.get("kl_beta",.01))
            after=policy.fingerprint(role)
            if policy.fingerprint(other)!=frozen:raise ContractError("Frozen role parameters changed")
            info={"step":step,"role":role,"length":length,"global":global_info,"local":local_info,"probe_cost":probe_cost,
                  "global_queries":sum(map(len,branches)),"parameter_hash_before":before,"parameter_hash_after":after,
                  "parameters_changed":before!=after,"frozen_role_unchanged":True,
                  "elapsed_wall_seconds":time.perf_counter()-begin,
                  "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,**metrics}
            if experience:experience.assert_unchanged()
            checkpoint=save_checkpoint(output/"checkpoints",step,policy,optimizers,receipt_hash,
                                       {"tasks":tasks,"metrics":info,"atlas_snapshots":[o.final_atlas for o in branches[0]]})
            append_jsonl(output/"train_metrics.jsonl",info)
            atomic_json(output/"status.json",{"phase":"training","completed_updates":step,"updates":updates,"checkpoint":str(checkpoint)})
        finally:
            if boundary:env.drop_snapshot(boundary.environment)
    completed=min(updates,stop_after) if stop_after is not None else updates
    committed=read_jsonl(output/"train_metrics.jsonl") if (output/"train_metrics.jsonl").exists() else []
    if len({r["step"] for r in committed})!=len(committed):raise ContractError("Duplicate committed update metrics")
    atomic_json(output/"train_summary.json",{"complete":completed==updates,"completed_updates":max(start,completed),
                "method":mode,"global_queries":sum(r["global_queries"] for r in committed),
                "probe_queries":sum(r["probe_cost"]["queries"] for r in committed),
                "updates_with_parameter_change":sum(r["parameters_changed"] for r in committed),
                "frozen_role_unchanged":all(r["frozen_role_unchanged"] for r in committed)})
    atomic_json(output/"status.json",{"phase":"complete" if completed==updates else "stopped_at_update_boundary",
        "completed_updates":max(start,completed),"updates":updates,"complete":completed==updates})
    return optimizers
