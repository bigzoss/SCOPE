"""B4 global, B5 window-local, B6 phase-conditional GRPO with real continuations."""
from __future__ import annotations
import copy
import time
from dataclasses import asdict
from pathlib import Path
from .checkpoint import save_checkpoint,load_checkpoint
from .contracts import ContractError
from .credit import global_samples,local_samples,paired_local_samples
from .io import atomic_json,append_jsonl,digest,read_jsonl
from .rollout import EpisodeRunner,capture_rng,restore_rng


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


def paired_rollouts(runner, boundary, group_size, repeats, greedy, seed):
    # Sample from the current manager before isolating the downstream random streams.
    samples = [runner.sample_boundary(boundary) for _ in range(group_size)]
    outside_rng = capture_rng()
    seeds = [int(digest([seed, index])[:8], 16) for index in range(repeats)]

    def continue_with(intervention):
        outcomes = []
        for pair_seed in seeds:
            restore_rng(boundary.rng)
            outcomes.append(runner.run(boundary.scope.task, boundary.scope.episode, boundary=boundary,
                                       forced=intervention, greedy=greedy, pair_seed=pair_seed))
        return outcomes

    try:
        baselines = continue_with("FINISH")
        candidates = []
        for sample in samples:
            outcomes = continue_with(sample)
            initial = [o.events[0]["result"] for o in outcomes]
            if any(x != initial[0] for x in initial):
                raise ContractError("The pre-action intervention differs across paired seeds")
            candidates.append({"boundary_hash": boundary.fingerprint(), "sample": sample, "outcomes": outcomes,
                               "accepted": bool(initial[0]["accepted"]), "operation": initial[0].get("operation")})
        return baselines, candidates, seeds
    finally:
        restore_rng(outside_rng)


def train(policy,env,registry,config,output,receipt,experience=None,resume=None,stop_after=None):
    import torch
    output=Path(output)
    cfg=config["training"];runtime=config["runtime"]
    selector = None
    if cfg.get("teacher_probe_hints"):
        from .probe_selection import TeacherProbeSelector
        selector = TeacherProbeSelector(cfg["teacher_probe_hints"], registry.ids("rl_train"))
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
    runner=EpisodeRunner(policy,env,runtime,"train",experience,probe_selector=selector)
    updates=cfg.get("updates",48)
    mode=cfg.get("method","conditional")
    paired = cfg.get("local_credit") == "paired"
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
                    candidate_capture = phase if branch == 0 and boundary is None and (paired or index == 0) else None
                    outcome=runner.run(task,f"step{step}:task{index}",capture_phase=candidate_capture,
                                       capture_turn=cfg.get("probe_turn",1),capture_opportunity=bool(paired and candidate_capture),
                                       **({"probe_opportunities":cfg["probe_opportunities"]} if cfg.get("probe_opportunities") else {}))
                    outcomes.append(outcome)
                    for saved_boundary in outcome.boundaries:
                        saved_boundary.continuation_tasks=tasks[index+1:]
                    journal_outcome(output/"trajectories.jsonl",outcome,{"step":step,"branch":branch,"query":index,"kind":"global"})
                    if outcome.boundaries and boundary is None:
                        boundary=outcome.boundaries[0];remaining=tasks[index+1:]
                branches.append(outcomes)
            global_weighted,global_info=global_samples(branches,role)
            # In paired mode the manager learns from its intervention, not unrelated actor variation.
            if paired and role == "manager":
                global_weighted = []
            local_weighted=[];local_info=None;probe_cost={"queries":0,"tokens":0,"actor_turns":0}
            if boundary:
                if paired:
                    baselines, candidates, seeds = paired_rollouts(runner, boundary, cfg.get("local_group_size",4),
                        cfg.get("local_repeats",1), cfg.get("local_continuation","greedy")=="greedy",
                        [config.get("seed",233),step])
                    local_weighted,local_info = paired_local_samples(boundary, candidates, baselines, cfg.get("local_cost_weight",.05))
                    local_info["continuation_seeds"] = seeds
                    local_info["boundary_hash"] = boundary.fingerprint()
                    local_info["global_manager_credit_applied"] = False
                    local_info["probe_selection"] = "offline_teacher_prefix_match" if selector else "opportunity_filter"
                    local_logs = [{"outcomes": baselines, "kind": "local_baseline"}] + candidates
                    sampling = {"manager_input": sum(len(c["sample"].prompt_ids) for c in candidates),
                                "manager_output": sum(len(c["sample"].completion_ids) for c in candidates)}
                    probe_cost["candidate_sampling_tokens"] = sum(sampling.values())
                    probe_cost["actual_model_tokens"] = {"manager_input": 0, "manager_output": 0, "actor_input": 0, "actor_output": 0}
                else:
                    candidates=local_rollouts(runner,boundary,remaining,cfg.get("local_group_size",4),mode)
                    local_weighted,local_info=local_samples(boundary,candidates,mode=mode)
                    local_logs = candidates
                for i,candidate in enumerate(local_logs):
                    for j,outcome in enumerate(candidate["outcomes"]):
                        journal_outcome(output/"trajectories.jsonl",outcome,{"step":step,"branch":i,"query":j,"kind":candidate.get("kind","local")})
                        probe_cost["queries"]+=1;probe_cost["tokens"]+=outcome.costs["charged_tokens"]
                        probe_cost["actor_turns"]+=outcome.costs["actor_turns"]
                        if paired:
                            for key, value in outcome.costs["continuation_token_breakdown"].items():
                                probe_cost["actual_model_tokens"][key] += value
                if paired:
                    for key, value in sampling.items():
                        probe_cost["actual_model_tokens"][key] -= (len(seeds) - 1) * value
            elif probe:
                local_info={"skipped":"selected_boundary_not_reached_or_manager_budget_exhausted"}
            before=policy.fingerprint(role);other="actor" if role=="manager" else "manager"
            frozen=policy.fingerprint(other)
            weighted=global_weighted+[(sample,weight*cfg.get("local_weight",.25)) for sample,weight in local_weighted]
            metrics=policy.grpo_update(weighted,role,optimizers[role],cfg.get("clip",.2),cfg.get("kl_beta",.01))
            after=policy.fingerprint(role)
            if policy.fingerprint(other)!=frozen:raise ContractError("Frozen role parameters changed")
            info={"step":step,"role":role,"length":length,"global":global_info,"local":local_info,"probe_cost":probe_cost,
                  "local_credit": "paired" if paired else "legacy",
                  "task_policy_gradient": ((bool(local_weighted) and cfg.get("local_weight", .25) > 0 and not local_info["task_zero_variance"])
                                           if paired and role == "manager" else not global_info["task_zero_variance"]),
                  "kl_only_update": bool(metrics["optimizer_step"] and not any(s.advantage and w for s, w in weighted)),
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
                "updates_with_task_policy_gradient":sum(r.get("task_policy_gradient",False) for r in committed),
                "kl_only_updates":sum(r.get("kl_only_update",False) for r in committed),
                "paired_opportunity_updates":sum(bool((r.get("local") or {}).get("opportunity")) for r in committed),
                "frozen_role_unchanged":all(r["frozen_role_unchanged"] for r in committed)})
    atomic_json(output/"status.json",{"phase":"complete" if completed==updates else "stopped_at_update_boundary",
        "completed_updates":max(start,completed),"updates":updates,"complete":completed==updates})
    return optimizers
