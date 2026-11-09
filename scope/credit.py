"""Global trajectory and conditional local group advantages, with explicit ownership."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .contracts import ContractError


def standardize(values):
    array=np.asarray(values,dtype=np.float64)
    if not np.isfinite(array).all():raise FloatingPointError("Non-finite group return")
    return ((array-array.mean())/(array.std()+1e-8)).tolist() if len(array)>1 and array.std()>1e-8 else [0.]*len(array)


def global_samples(branches,role):
    advantages=standardize([sum(o.reward for o in branch)/len(branch) for branch in branches])
    weighted=[]
    for branch,adv in zip(branches,advantages):
        samples=[s for outcome in branch for s in outcome.samples if s.role==role]
        total=sum(len(s.completion_ids) for s in samples)
        for sample in samples:
            sample.advantage=adv;sample.credit_scope="global"
            weighted.append((sample,len(sample.completion_ids)/max(1,total)/len(branches)))
    task_values=[sum((o.metrics["success"]+.1*o.metrics["r_loose"])/1.1 for o in b)/len(b) for b in branches]
    task_constant=bool(np.std(task_values)<=1e-8)
    return weighted,{"returns":[sum(o.reward for o in b)/len(b) for b in branches],"advantages":advantages,
                     "zero_variance":all(a==0 for a in advantages),"task_returns":task_values,"task_zero_variance":task_constant,
                     "non_task_only_advantage":task_constant and any(a!=0 for a in advantages)}


def local_samples(boundary,candidates,role="manager",mode="conditional"):
    """Each candidate has a shared *pre-action* boundary. Post-action states may differ."""
    if len(candidates)<2:raise ContractError("A local GRPO group needs at least two sampled candidates")
    if any(c["boundary_hash"]!=boundary.fingerprint() for c in candidates):
        raise ContractError("Local candidates were generated at different boundaries")
    if not any(c["selected"] for c in candidates):return [],{"skipped":"no_sampled_manager_action"}
    if any(not c["selected"] for c in candidates):raise ContractError("Incomplete local action group")
    anchor=candidates[0]["selected"][0]
    if any(c["selected"][0].prompt_ids!=anchor.prompt_ids or c["selected"][0].version!=anchor.version for c in candidates):
        raise ContractError("Local candidates have different pre-action contexts or policy versions")
    adv=standardize([c["return"] for c in candidates])
    weighted=[]
    for candidate,a in zip(candidates,adv):
        selected=candidate["selected"]
        if not selected:continue
        for sample in selected:
            if sample.role!=role or (mode=="conditional" and sample.phase!=boundary.phase):
                raise ContractError("Credit cannot cross role or phase boundaries")
            sample.advantage=a;sample.credit_scope="local:"+boundary.phase
            weighted.append((sample,1/len(candidates)/len(selected)))
    task_values=[sum((o.metrics["success"]+.1*o.metrics["r_loose"])/1.1 for o in c["outcomes"])/len(c["outcomes"])
                 for c in candidates] if all("outcomes" in c for c in candidates) else None
    return weighted,{"returns":[c["return"] for c in candidates],"advantages":adv,
                     "phase":boundary.phase,"zero_variance":all(a==0 for a in adv),"task_returns":task_values,
                     "task_zero_variance":bool(np.std(task_values)<=1e-8) if task_values else None,
                     "non_task_only_advantage":bool(np.std(task_values)<=1e-8 and any(a!=0 for a in adv)) if task_values else None}


def interaction(q00,q01,q10,q11):
    return {"context_at_original":q01-q00,"context_at_revised":q11-q10,
            "interaction":q11-q10-q01+q00}


def paired_local_samples(boundary, candidates, baselines, cost_weight):
    """Strict task utility is paired by continuation seed. Compression is a bounded tie-breaker."""
    if not baselines or any(len(c["outcomes"]) != len(baselines) for c in candidates):
        raise ContractError("Every local candidate requires the same continuation seeds as FINISH")
    if any(c["boundary_hash"] != boundary.fingerprint() for c in candidates):
        raise ContractError("Paired candidates crossed pre-action boundaries")
    if any(c["sample"].role != "manager" or c["sample"].phase != boundary.phase or
           c["sample"].version != boundary.role_versions["manager"] or c["sample"].origin != "policy" for c in candidates):
        raise ContractError("Credit cannot cross role, version, or phase boundaries")
    if candidates and any(c["sample"].prompt_ids != candidates[0]["sample"].prompt_ids for c in candidates):
        raise ContractError("Paired candidates saw different pre-action prompts")
    baseline_success = [int(o.metrics["success"]) for o in baselines]
    valid = [i for i, c in enumerate(candidates) if c["accepted"]]
    returns, savings, strict_values, retention = [], [], [], []

    def preserves(candidate, baseline):
        left = next((e for e in baseline.events if e["phase"] == "execute"), None)
        right = next((e for e in candidate.events if e["phase"] == "execute"), None)
        return left is not None and right is not None and set(left["required_evidence_ids"]) <= set(right["context_evidence_ids"])

    for c in candidates:
        strict = [int(o.metrics["success"]) for o in c["outcomes"]]
        strict_values.append(strict)
        returns.append(float(np.mean([a - b for a, b in zip(strict, baseline_success)])))
        kept = all(preserves(o, b) for o, b in zip(c["outcomes"], baselines))
        retention.append(kept)
        if c["operation"] == "SUMMARIZE" and kept and all(strict) and all(baseline_success):
            saving = float(np.mean([(b.costs["charged_tokens"] - o.costs["charged_tokens"]) /
                                   max(1, b.costs["charged_tokens"]) for o, b in zip(c["outcomes"], baselines)]))
        else:
            saving = 0.
        savings.append(max(-1., min(1., saving)))
    centered = standardize([returns[i] for i in valid])
    task_zero_variance = not any(centered)
    weighted, advantages = [], [0.] * len(candidates)
    for index, advantage in zip(valid, centered):
        # Small successful compression differences must not be amplified by normalization.
        if task_zero_variance and len(valid) >= 2:
            advantage = cost_weight * savings[index]
        sample = candidates[index]["sample"]
        sample.advantage, sample.credit_scope = advantage, "paired:" + boundary.phase
        advantages[index] = advantage
        if advantage:
            weighted.append((sample, 1 / len(valid)))
    return weighted, {"phase": boundary.phase, "opportunity": boundary.opportunity,
        "baseline_strict": baseline_success, "candidate_strict": strict_values, "strict_deltas": returns,
        "advantages": advantages, "valid_candidates": len(valid), "operations": [c["operation"] for c in candidates],
        "task_zero_variance": task_zero_variance, "zero_variance": not any(advantages),
        "non_task_only_advantage": task_zero_variance and any(advantages),
        "baseline_success_count":sum(baseline_success),
        "candidate_success_count":sum(sum(values) for values in strict_values),
        "positive_effect_candidates":sum(returns[i] > 0 for i in valid),
        "negative_effect_candidates":sum(returns[i] < 0 for i in valid),
        "all_branches_failed":not any(baseline_success) and not any(any(values) for values in strict_values),
        "candidate_origin":"student_policy", "teacher_scores_used_as_reward":False,
        "required_evidence_preserved": retention, "compression_token_savings_fraction": savings}

