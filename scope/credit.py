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

