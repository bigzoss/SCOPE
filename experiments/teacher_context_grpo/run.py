"""Source-only actor distillation followed by student/teacher manager GRPO."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from data import prepare_examples
from experiments.memory_curriculum import full_hotpot_grpo as full
from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.contracts import ContractError
from scope.hotpot import split_row
from scope.io import atomic_json, digest, file_sha
from scope.warmup import warmup
from train_speed_policy import TrainingSpeedPolicy


def read(path):
    return json.loads(path.read_text())


def main(args):
    output = args.output
    parent = args.parent
    sys.path.insert(0, str(parent))
    previous = importlib.import_module("run_validation")
    started = time.monotonic()
    original = read(args.base/"train/receipt.json")
    parent_receipt = read(parent/"receipt.json")
    if file_sha(args.base/"train/receipt.json") != parent_receipt["base_receipt_sha256"]:
        raise ContractError("Parent experiment used a different base receipt")
    for name, sha in parent_receipt["source_sha256"].items():
        if file_sha(parent/name) != sha:
            raise ContractError("Parent training implementation changed")
    initial = parent/"initial"
    if file_sha(initial/"state.pt") != parent_receipt["base_checkpoint_sha256"]:
        raise ContractError("Initial checkpoint changed")
    rows, dev, manifest = full.prepare(args.base/"dataset")
    if any(original.get(key) != value for key,value in manifest.items()):
        raise ContractError("Original training manifest changed")
    by_id = {row["id"]:row for row in rows}
    tasks = [split_row(by_id[task_id]) for task_id in parent_receipt["train_ids"]]
    del rows, by_id
    if [p["id"] for p,_ in dev] != parent_receipt["dev_ids"]:
        raise ContractError("Development IDs differ from the baseline")
    task_map = {p["id"]:(p,g) for p,g in tasks}
    data = prepare_examples(parent, task_map, set(parent_receipt["dev_ids"]), output)
    receipt = {
        "initial_checkpoint_sha256":file_sha(initial/"state.pt"),
        "initial_step":parent_receipt["base_checkpoint_step"],"parent_receipt_sha256":file_sha(parent/"receipt.json"),
        "train_ids":parent_receipt["train_ids"],"dev_ids":parent_receipt["dev_ids"],"teacher_data":data,
        "actor_sft":{"epochs":3,"batch_size":2,"lr":1e-5,"max_input_tokens":4096,"max_output_tokens":24,
                     "resume_test_after_first_update":True,"selection_uses_dev":False},
        "manager_training":{"questions":len(tasks),"arms":["actor_student","actor_teacher"],
                            "lr":1e-5,"executor":"frozen after SFT",
                            "protocol":"same parent local decisions and strict on-policy GRPO",
                            "teacher_auxiliary":"separate optimizer with evidence intervention and improvement check"},
        "source_sha256":{name:file_sha(Path(__file__).parent/name) for name in ("run.py","data.py")},
        "gpu":4,"model":parent_receipt["model"],
    }
    if (output/"receipt.json").exists():
        raise ContractError("Use a new output for this fixed experiment")
    atomic_json(output/"receipt.json",receipt)
    identity = digest(receipt)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    random.seed(full.SEED)
    np.random.seed(full.SEED)
    torch.manual_seed(full.SEED)
    free,total = torch.cuda.mem_get_info()
    if free < 24*2**30:
        raise RuntimeError("GPU4 requires 24 GiB free")
    torch.cuda.set_per_process_memory_fraction(min(1., 26*2**30/total))
    TrainingSpeedPolicy.fast_update = True
    policy = TrainingSpeedPolicy(receipt["model"])
    manager_optimizer = torch.optim.AdamW(policy.parameters("manager"),lr=1e-5,weight_decay=0.)
    actor_optimizer = torch.optim.AdamW(policy.parameters("actor"),lr=1e-5,weight_decay=0.)
    load_checkpoint(initial,policy,{"manager":manager_optimizer},digest(original))
    manager_before = policy.fingerprint("manager")
    actor_before = policy.fingerprint("actor")
    atomic_json(output/"status.json",{"phase":"actor_sft","accepted_examples":data["accepted"]})
    kwargs = dict(epochs=3,batch_size=2,allowed_tasks=receipt["train_ids"],role="actor",
                  graph_token_budget=(4096,24),checkpoint_every=1)
    optimizers = {"actor":actor_optimizer}
    first = warmup(policy,output/"actor_examples.jsonl",optimizers,output/"actor_sft",identity,stop_after=1,**kwargs)
    first_fingerprint = policy.fingerprint()
    saved = load_checkpoint(Path(first["checkpoint"]),policy,optimizers,identity)
    if policy.fingerprint()!=first_fingerprint or saved["step"]!=1:
        raise ContractError("Actor SFT checkpoint failed exact restoration")
    atomic_json(output/"actor_resume_validation.json",{"passed":True,"restored_step":1,
                                                       "optimizer_states":len(actor_optimizer.state)})
    trained = warmup(policy,output/"actor_examples.jsonl",optimizers,output/"actor_sft",identity,
                     resume=Path(first["checkpoint"]),**kwargs)
    if not trained["complete"] or trained["filtered_over_budget"] or policy.fingerprint("manager")!=manager_before:
        raise ContractError("Actor SFT did not complete its exact dataset or changed the manager")
    if policy.fingerprint("actor")==actor_before:
        raise ContractError("Actor SFT did not change parameters")
    actor_checkpoint = Path(trained["checkpoint"])
    atomic_json(output/"status.json",{"phase":"actor_only_evaluation","actor_updates":trained["updates"]})
    actor_only = previous.frozen_eval(policy,dev,output,"actor_only")
    results = {"actor_only":actor_only}
    # Reuse complete baseline results with identical initialization, IDs and source.
    for old,new in (("initial","initial"),("student","manager_student"),("teacher","manager_teacher")):
        results[new] = [json.loads(line) for line in (parent/old/"dev.jsonl").read_text().splitlines()]
        if [row["task_id"] for row in results[new]] != receipt["dev_ids"]:
            raise ContractError("Cached baseline development coverage differs")
    teacher = previous.ProposalTeacher(args.config,output/"teacher_cache")
    counts = {}
    try:
        for arm in ("actor_student","actor_teacher"):
            load_checkpoint(initial,policy,{"manager":manager_optimizer},digest(original))
            load_checkpoint(actor_checkpoint,policy,receipt_hash=identity,restore_random=False)
            random.seed(full.SEED)
            np.random.seed(full.SEED)
            torch.manual_seed(full.SEED)
            actor = policy.fingerprint("actor")
            if policy.fingerprint("manager") != manager_before:
                raise ContractError("Manager branch initial parameters differ")
            counter = Counter()
            arm_rows = previous.run_arm(
                policy,manager_optimizer,teacher if arm=="actor_teacher" else None,tasks,
                receipt["initial_step"],output,arm,counter,actor,policy.frozen_base_signature())
            if [r["task_id"] for r in arm_rows] != receipt["train_ids"]:
                raise ContractError("Branch training coverage differs")
            counts[arm] = dict(counter)
            save_checkpoint(output/arm/"checkpoints",receipt["initial_step"]+len(tasks),policy,
                            {"manager":manager_optimizer},identity)
            atomic_json(output/"status.json",{"phase":"dev_evaluation","arm":arm,"counts":counts})
            results[arm] = previous.frozen_eval(policy,dev,output,arm)
    finally:
        teacher.close()
    summary = {
        "complete":True,"actor_data":data,"actor_sft":trained,"manager_counts":counts,
        "dev":{arm:{"n":len(items),"em_count":sum(x["em"] for x in items),
                    "em":float(np.mean([x["em"] for x in items])),
                    "f1":float(np.mean([x["f1"] for x in items]))} for arm,items in results.items()},
        "paired":{
            "actor_only_vs_initial":previous.paired(results["initial"],results["actor_only"]),
            "actor_teacher_vs_manager_teacher":previous.paired(results["manager_teacher"],results["actor_teacher"]),
            "teacher_effect_after_actor_sft":previous.paired(results["actor_student"],results["actor_teacher"]),
        },
        "api":dict(teacher.counts),"elapsed_seconds":time.monotonic()-started,
        "interpretation":"source development pilot; eight selected successful teacher answers; no target-domain updates",
    }
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"status.json",{"phase":"complete",**summary})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for key in ("base","parent","output","config"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        main(args)
    except Exception:
        atomic_json(args.output/"error.json",{"traceback":traceback.format_exc()})
        raise
