"""Fixed 10k source experiment: teacher repairs, paired credit, frozen actor.

Run from an immutable code snapshot. The prepared receipt owns all task IDs,
model identities and budgets. No development-based checkpoint selection.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import gzip
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.memory_curriculum.hotpot_grpo import step_data
from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.contracts import ContractError, Sample
from scope.hotpot import rollout, score, split_row
from scope.io import atomic_json, atomic_jsonl, digest, file_sha
from scope.teacher_api import APITeacher
from train_speed_policy import TrainingSpeedPolicy


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(root, receipt):
    for name, sha in receipt["source_sha256"].items():
        if file_sha(root/"source"/name) != sha:
            raise ContractError("Pinned source changed: " + name)
    for name in ("train", "dev"):
        if file_sha(root/receipt[name]["file"]) != receipt[name]["sha256"]:
            raise ContractError("Prepared split changed")
    if file_sha(Path(receipt["warmstart"])/"state.pt") != receipt["warmstart_sha256"]:
        raise ContractError("Initialization checkpoint changed")


def checkpoint(root, step, policy, optimizers, identity, extra):
    path = save_checkpoint(root/"checkpoints",step,policy,optimizers,identity,extra)
    # Only files owned by this run: retain initial, real resume check, 1k
    # milestones and the latest two regular saves. Never touch another run.
    checkpoint_root = (root/"checkpoints").resolve()
    for old in checkpoint_root.glob("step_*"):
        number = int(old.name.split("_")[-1])
        if number in {0,8} or number % 1000 == 0 or number >= step-100:
            continue
        if old.is_symlink() or old.resolve().parent != checkpoint_root:
            raise ContractError("Unsafe checkpoint retention path")
        if read(old/"metadata.json")["receipt_hash"] != identity:
            raise ContractError("Checkpoint retention ownership mismatch")
        shutil.rmtree(old)
    return path


def evaluate(root, policy, tasks, name, options, actor):
    path = root/f"dev_{name}.jsonl"
    parameters = policy.fingerprint("manager")
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if [r["task_id"] for r in rows] != [p["id"] for p,_ in tasks[:len(rows)]]:
        raise ContractError("Development journal identity mismatch")
    if any(r["manager_sha256"] != parameters or r["actor_sha256"] != actor for r in rows):
        raise ContractError("Development journal uses another model")
    for public,gold in tasks[len(rows):]:
        with policy.task_cache():
            trace,_ = rollout(policy,public,"validation",233,greedy=True,**options)
        rows.append({"task_id":public["id"],**score(trace,gold),"answer":trace["answer"],
                     "view_sha256":digest(trace["answer_view"]),"memory_flow":trace["memory_flow"],
                     "manager_sha256":parameters,"actor_sha256":actor})
        atomic_jsonl(path,rows)
        atomic_json(root/"status.json",{"phase":"evaluate_"+name,"completed":len(rows),"expected":len(tasks)})
    if policy.fingerprint("manager") != parameters or policy.fingerprint("actor") != actor:
        raise ContractError("Evaluation changed parameters")
    return {"n":len(rows),"em_count":sum(r["em"] for r in rows),
            "f1":sum(r["f1"] for r in rows)/len(rows),"parameters_unchanged":True,
            "independent_empty_memory_per_task":True,"sha256":file_sha(path)}


def paired(root):
    before=[json.loads(l) for l in (root/"dev_initial.jsonl").read_text().splitlines()]
    after=[json.loads(l) for l in (root/"dev_final.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in before] == [r["task_id"] for r in after]
    deltas=np.array([b["em"]-a["em"] for a,b in zip(before,after,strict=True)])
    rng=np.random.default_rng(233)
    boot=rng.choice(deltas,size=(5000,len(deltas)),replace=True).mean(axis=1)
    return {"delta_em":float(deltas.mean()),"wins":int((deltas>0).sum()),"losses":int((deltas<0).sum()),
            "bootstrap_95":[float(x) for x in np.quantile(boot,[.025,.975])]}


def execute(root, config, resume, stop_after, target=10000):
    receipt=read(root/"receipt.json");identity=digest(receipt)
    teacher_protocol=receipt.get("teacher_protocol","legacy")
    teacher_selection=receipt.get("teacher_selection","failed_public_gap" if teacher_protocol=="evidence_v2" else "periodic")
    if teacher_protocol not in {"legacy","evidence_v2"} or teacher_selection not in {"periodic","failed_public_gap"}:
        raise ContractError("Unknown teacher protocol/selection")
    if teacher_selection=="failed_public_gap" and teacher_protocol!="evidence_v2":
        raise ContractError("Evidence-gap scheduling requires evidence_v2")
    verify_inputs(root,receipt)
    tasks={name:[split_row(row) for row in pq.read_table(root/receipt[name]["file"]).to_pylist()]
           for name in ("train","dev")}
    for name in tasks:
        assert [p["id"] for p,_ in tasks[name]] == receipt[name]["ids"]
    assert len(tasks["train"])==10000 and len(tasks["dev"])==512
    if type(target) is not int or not 1 <= target <= len(tasks["train"]):
        raise ContractError("Training target must be a nonempty prefix of the fixed manifest")
    if stop_after is not None and not 1 <= stop_after <= target:
        raise ContractError("Stop boundary must be within the requested target")
    random.seed(233);np.random.seed(233);torch.manual_seed(233);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,26*2**30/total))
    policy=TrainingSpeedPolicy(Path(receipt["model"]));policy.fast_update=True
    optimizers={"manager":torch.optim.AdamW(policy.parameters("manager"),lr=1e-5,weight_decay=0.),
                "teacher_manager":torch.optim.AdamW(policy.parameters("manager"),lr=1e-6,weight_decay=0.)}
    options=receipt["rollout_options"]
    start=0;counts=Counter();train_seconds=0.
    if resume:
        path=root/"checkpoints"/read(root/"checkpoints/latest.json")["path"]
        saved=load_checkpoint(path,policy,optimizers,identity)
        start=saved["step"];counts.update(saved["extra"]["counts"]);train_seconds=saved["extra"]["train_seconds"]
        atomic_json(root/"resume_validation.json",{"passed":True,"restored_step":start,
            "manager_sha256":policy.fingerprint("manager"),"actor_sha256":policy.fingerprint("actor"),
            "optimizer_state_counts":{k:len(v.state) for k,v in optimizers.items()},"rng_restored":True,
            "checkpoint_sha256":file_sha(path/"state.pt")})
        if start > target:
            raise ContractError("Checkpoint is beyond the requested training target")
    else:
        load_checkpoint(Path(receipt["warmstart"]),policy,restore_random=False)
        policy.reset_reference("manager")
        checkpoint(root,0,policy,optimizers,identity,{"counts":{},"train_seconds":0.})
    actor=policy.fingerprint("actor");frozen_base=policy.frozen_base_signature()
    if start==0:
        initial=evaluate(root,policy,tasks["dev"],"initial",options,actor)
        atomic_json(root/"initial_summary.json",initial)
    teacher_type=APITeacher
    if teacher_protocol=="evidence_v2":
        from scope.teacher_rate_limit import RateLimitedTeacher
        teacher_type=RateLimitedTeacher
    teacher=teacher_type(config,root/"teacher_cache",max_requests=receipt.get("max_teacher_http_requests",1200),
                       model=receipt["teacher"],allow_unusable=True)
    if (root/"teacher_api_usage.json").exists():teacher.counts.update(read(root/"teacher_api_usage.json"))
    session_start=time.monotonic();session_done=start
    limit=stop_after or target
    try:
        for index in range(start+1,limit+1):
            began=time.monotonic()
            atomic_json(root/"status.json",{"phase":"training","completed":index-1,"expected":target,
                "durable_step":read(root/"checkpoints/latest.json")["step"],"counts":dict(counts),
                "teacher_api":dict(teacher.counts),"train_seconds":train_seconds})
            # One local question per block of ten; rotate the odd position so
            # the five stage preferences are all covered (every tenth alone
            # would always select the same preferred stage).
            use_teacher=(index%2==1 if teacher_selection=="failed_public_gap" else
                         (index-1)%10 == 2*(((index-1)//10)%5))
            with policy.task_cache():
                try:
                    weighted,record=step_data(policy,tasks["train"][index-1],index,"alternating",0,options,0.,
                        training_seed=233,local_candidates=4,local_repeats=1,local_continuation="greedy",
                        trajectory_objective="strict",local_objective="paired_delta",teacher=teacher if use_teacher else None,
                        teacher_protocol=teacher_protocol)
                finally:
                    atomic_json(root/"teacher_api_usage.json",dict(teacher.counts))
                cache=dict(policy.cache_stats)
            metrics=policy.grpo_update(weighted,"manager",optimizers["manager"],beta=.01)
            repair=record.get("teacher_repair",{})
            if repair.get("accepted"):
                record["teacher_sft"]=policy.sft_update([Sample(**repair["supervised_sample"])],optimizers["teacher_manager"])
                counts["teacher_sft_updates"]+=1
            if policy.fingerprint("actor") != actor or policy.frozen_base_signature()!=frozen_base:
                raise ContractError("Training changed the frozen executor/base")
            counts["optimizer_steps"]+=metrics["optimizer_step"]
            counts[record["update_kind"]+"_questions"]+=1
            if record.get("local_group"):
                m=record["local_group"]["metrics"]
                counts["local_task_signal"]+=bool(any(m["task_advantages"]))
                counts["homogeneous_group_with_paired_signal"]+=m["group_zero_variance"] and bool(any(m["task_advantages"]))
            if repair:
                counts["teacher_repair_prefixes"]+=1
                counts["teacher_reason_"+repair["reason"]]+=1
                counts["teacher_corrected_after_feedback"]+=bool(repair.get("corrected_after_feedback"))
            if teacher_protocol == "evidence_v2" and use_teacher:
                counts["teacher_gate_"+record.get("teacher_gate",{}).get("reason","not_local")]+=1
            if record.get("evidence_use_queue"):
                counts["offline_queue_"+record["evidence_use_queue"]["route"]]+=1
            record.update(grpo=metrics,cache=cache,teacher_scheduled=use_teacher,
                          manager_version=policy.versions["manager"],actor_unchanged=True)
            # Keep the full replay provenance, loss samples and original/repair
            # trajectories. Compression only changes storage, not training data.
            path=root/"updates"/f"step_{index:05d}.json.gz";path.parent.mkdir(exist_ok=True)
            pending=path.with_suffix(".pending")
            with gzip.open(pending,"wt",encoding="utf-8") as f:json.dump(record,f,ensure_ascii=False)
            pending.replace(path)
            train_seconds+=time.monotonic()-began
            if index%100==0 or index in {8,limit}:
                checkpoint(root,index,policy,optimizers,identity,{"counts":dict(counts),"train_seconds":train_seconds,
                    "journal_sha256":file_sha(path),"actor_sha256":actor})
            elapsed=time.monotonic()-session_start
            atomic_json(root/"status.json",{"phase":"training","completed":index,"expected":target,
                "durable_step":read(root/"checkpoints/latest.json")["step"],"counts":dict(counts),
                "teacher_api":dict(teacher.counts),"train_seconds":train_seconds,
                "session_seconds":elapsed,"session_questions":index-session_done})
            print(json.dumps({"question":index,"kind":record["update_kind"],"seconds":time.monotonic()-began,
                              "grpo_update":metrics["optimizer_step"],"teacher":repair.get("reason")}),flush=True)
    finally:
        atomic_json(root/"teacher_api_usage.json",dict(teacher.counts));teacher.close()
    if limit<target:
        atomic_json(root/"status.json",{"phase":"stopped_at_update_boundary","completed":limit,"expected":target})
        return
    final=evaluate(root,policy,tasks["dev"],"final",options,actor)
    journal_ids=[]
    for index in range(1,target+1):
        with gzip.open(root/"updates"/f"step_{index:05d}.json.gz","rt",encoding="utf-8") as stream:
            item=json.load(stream)
        if item["step"]!=index or not item["actor_unchanged"]:
            raise ContractError("Training journal boundary/freeze mismatch")
        journal_ids.append(item["task_id"])
    if journal_ids!=receipt["train"]["ids"][:target] or len(set(journal_ids))!=target:
        raise ContractError("Training journal is not the complete unique requested manifest prefix")
    summary={"complete":True,"completed":target,"expected":target,"counts":dict(counts),
        "initial":read(root/"initial_summary.json"),"final":final,"paired":paired(root),
        "train_seconds":train_seconds,"teacher_api":read(root/"teacher_api_usage.json"),
        "actor_unchanged":policy.fingerprint("actor")==actor,"training_ids_complete_unique":True,
        "receipt_sha256":file_sha(root/"receipt.json"),
        "checkpoint":str(root/"checkpoints"/f"step_{target:05d}"),"scope":"single-seed source development experiment"}
    atomic_json(root/"summary.json",summary)
    atomic_json(root/"status.json",{"phase":"complete","completed":target,"expected":target})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--teacher-config",type=Path,required=True);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--target",type=int,default=10000)
    args=parser.parse_args()
    with (args.root/"worker.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:execute(args.root,args.teacher_config,args.resume,args.stop_after,args.target)
        except Exception as error:
            atomic_json(args.root/"worker_error.json",{"type":type(error).__name__,"error":str(error),
                "traceback":traceback.format_exc()})
            raise


if __name__=="__main__":main()
