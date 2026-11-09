"""Fixed source-only HotpotQA training and paired, frozen validation."""
import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import random
import unicodedata

import numpy as np
import pyarrow.parquet as pq

from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.cli import model_identity, source_hashes, verify_checkpoint_base
from scope.contracts import ContractError
from scope.hotpot import NOOP, STAGES, rollout, score, split_row, weights
from scope.hotpot_intervention import compression_efficiency, continuation_tokens, coverage, factorial, public_support_inventory
from scope.io import atomic_json, atomic_jsonl, digest, file_sha
from scope.policy import HFPolicy


SEED = 233
UPDATES = 16
GLOBAL_UPDATES = 6
GROUP = 4
VAL_N = 16
LOCAL_WEIGHT = .75
LR = 1e-5
LOCAL_ORDER = (3,0,2,4,1)


def selected(path, n, split, excluded):
    rows = pq.read_table(path).to_pylist()
    by_type = {}
    for row in rows:
        if row["id"] in excluded:
            continue
        by_type.setdefault(row["type"],[]).append(row)
    for kind in by_type:
        by_type[kind].sort(key=lambda row:digest([SEED,split,row["id"]]))
    interleaved = [by_type[kind][index] for index in range(max(map(len,by_type.values()),default=0))
                   for kind in sorted(by_type) if index < len(by_type[kind])]
    if len({r["id"] for r in rows}) != len(rows) or len(interleaved)<n:
        raise ContractError("HotpotQA source IDs are not unique or sufficient")
    return [split_row(row) for row in interleaved[:n]],len(rows)


def rng_seed(*values, seed=SEED):
    return int(digest([seed,*values])[:8],16)


def step_data(policy, task, update, credit_mode, global_updates, rollout_options, compression_token_weight,
              evidence_mask="delete", training_seed=SEED):
    public,gold = task
    stage = LOCAL_ORDER[(update-global_updates-1)%len(LOCAL_ORDER)] if update>global_updates and credit_mode!="global" else None
    groups, controls = [], []
    for branch in range(GROUP):
        seed = rng_seed("train",update,branch,seed=training_seed)
        trace,samples = rollout(policy,public,"rl_train",seed,**rollout_options)
        group = {"trace":trace,"samples":samples,"score":score(trace,gold)}
        if stage is not None and samples[stage] and trace["actions"][stage]["operation"] != NOOP[STAGES[stage]]:
            control,_ = rollout(policy,public,"rl_train",seed,factual=trace,intervention=stage,**rollout_options)
            group["control_score"] = score(control,gold)
            controls.append({"branch":branch,"score":group["control_score"],"trace":control,
                             "boundary":trace["actions"][stage]["boundary"]})
            if credit_mode=="evidence":
                contrast = factorial(policy,public,gold,trace,control,stage,mask_mode=evidence_mask)
                group["evidence_credit"] = contrast["evidence_credit"]
                controls[-1]["factorial"] = contrast
            if STAGES[stage]=="compress":
                efficiency = compression_efficiency(trace,control,stage,group["score"],group["control_score"])
                group["compression_credit"] = compression_token_weight * efficiency["relative_saving"]
                controls[-1]["compression_efficiency"] = {**efficiency,"credit":group["compression_credit"]}
        groups.append(group)
    weighted,global_adv,delta = weights(groups,stage,LOCAL_WEIGHT)
    for group in groups:
        for index,sample in enumerate(group["samples"]):
            if sample:
                group["trace"]["actions"][index]["sample"] = asdict(sample)
    record = {"step":update,"task_id":public["id"],"local_stage":STAGES[stage] if stage is not None else None,
              "primary":[{"trace":g["trace"],"score":g["score"]} for g in groups],"controls":controls,
              "global_advantages":global_adv,"local_reward_deltas":delta}
    record["sampling_tokens"] = {
        "primary":sum(sum(continuation_tokens(g["trace"]).values()) for g in groups),
        "control":sum(sum(continuation_tokens(c["trace"]).values()) for c in controls),
        "evidence_reader_replays":sum(cell["prompt_tokens"]+cell["completion_tokens"]
                                     for c in controls for key,cell in c.get("factorial",{}).get("cells",{}).items()
                                     if key.endswith("_masked") and not cell["reused_visible"]),
    }
    return weighted,record


def evaluate(policy, tasks, output, name, rollout_options):
    before = policy.fingerprint()
    rows = []
    for public,gold in tasks:
        trace,_ = rollout(policy,public,"validation",rng_seed("validation",public["id"]),greedy=True,**rollout_options)
        rows.append({"trace":trace,"score":score(trace,gold),"coverage":coverage(trace,public,gold)})
    if policy.fingerprint()!=before:
        raise ContractError("Frozen validation changed the policy")
    file = output/f"validation_{name}.jsonl"
    atomic_jsonl(file,rows)
    metrics = {key:float(np.mean([r["score"][key] for r in rows])) for key in ("em","f1","support_visible_recall","reward")}
    metrics.update(n=len(rows),em_count=sum(r["score"]["em"] for r in rows),parameters_unchanged=True,
                   results_sha256=file_sha(file),by_task={r["trace"]["task_id"]:r["score"]["em"] for r in rows},
                   mean_actor_prompt_tokens=float(np.mean([len(r["trace"]["answer_sample"]["prompt_ids"]) for r in rows])),
                   coverage_totals={key:sum(r["coverage"][key] for r in rows) for key in rows[0]["coverage"]},
                   operations=dict(Counter(a["operation"] for r in rows for a in r["trace"]["actions"])))
    atomic_json(output/f"validation_{name}.json",metrics)
    return metrics


def prepare_splits(args):
    updates = args.updates
    excluded = set()
    for path in args.exclude_receipt:
        previous = json.loads(path.read_text())
        excluded.update(previous.get("train_ids",previous.get("train",{}).get("ids",[])))
        excluded.update(previous.get("validation",{}).get("ids",[]))
    train,train_n = selected(args.train,updates,"rl_train",excluded)
    dev,dev_n = selected(args.validation,args.validation_count,"validation",excluded)
    train_ids,dev_ids = [item[0]["id"] for item in train],[item[0]["id"] for item in dev]
    def questions(tasks):
        return {" ".join(unicodedata.normalize("NFKC",p["question"]).casefold().split()) for p,_ in tasks}

    if set(train_ids)&set(dev_ids) or questions(train)&questions(dev):
        raise ContractError("HotpotQA train and validation overlap")
    public_support = [public_support_inventory(public,gold) for public,gold in dev]
    missing_support = sum(row["outside_public"] for row in public_support)
    unlabeled = sum(row["expected"]==0 for row in public_support)
    manifest = {
        "train":{"path":str(args.train),"sha256":file_sha(args.train),"available":train_n,"ids":train_ids},
        "validation":{"path":str(args.validation),"sha256":file_sha(args.validation),"available":dev_n,"ids":dev_ids,
                      "require_complete_support":args.require_complete_support,"outside_public":missing_support,
                      "without_support_labels":unlabeled,
                      "expected_support":sum(row["expected"] for row in public_support)},
        "excluded_ids_sha256":digest(sorted(excluded)),
    }
    return train,dev,manifest


def execute(args, output):
    train,dev,manifest = prepare_splits(args)
    validation = manifest["validation"]
    if args.require_complete_support and (validation["outside_public"] or validation["without_support_labels"]):
        raise ContractError("Fixed validation lacks complete support: "
                            f"{validation['outside_public']} missing sentences, "
                            f"{validation['without_support_labels']} tasks without support labels")
    if args.preflight_only:
        atomic_json(output/"preflight.json",manifest)
        print(json.dumps(manifest,ensure_ascii=False),flush=True)
        return
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    updates = args.updates
    train_ids,dev_ids = manifest["train"]["ids"],validation["ids"]
    model = model_identity(args.model,args.identity_cache)
    warmstart = None
    if args.warmstart:
        verify_checkpoint_base(args.warmstart,model)
        warmstart = {"path":str(args.warmstart),"state_sha256":file_sha(args.warmstart/"state.pt")}
    rollout_options = {"sentence_candidates":args.bridge_candidates,"bridge_candidates":args.bridge_candidates,
                       "shared_interface":args.shared_interface}
    receipt = {"method":"SCOPE HotpotQA graph memory GRPO pilot","seed":args.seed,"gpu":args.gpu,
        "split_seed":SEED,"evaluation_seed":SEED,
        "protocol":"supplied-paragraph stream, no external corpus search; question released after context pressure",
        **manifest,
        "fixed_updates":updates,"outcome_only_updates":args.outcome_only_updates,"rollouts_per_update":GROUP,
        "credit_mode":args.credit_mode,"rollout_options":rollout_options,"warmstart":warmstart,
        "compression_token_weight":args.compression_token_weight,
        "evidence_mask":args.evidence_mask,
        "local_stage_order":[STAGES[i] for i in LOCAL_ORDER],"local_weight":LOCAL_WEIGHT,"learning_rate":LR,
        "kl_beta":.01,"clip":.2,"actor":"frozen Qwen3-8B greedy","selected_checkpoint":f"step_{updates:05d}",
        "selection_uses_validation":False,"source":source_hashes(),"driver_sha256":file_sha(Path(__file__)),
        "model":model,"torch":torch.__version__}
    identity = digest(receipt)
    receipt_file = output/"receipt.json"
    if receipt_file.exists():
        if json.loads(receipt_file.read_text())!=receipt:
            raise ContractError("Training receipt changed")
    elif args.resume:
        raise ContractError("Cannot resume without the original training receipt")
    else:
        atomic_json(receipt_file,receipt)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,26 * 2**30/total))
    policy = HFPolicy(args.model)
    optimizer = torch.optim.AdamW(policy.parameters("manager"),lr=LR,weight_decay=0.)
    checkpoints = output/"checkpoints"
    start = 0
    if args.resume:
        latest = json.loads((checkpoints/"latest.json").read_text())
        path = checkpoints/latest["path"]
        saved = load_checkpoint(path,policy,{"manager":optimizer},identity)
        start = saved["step"]
        loaded_optimizer = optimizer.state_dict()
        expected_optimizer = saved["optimizers"]["manager"]
        if loaded_optimizer["param_groups"]!=expected_optimizer["param_groups"] or set(loaded_optimizer["state"])!=set(expected_optimizer["state"]):
            raise ContractError("Resumed optimizer state differs from the checkpoint")
        atomic_json(output/"resume_validation.json",{"passed":True,"restored_update":start,
            "manager_version":policy.versions["manager"],"checkpoint":str(path),
            "optimizer_restored":True,"optimizer_parameter_states":len(optimizer.state),"rng_restored":True})
    else:
        if args.warmstart:
            load_checkpoint(args.warmstart,policy,restore_random=False)
            policy.reset_reference("manager")
        save_checkpoint(checkpoints,0,policy,{"manager":optimizer},identity)
    actor = policy.fingerprint("actor")
    base = policy.frozen_base_signature()
    target = args.stop_after or updates
    if target<start:
        raise ContractError("Stop boundary precedes the saved checkpoint")
    for update in range(start+1,target+1):
        atomic_json(output/"status.json",{"phase":"training","completed_updates":update-1,"running_update":update,
                                         "expected_updates":updates})
        torch.manual_seed(rng_seed("update",update,seed=args.seed))
        before = policy.fingerprint("manager")
        weighted, record = step_data(policy,train[update-1],update,args.credit_mode,args.outcome_only_updates,rollout_options,
                                     args.compression_token_weight,args.evidence_mask,args.seed)
        metrics = policy.grpo_update(weighted,"manager",optimizer,beta=.01)
        after = policy.fingerprint("manager")
        if policy.fingerprint("actor")!=actor or policy.frozen_base_signature()!=base:
            raise ContractError("Manager training modified the frozen execution policy")
        metrics.update(manager_changed=before!=after,manager_before=before,manager_after=after,
                       manager_version=policy.versions["manager"],actor_unchanged=True,
                       local_controls=len(record["controls"]),global_signal=any(record["global_advantages"]),
                       local_signal=any(record["local_reward_deltas"]))
        record["metrics"] = metrics
        journal = output/"updates"/f"step_{update:05d}.json"
        atomic_json(journal,record)
        save_checkpoint(checkpoints,update,policy,{"manager":optimizer},identity,
                        {"journal_sha256":file_sha(journal),"metrics":metrics})
        atomic_json(output/"status.json",{"phase":"training","completed_updates":update,"expected_updates":updates})
        print(f"update={update}/{updates} task={record['task_id']} {metrics}",flush=True)
    if target<updates:
        atomic_json(output/"status.json",{"phase":"stopped_at_update_boundary","completed_updates":target,"expected_updates":updates})
        return

    units = [json.loads((output/"updates"/f"step_{i:05d}.json").read_text()) for i in range(1,updates+1)]
    if [r["task_id"] for r in units]!=train_ids or [r["step"] for r in units]!=list(range(1,updates+1)):
        raise ContractError("Fixed training task/update history changed")
    final = checkpoints/f"step_{updates:05d}"
    metadata = json.loads((final/"metadata.json").read_text())
    load_checkpoint(checkpoints/"step_00000",policy,restore_random=False)
    atomic_json(output/"status.json",{"phase":"validate_initial","completed_updates":updates})
    initial = evaluate(policy,dev,output,"initial",rollout_options)
    load_checkpoint(final,policy,restore_random=False)
    atomic_json(output/"status.json",{"phase":"validate_final","completed_updates":updates})
    trained = evaluate(policy,dev,output,"final",rollout_options)
    probes = [c["factorial"] for r in units for c in r["controls"] if "factorial" in c]
    compression = [c["compression_efficiency"] for r in units for c in r["controls"] if "compression_efficiency" in c]
    summary = {"complete":True,"completed_updates":updates,"factual_trajectories":sum(len(r["primary"]) for r in units),
               "paired_controls":sum(len(r["controls"]) for r in units),
               "sampling_tokens":{key:sum(r["sampling_tokens"][key] for r in units)
                                  for key in units[0]["sampling_tokens"]},
               "updates_with_global_signal":sum(r["metrics"]["global_signal"] for r in units),
               "updates_with_local_signal":sum(r["metrics"]["local_signal"] for r in units),
               "compression_probes":{"attempted":len(compression),"eligible":sum(p["eligible"] for p in compression),
                                     "nonzero_credit":sum(p["credit"]!=0 for p in compression)},
               "evidence_probes":{"attempted":len(probes),"eligible":sum(p["eligible"] for p in probes),
                   "nonzero":sum(p["evidence_credit"]!=0 for p in probes),
                   "abstentions":dict(Counter(p["reason"] for p in probes if not p["eligible"])),
                   "reader_reruns":sum(not cell["reused_visible"] for p in probes if p["eligible"]
                                       for key,cell in p["cells"].items() if key.endswith("_masked"))},
               "updates_with_manager_change":sum(r["metrics"]["manager_changed"] for r in units),
               "actor_unchanged":policy.fingerprint("actor")==actor,"checkpoint":str(final),
               "checkpoint_sha256":metadata["state_sha256"],"initial":initial,"final":trained,
               "em_paired":{"improved":sum(trained["by_task"][i]>initial["by_task"][i] for i in dev_ids),
                            "worsened":sum(trained["by_task"][i]<initial["by_task"][i] for i in dev_ids)}}
    atomic_json(output/"policy.json",{"base_model":str(args.model),"checkpoint":str(final),"checkpoint_sha256":metadata["state_sha256"],
        "training_receipt_sha256":file_sha(receipt_file),"selected_step":updates,"manager_role":"manager",
        "runtime":"scope.hotpot.rollout","rollout_options":rollout_options})
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"status.json",{"phase":"complete","completed_updates":updates})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ("train","validation","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--model",type=Path)
    p.add_argument("--identity-cache",type=Path)
    p.add_argument("--gpu",type=int)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--stop-after",type=int)
    p.add_argument("--updates",type=int,default=UPDATES)
    p.add_argument("--outcome-only-updates",type=int,default=GLOBAL_UPDATES)
    p.add_argument("--validation-count",type=int,default=VAL_N)
    p.add_argument("--seed",type=int,default=SEED,help="Training seed. Task selection and evaluation seeds stay fixed.")
    p.add_argument("--credit-mode",choices=("global","paired","evidence"),default="paired")
    p.add_argument("--evidence-mask",choices=("delete","length_matched"),default="delete")
    p.add_argument("--bridge-candidates",action="store_true")
    p.add_argument("--shared-interface",action="store_true",
                   help="Use the target manager's tool vocabulary and selectable sentence compression.")
    p.add_argument("--compression-token-weight",type=float,default=0.)
    p.add_argument("--require-complete-support",action="store_true")
    p.add_argument("--warmstart",type=Path)
    p.add_argument("--exclude-receipt",type=Path,action="append",default=[])
    p.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if not args.preflight_only and (args.model is None or args.identity_cache is None or args.gpu is None):
        p.error("training requires --model, --identity-cache and --gpu")
    if args.preflight_only and (args.resume or args.stop_after is not None):
        p.error("--preflight-only cannot be combined with training continuation flags")
    if args.evidence_mask!="delete" and args.credit_mode!="evidence":
        p.error("length-matched masking requires --credit-mode evidence")
    if args.shared_interface and not args.bridge_candidates:
        p.error("--shared-interface requires --bridge-candidates")
    if not (args.updates>0 and args.validation_count>0 and 0<=args.outcome_only_updates<=args.updates):
        p.error("updates and validation count must be positive; outcome-only updates must fit the schedule")
    if not 0<=args.seed<2**32:
        p.error("seed must be in [0, 2**32)")
    if args.stop_after is not None and not 1<=args.stop_after<=args.updates:
        p.error("stop-after must be within the fixed training schedule")
    if not 0<=args.compression_token_weight<=.25 or (args.credit_mode=="global" and args.compression_token_weight):
        p.error("compression token weight must be in [0,0.25] and requires paired or evidence mode")
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args,args.output)


if __name__=="__main__":
    main()
