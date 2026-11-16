"""One deterministic pass over both verified official HotpotQA training shards."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import re
import shutil
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.memory_curriculum.hotpot_grpo import (
    SEED, evaluate, rng_seed, selected, step_data,
)
from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.cli import model_identity, source_hashes, verify_checkpoint_base
from scope.contracts import ContractError
from scope.hotpot import split_row
from scope.hotpot_intervention import public_support_inventory
from scope.io import atomic_json, digest, file_sha
from scope.policy import HFPolicy

SHARDS = {
    "train-00000-of-00002.parquet": (45224, "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68"),
    "train-00001-of-00002.parquet": (45223, "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f"),
}
VALIDATION_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
TOTAL = 90447
CHECKPOINT_INTERVAL = 100
WARMUP = 6
OPTIONS = {"sentence_candidates": True, "bridge_candidates": True, "shared_interface": True}
EMPTY_CHAIN = digest("hotpot-full-epoch-v1")


def read(path):
    return json.loads(path.read_text())


def question_key(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def prepare(dataset):
    rows = []
    shards = []
    for name, (expected, sha) in SHARDS.items():
        path = dataset / name
        if file_sha(path) != sha:
            raise ContractError(f"Official train shard SHA mismatch: {name}")
        data = pq.read_table(path).to_pylist()
        if len(data) != expected:
            raise ContractError(f"Official train shard row count mismatch: {name}")
        rows.extend(data)
        shards.append({"file": str(path), "rows": expected, "sha256": sha})

    ids = [row["id"] for row in rows]
    if len(ids) != TOTAL or len(set(ids)) != TOTAL:
        raise ContractError("Training set must contain exactly 90,447 unique official IDs")
    train_questions = {question_key(row["question"]) for row in rows}
    rows.sort(key=lambda row: digest([SEED, "full-train-epoch-1", row["id"]]))
    valid_path = dataset / "validation-00000-of-00001.parquet"
    if file_sha(valid_path) != VALIDATION_SHA:
        raise ContractError("Official validation shard SHA mismatch")
    validation_keys = pq.read_table(valid_path, columns=["id", "question"]).to_pylist()
    train_ids = set(ids)
    excluded = {row["id"] for row in validation_keys
                if row["id"] in train_ids or question_key(row["question"]) in train_questions}
    dev, available = selected(valid_path, 48, "validation", excluded)
    if any((item := public_support_inventory(public, gold))["outside_public"] or
           not item["expected"] for public, gold in dev):
        raise ContractError("Validation questions must have all support sentences in public input")
    manifest = {"dataset": "hotpotqa/hotpot_qa distractor", "official_shards": shards,
                "train_count": len(rows), "epochs": 1, "ordered_train_ids": [row["id"] for row in rows],
                "validation": {"path": str(valid_path), "sha256": VALIDATION_SHA,
                               "available": available, "ids": [public["id"] for public, _ in dev],
                               "excluded_train_overlap": len(excluded)}}
    return rows, dev, manifest


def compact(record, metrics):
    group = record.get("local_group")
    result = {
        "step": record["step"], "task_id": record["task_id"], "kind": record["update_kind"],
        "local_stage": record["local_stage"], "sampling_tokens": record["sampling_tokens"],
        "trajectory_scores": [branch["score"] for branch in record["primary"]],
        "global_advantages": record["global_advantages"],
        "metrics": metrics,
    }
    if group:
        result["local"] = {**group["metrics"], "boundary": group["boundary"],
                           "continuation_seeds": group["continuation_seeds"],
                           "choices": [candidate["sample"]["text"] for candidate in group["candidates"]],
                           "operations": [candidate["operation"] for candidate in group["candidates"]],
                           "candidate_trajectories": sum(len(c["runs"]) for c in group["candidates"])}
    return result


def ledger(output, rows, until):
    counts = Counter()
    tokens = Counter()
    chain = EMPTY_CHAIN
    for step in range(1, until + 1):
        entry = read(output / "updates" / f"step_{step:05d}.json")
        if (entry["step"], entry["task_id"]) != (step, rows[step - 1]["id"]):
            raise ContractError(f"Training ledger does not match the fixed epoch at step {step}")
        chain = digest([chain, entry])
        account(counts, tokens, entry)
    return chain, counts, tokens


def account(counts, tokens, entry):
    metrics = entry["metrics"]
    counts["optimizer_steps"] += metrics["optimizer_step"]
    counts["manager_changes"] += metrics["manager_changed"]
    counts["trajectory_steps"] += entry["kind"] == "trajectory"
    counts["local_steps"] += entry["kind"] == "local"
    counts["trajectory_rollouts"] += len(entry["trajectory_scores"])
    counts["local_groups"] += "local" in entry
    if "local" in entry:
        counts["local_signal_groups"] += not entry["local"]["task_zero_variance"]
        counts["local_candidate_trajectories"] += entry["local"]["candidate_trajectories"]
    tokens.update(entry["sampling_tokens"])


def keep_recent(checkpoints):
    completed = sorted(p for p in checkpoints.glob("step_[0-9][0-9][0-9][0-9][0-9]")
                       if p.is_dir() and not p.is_symlink() and p.name != "step_00000"
                       and (p / "metadata.json").is_file())
    for old in completed[:-2]:
        if old.resolve().parent != checkpoints.resolve():
            raise ContractError("Checkpoint path escaped the current run")
        shutil.rmtree(old)


def train(args, output):
    started = time.monotonic()
    rows, dev, manifest = prepare(args.run / "dataset")
    manifest["ordered_train_ids_sha256"] = digest(manifest["ordered_train_ids"])
    if args.preflight:
        atomic_json(output / "preflight.json", manifest)
        print(f"Verified {len(rows)} unique train IDs and {len(dev)} separate validation IDs", flush=True)
        return

    model = model_identity(args.model, args.identity_cache)
    verify_checkpoint_base(args.warmstart, model)
    receipt = {**manifest, "model": model, "warmstart_sha256": file_sha(args.warmstart / "state.pt"),
               "warmstart": str(args.warmstart), "gpu": args.gpu, "seed": SEED,
               "fixed_updates": TOTAL, "selected_checkpoint": f"step_{TOTAL:05d}",
               "credit_mode": "alternating", "outcome_only_updates": WARMUP,
               "trajectory_objective": "strict_em", "local_candidates": 4, "continuation": "greedy",
               "lr": 1e-5, "clip": .2, "kl_beta": .01, "compression_token_weight": 0.,
               "checkpoint_interval": CHECKPOINT_INTERVAL, "checkpoint_retention": "initial_and_latest_two",
               "rollout_options": OPTIONS, "target_updates": 0, "selection_uses_validation": False,
               "source": source_hashes(), "driver_sha256": file_sha(Path(__file__)),
               "step_data_sha256": file_sha(Path(__file__).with_name("hotpot_grpo.py")),
               "torch": torch.__version__}
    receipt_file = output / "receipt.json"
    checkpoints = output / "checkpoints"
    if args.resume:
        if read(receipt_file) != receipt:
            raise ContractError("Full training receipt changed")
    elif receipt_file.exists():
        raise ContractError("Training run already exists. Use --resume.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / total_memory))
    policy = HFPolicy(args.model)
    optimizer = torch.optim.AdamW(policy.parameters("manager"), lr=1e-5, weight_decay=0.)
    optimizers = {"manager": optimizer}
    identity = digest(receipt)
    if args.resume:
        candidates = sorted(p for p in checkpoints.glob("step_*")
                            if re.fullmatch(r"step_[0-9]{5}", p.name) and (p / "metadata.json").exists())
        restored = load_checkpoint(candidates[-1], policy, optimizers, identity)
        start = restored["step"]
        chain, counts, tokens = ledger(output, rows, start)
        if restored["extra"]["coverage_chain"] != chain:
            raise ContractError("Checkpoint and completed training ledger differ")
        atomic_json(checkpoints / "latest.json", {"path": candidates[-1].name, "step": start})
        atomic_json(output / "resume_validation.json", {
            "passed": True, "restored_update": start, "manager_version": policy.versions["manager"],
            "coverage_chain": chain, "optimizer_parameter_states": len(optimizer.state), "rng_restored": True})
    else:
        load_checkpoint(args.warmstart, policy, restore_random=False)
        policy.reset_reference("manager")
        atomic_json(receipt_file, receipt)
        start, chain, counts, tokens = 0, EMPTY_CHAIN, Counter(), Counter()
        save_checkpoint(checkpoints, 0, policy, optimizers, identity, {"coverage_chain": chain})

    actor = policy.fingerprint("actor")
    base = policy.frozen_base_signature()
    target = args.stop_after or TOTAL
    if not start <= target <= TOTAL:
        raise ContractError("Invalid continuation boundary")
    latest = start
    for step in range(start + 1, target + 1):
        torch.manual_seed(rng_seed("update", step))
        before = policy.fingerprint("manager")
        weighted, record = step_data(policy, split_row(rows[step - 1]), step, "alternating", WARMUP,
                                     OPTIONS, 0., local_candidates=4, trajectory_objective="strict")
        metrics = policy.grpo_update(weighted, "manager", optimizer, beta=.01)
        after = policy.fingerprint("manager")
        if policy.fingerprint("actor") != actor or policy.frozen_base_signature() != base:
            raise ContractError("Frozen actor or backbone changed during source training")
        metrics.update(manager_changed=before != after, manager_version=policy.versions["manager"])
        entry = compact(record, metrics)
        atomic_json(output / "updates" / f"step_{step:05d}.json", entry)
        chain = digest([chain, entry])
        account(counts, tokens, entry)
        if step % CHECKPOINT_INTERVAL == 0 or step == target:
            save_checkpoint(checkpoints, step, policy, optimizers, identity, {"coverage_chain": chain})
            latest = step
            keep_recent(checkpoints)
        state = {"phase": "training", "completed_updates": step, "expected_updates": TOTAL,
                 "durable_checkpoint_update": latest, "counts": dict(counts), "last_task_id": entry["task_id"],
                 "session_elapsed_seconds": time.monotonic() - started, "session_start_update": start}
        atomic_json(output / "status.json", state)
        print(f"{step}/{TOTAL} task={entry['task_id']} kind={entry['kind']} optimizer={metrics['optimizer_step']} "
              f"manager_changed={metrics['manager_changed']}", flush=True)
    if target != TOTAL:
        atomic_json(output / "status.json", {**state, "phase": "checkpoint_boundary"})
        return

    checked_chain, checked_counts, checked_tokens = ledger(output, rows, TOTAL)
    if (checked_chain, checked_counts, checked_tokens) != (chain, counts, tokens):
        raise ContractError("Completed epoch does not match ordered training receipt")
    final = checkpoints / f"step_{TOTAL:05d}"
    final_sha = read(final / "metadata.json")["state_sha256"]
    load_checkpoint(checkpoints / "step_00000", policy, receipt_hash=identity, restore_random=False)
    atomic_json(output / "status.json", {"phase": "validate_initial", "completed_updates": TOTAL})
    initial = evaluate(policy, dev, output, "initial", OPTIONS)
    load_checkpoint(final, policy, receipt_hash=identity, restore_random=False)
    atomic_json(output / "status.json", {"phase": "validate_final", "completed_updates": TOTAL})
    trained = evaluate(policy, dev, output, "final", OPTIONS)
    summary = {"complete": True, "completed_updates": TOTAL, "unique_train_tasks": TOTAL,
               "coverage_chain": chain, "ordered_train_ids_sha256": manifest["ordered_train_ids_sha256"],
               "counts": dict(counts), "sampling_tokens": dict(tokens), "actor_unchanged": policy.fingerprint("actor") == actor,
               "checkpoint": str(final), "checkpoint_sha256": final_sha, "initial": initial, "final": trained}
    if not summary["actor_unchanged"]:
        raise ContractError("Frozen actor changed")
    atomic_json(output / "policy.json", {"base_model": str(args.model), "checkpoint": str(final),
                "checkpoint_sha256": final_sha, "training_receipt_sha256": file_sha(receipt_file),
                "selected_step": TOTAL, "manager_role": "manager", "rollout_options": OPTIONS})
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", {"phase": "complete", "completed_updates": TOTAL, "expected_updates": TOTAL})


def main():
    parser = argparse.ArgumentParser()
    for option in ("run", "model", "identity-cache", "warmstart"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.stop_after is not None and not 1 <= args.stop_after <= TOTAL:
        parser.error("--stop-after must be a positive step within the full epoch")
    if args.preflight and (args.resume or args.stop_after is not None):
        parser.error("Preflight cannot resume or train")
    output = args.run / ("preflight" if args.preflight else "train")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args, output)


if __name__ == "__main__":
    main()
