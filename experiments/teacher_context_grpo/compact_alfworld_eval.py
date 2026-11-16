"""Evaluate public prompting interventions and gated memory on independent ALFWorld tasks."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import traceback

import torch

from action_memory_dev import DevPolicy
from compact_alfworld import CompactALFRunner, paired_trace
from experiments.memory_curriculum import shared_memory_eval as shared
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.io import atomic_json, atomic_jsonl, digest, file_sha
from scope.policy import HFPolicy
from target_action_adaptation import create_env


def run(root):
    manifest = shared.read(root / "receipt.json")
    for file, expected in manifest["source_hashes"].items():
        if file_sha(root / "source" / file) != expected:
            raise ContractError("ALFWorld driver source changed")
    for file, expected in manifest["assets"].items():
        if file_sha(file) != expected:
            raise ContractError("ALFWorld dataset asset changed")
    checkpoint = Path(manifest["checkpoint"]["path"])
    if file_sha(checkpoint / "state.pt") != manifest["checkpoint"]["sha256"]:
        raise ContractError("ALFWorld checkpoint differs")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    inner = HFPolicy(manifest["model"], gradient_checkpointing=False)
    load_checkpoint(checkpoint, inner, restore_random=False)
    frozen = inner.fingerprint()
    base = inner.frozen_base_signature()
    policy = DevPolicy(inner)
    shared.BRIDGE = root / "serve_alfworld.py"
    specs = manifest["tasks"]["alfworld"]
    ids = [task["id"] for task in specs]
    if len(set(ids)) != len(ids) or not ids or set(ids) & set(manifest["held_out_ids"]):
        raise ContractError("ALFWorld development split changed or overlaps final tasks")
    runtime = manifest["runtime"]
    quote_window = manifest.get("quote_probe_history_turns")
    config = {"environment": {"kind": "alfworld_http"}, "runtime": runtime}
    env, service = create_env(root, "alfworld", "adapted_interface", specs, config)
    pairs, gates = [], []
    with service(config, root):
        try:
            env.health()
            probe_runtime = {**runtime, "manager_mode": "rule"}
            if quote_window:
                probe_runtime.update(paired_history_window=quote_window, paired_require_quote=True)
            probe = CompactALFRunner(policy, env, probe_runtime, "dev",
                                     domain="alfworld", arm="public_candidates", compact=False)
            gated = CompactALFRunner(policy, env, runtime, "dev", domain="alfworld",
                                     arm="public_candidates", gated_manager=True)
            for task_id in ids:
                path = root / "cases" / (digest(task_id) + ".json")
                if path.exists():
                    row = shared.read(path)
                    if row["task_id"] != task_id:
                        raise ContractError("Resumed public prefix has the wrong ID")
                else:
                    pair = paired_trace(probe, task_id)
                    row = {"task_id": task_id, "pair": pair}
                    if not quote_window:
                        optimized, steps = gated.trace(task_id)
                        ops = Counter(event["result"].get("operation") for event in optimized.events
                                      if event["phase"] != "execute" and event["sampled"])
                        row["gated"] = {"metrics": optimized.metrics, "costs": optimized.costs,
                                        "model_operations": {k: v for k, v in ops.items() if k},
                                        "retrieval_visible_steps": sum(bool(e["retrieved_evidence_in_view"]) for e in steps)}
                    atomic_json(path, row)
                pairs.append(row["pair"])
                if "gated" in row:
                    gates.append({"task_id": task_id, **row["gated"]})
                atomic_json(root / "status.json", {"phase": "evaluating", "completed": len(pairs), "expected": len(ids)})
        finally:
            env.close()
    if not shared.read(root / "alf_cleanup.json")["own_processes_ended"]:
        raise ContractError("Owned ALFWorld bridge still running")
    if inner.fingerprint() != frozen or inner.frozen_base_signature() != base:
        raise ContractError("Frozen policy parameters changed in evaluation")
    atomic_jsonl(root / "paired.jsonl", pairs)
    atomic_jsonl(root / "gated.jsonl", gates)
    contrasts = {}
    for left, right in (("history", "card"), ("card", "card_quote")):
        cells = [(row["branches"][left], row["branches"][right]) for row in pairs
                 if {left, right} <= set(row.get("branches", {}))]
        contrasts[f"{right}_vs_{left}"] = {
            "n": len(cells),
            "changed_actions": sum(a["command"] != b["command"] for a, b in cells),
            "strict_success": [sum(pair[i]["metrics"]["success"] for pair in cells) for i in (0, 1)],
            "confirmed_action_effects": [sum(pair[i]["public_confirmed_effect"] for pair in cells) for i in (0, 1)],
            "mean_token_delta": sum(b["tokens"] - a["tokens"] for a, b in cells) / len(cells) if cells else None}
    atomic_json(root / "summary.json", {"complete": True, "n": len(ids), "tasks": ids, "pairs": contrasts,
                "parameters_unchanged": True, "control_success": sum(v["baseline"]["success"] for v in pairs),
                "gated_success": sum(v["metrics"]["success"] for v in gates) if gates else None,
                "control_mean_tokens": sum(v["baseline_costs"]["charged_tokens"] for v in pairs) / len(ids),
                "gated_mean_tokens": sum(v["costs"]["charged_tokens"] for v in gates) / len(ids) if gates else None,
                "gated_mean_manager_input": sum(v["costs"]["manager_input"] for v in gates) / len(ids) if gates else None,
                "quote_probe_history_turns": quote_window,
                "offline_probe_tokens": sum(v["discovery_tokens"] for v in pairs),
                "paired_sha256": file_sha(root / "paired.jsonl"), "gated_sha256": file_sha(root / "gated.jsonl")})
    atomic_json(root / "status.json", {"phase": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with (args.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve())
        except Exception as exc:
            atomic_json(args.root / "error.json", {"error": str(exc), "traceback": traceback.format_exc()})
            atomic_json(args.root / "status.json", {"phase": "needs_attention"})
            raise
