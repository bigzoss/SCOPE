"""Frozen transfer of the HotpotQA manager to complete target test manifests."""
import argparse
import fcntl
from pathlib import Path
import random
import signal

import numpy as np
import torch

from experiments.memory_curriculum.shared_memory_eval import alf_service, port, read
from scope.alfworld import ALFWorldEnvironment, ALFWorldRegistry
from scope.checkpoint import load_checkpoint
from scope.cli import model_identity, source_hashes, verify_checkpoint_base
from scope.config import validate_config
from scope.contracts import ContractError
from scope.data import SplitRegistry
from scope.environments import ShopEnvironment
from scope.evaluation import evaluate, paired_comparison
from scope.io import atomic_json, digest, file_sha
from scope.launcher import own_service
from scope.policy import HFPolicy

PROJECT = Path("/home/hyx/lm_workspace/SCOPE")
SEED = 233
TEST_N = {"shopsimulator": 900, "alfworld": 134}


def prepare(domain, run, gpu, model):
    if domain == "shopsimulator":
        config = read(PROJECT/"runs/memory_gap_diagnostics_20261109.L3Bq73/config.json")
        registry = SplitRegistry(config["data_root"], SEED)
        config["environment"]["manifest"] = str(Path(config["data_root"])/"manifests/test_unseen.json")
        manifest = read(config["environment"]["manifest"])
        if set(registry.ids("test")) != {str(row["task_id"]) for row in manifest["tasks"]}:
            raise ContractError("ShopSimulator test registry and service manifest differ")
    else:
        config = read(PROJECT/"configs/alfworld_http.json")
        original = Path(config["environment"]["manifest"])
        registry = ALFWorldRegistry(original, SEED)
        manifest = read(original)
        tasks = [task for task in manifest["tasks"] if task["split"] == "valid_unseen"]
        if any(file_sha(task["gamefile"]) != task["game_sha256"] for task in tasks):
            raise ContractError("ALFWorld game checksum mismatch")
        subset = {**manifest, "tasks": tasks, "counts": {"valid_unseen": len(tasks)}}
        manifest_path = run/"alfworld"/"test_manifest.json"
        if manifest_path.exists() and read(manifest_path) != subset:
            raise ContractError("ALFWorld test manifest changed")
        atomic_json(manifest_path, subset)
        config["environment"]["manifest"] = str(manifest_path)
    if len(registry.ids("test")) != TEST_N[domain]:
        raise ContractError("Unexpected number of target test tasks")
    config["model"] = {"path": str(model), "rank": 16, "alpha": 32}
    config["gpu"] = gpu
    config["evaluation"] = {"seed": SEED, "sequence_length": 1}
    config["runtime"].update(state_graph=True, memory_choice=True, shared_memory=True,
                             decision_evidence=True, action_mode="grounded", loop_policy="state",
                             max_turns=12 if domain == "shopsimulator" else 24)
    return validate_config(config), registry


def execute(args, output):
    source = args.run/"train"
    source_receipt = read(source/"receipt.json")
    model = Path(source_receipt["model"]["path"])
    start = source/"checkpoints"/"step_00000"
    manager = start
    if args.arm == "transferred":
        summary = read(source/"summary.json")
        if not summary["complete"] or summary["completed_updates"] != source_receipt["fixed_updates"]:
            raise ContractError("HotpotQA training has not reached its fixed endpoint")
        manager = source/"checkpoints"/source_receipt["selected_checkpoint"]
        if file_sha(manager/"state.pt") != summary["checkpoint_sha256"]:
            raise ContractError("Final HotpotQA checkpoint checksum differs")
    config, registry = prepare(args.domain, args.run, args.gpu, model)
    manifest = config["environment"]["manifest"]
    manifest_sha = file_sha(manifest)
    test_ids = registry.ids("test")
    identity = model_identity(model, PROJECT/"runs/.identity-cache")
    for checkpoint in {start, manager}:
        verify_checkpoint_base(checkpoint, identity)

    receipt = {"domain": args.domain, "arm": args.arm, "test_ids": test_ids,
               "manifest_sha256": manifest_sha, "dataset_sha256": registry.hashes,
               "actor_checkpoint_sha256": file_sha(start/"state.pt"),
               "manager_checkpoint_sha256": file_sha(manager/"state.pt"),
               "source_training_receipt_sha256": file_sha(source/"receipt.json"),
               "source_sha256": source_hashes(), "driver_sha256": file_sha(Path(__file__)),
               "model_sha256": identity["sha256"], "seed": SEED, "runtime": config["runtime"],
               "evaluation": config["evaluation"], "gpu": args.gpu}
    receipt_path = output/"receipt.json"
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ContractError("Frozen evaluation receipt changed")
    atomic_json(receipt_path, receipt)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / total))
    policy = HFPolicy(model, gradient_checkpointing=False)
    load_checkpoint(start, policy, restore_random=False)
    actor_hash = policy.fingerprint("actor")
    if manager != start:
        load_checkpoint(manager, policy, restore_random=False, roles={"manager"})
    if policy.fingerprint("actor") != actor_hash:
        raise ContractError("Manager transfer modified the actor")
    atomic_json(output/"roles.json", {"actor": actor_hash, "manager": policy.fingerprint("manager")})
    endpoint = config["environment"]["base_url"] = f"http://127.0.0.1:{port()}"
    if args.domain == "shopsimulator":
        env = ShopEnvironment(endpoint, test_ids, manifest_sha)
        service = own_service(config, output, True)
    else:
        env = ALFWorldEnvironment(endpoint, test_ids, manifest)
        service = alf_service(config, output)
    with service:
        env.health()
        evaluate(policy, env, registry, config, output, "test")
    if policy.fingerprint("actor") != actor_hash:
        raise ContractError("Actor changed during target evaluation")
    cleanup = read(output/("cleanup.json" if args.domain == "shopsimulator" else "alf_cleanup.json"))
    if not cleanup["own_processes_ended"]:
        raise ContractError("Owned simulator did not terminate")
    atomic_json(output/"verified.json", {"actor": actor_hash, "receipt_sha256": file_sha(receipt_path),
                                       "service_cleaned_up": True})


def aggregate(run):
    domains = {}
    for domain, expected in TEST_N.items():
        initial, transfer = (run/domain/arm for arm in ("initial", "transferred"))
        left, right = read(initial/"receipt.json"), read(transfer/"receipt.json")
        for field in ("test_ids", "manifest_sha256", "dataset_sha256", "actor_checkpoint_sha256",
                      "source_training_receipt_sha256", "model_sha256", "runtime", "evaluation",
                      "source_sha256", "driver_sha256", "seed"):
            if left[field] != right[field]:
                raise ContractError(f"{domain}: paired evaluation identity differs in {field}")
        actor = read(initial/"verified.json")["actor"]
        if read(transfer/"verified.json")["actor"] != actor:
            raise ContractError(f"{domain}: paired actor weights differ")
        comparison = paired_comparison(initial, transfer)
        if comparison["n"] != expected:
            raise ContractError(f"{domain}: paired test is incomplete")
        counts = {}
        for arm, directory in (("initial", initial), ("transferred", transfer)):
            summary = read(directory/"summary.json")
            if summary["task_ids_sha256"] != digest(sorted(left["test_ids"])):
                raise ContractError(f"{domain}: completed tests differ from frozen manifest")
            counts[arm] = {field: summary[field] for field in
                           ("n", "success_count", "strict_success_rate", "mean_total_tokens",
                            "mean_loose_reward", "mean_strict_reward", "illegal_action_rate_per_turn")}
        domains[domain] = {"paired": comparison, **counts}
    atomic_json(run/"transfer_summary.json", {"complete": True, "domains": domains})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--domain", choices=TEST_N)
    parser.add_argument("--arm", choices=("initial", "transferred"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.run)
        return
    if None in (args.domain, args.arm, args.gpu):
        parser.error("evaluation requires --domain, --arm and --gpu")
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    output = args.run/args.domain/args.arm
    output.mkdir(parents=True, exist_ok=True)
    with (output/".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, output)


if __name__ == "__main__":
    main()
