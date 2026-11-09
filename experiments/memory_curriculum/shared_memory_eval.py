"""Small paired, frozen tool ablation on the real ShopSimulator and ALFWorld bridges."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time

import numpy as np
import requests
import torch

from scope.alfworld import ALFWorldEnvironment
from scope.checkpoint import load_checkpoint
from scope.cli import model_identity, source_hashes, verify_checkpoint_base
from scope.config import validate_config
from scope.contracts import ContractError
from scope.data import SplitRegistry
from scope.environments import ShopEnvironment
from scope.evaluation import evaluate
from scope.io import atomic_json, digest, file_sha, read_jsonl
from scope.launcher import own_service
from scope.policy import HFPolicy


BRIDGE = Path("/home/hyx/lm_workspace/longagent/outputs/code_revisions/teica-v11-optimized-20261025-v1/scripts/serve_alfworld.py")
ALF_PYTHON = "/home/hyx/lm_workspace/longagent/.venv-alfworld/bin/python"
ARMS = {"full": [], "no_retrieval": ["RETRIEVE"], "no_compression": ["SUMMARIZE"]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def alf_service(config, output):
    url = config["environment"]["base_url"]
    manifest = Path(config["environment"]["manifest"])
    bootstrap = ("from scope.lifecycle import install_parent_guard;import sys,runpy;"
                 "install_parent_guard(int(sys.argv[1]));sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')")
    args = [ALF_PYTHON, "-c", bootstrap, str(os.getpid()), str(BRIDGE), "--manifest", str(manifest),
            "--port", url.rsplit(":",1)[1], "--pool-size", "1"]
    with (output/"alf_service.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(Path(__file__).resolve().parents[2])})
        try:
            deadline = time.monotonic()+120
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    health = requests.get(url+"/healthz", timeout=2).json()
                    if health.get("ready") and health.get("manifest_sha256") == file_sha(manifest):
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("ALFWorld service failed to become ready; see alf_service.log")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            atomic_json(output/"alf_cleanup.json",{"pid":process.pid,"own_processes_ended":process.poll() is not None,
                                                  "url":url})


class FixedTasks:
    def __init__(self, task_ids):
        self.task_ids = task_ids

    def evaluation_sequences(self, split, length, seed):
        return [[task] for task in self.task_ids]


def prepare(args):
    shop = read(args.project/"runs/memory_gap_diagnostics_20261109.L3Bq73/config.json")
    alf = read(args.project/"configs/alfworld_http.json")
    alf["model"] = copy.deepcopy(shop["model"])
    prior = read(args.project/"runs/transfer_gpu1_20261108/protocol.json")
    shop_registry = SplitRegistry(shop["data_root"],233)
    excluded = set(prior["source_diagnostic_tasks"])
    for name in ("memory_choice_20261109.FpCdgM","memory_gap_diagnostics_20261109.L3Bq73"):
        receipt = read(args.project/"runs"/name/"study_receipt.json")
        excluded.update(receipt["dev_tasks"])
        excluded.update(receipt.get("excluded_previous_dev",[]))
    n = args.tasks_per_domain
    shop_ids = sorted(set(shop_registry.ids("internal_dev"))-excluded,key=lambda task:digest(["shared-memory",task]))[:n]
    original = Path(alf["environment"]["manifest"])
    manifest = read(original)
    unseen = [r for r in manifest["tasks"] if r["split"]=="valid_unseen"]
    previous = set(prior["target_test_tasks"])
    selected = []
    kinds = sorted({r["task_type"] for r in unseen})
    if args.decision_comparison:
        kinds.sort(key=lambda kind:not kind.startswith(("pick_cool","pick_heat")))
    for kind in kinds[:n]:
        pool = [r for r in unseen if r["task_type"]==kind and r["task_id"] not in previous]
        row = min(pool,key=lambda r:digest(["shared-memory",r["task_id"]]))
        if file_sha(row["gamefile"]) != row["game_sha256"]:
            raise ContractError("ALFWorld game checksum mismatch")
        selected.append(row)
    if len(shop_ids)!=n or len(selected)!=n or len({r["game_sha256"] for r in selected})!=n:
        raise ContractError("Could not reserve distinct tasks per benchmark")
    alf_manifest = args.output/"alf_manifest.json"
    atomic_json(alf_manifest,{**manifest,"tasks":selected,"counts":{"valid_unseen":len(selected)}})
    alf["environment"]["manifest"] = str(alf_manifest)
    configs = {"shopsim":shop,"alfworld":alf}
    for name,cfg in configs.items():
        cfg["gpu"] = args.gpu
        cfg["environment"]["base_url"] = f"http://127.0.0.1:{port()}"
        cfg["runtime"].update(state_graph=True,memory_choice=True,shared_memory=True,
                              action_mode="grounded",loop_policy="state",max_turns=12 if name=="shopsim" else 24)
        cfg["evaluation"] = {"sequence_length":1,"seed":233}
        configs[name] = validate_config(cfg)
    ids = {"shopsim":shop_ids,"alfworld":[r["task_id"] for r in selected]}
    atomic_json(args.output/"receipt.json",{"tasks":ids,"config":configs,"conditions":args.conditions,"source_sha256":source_hashes(),
        "bridge_sha256":file_sha(BRIDGE),"original_alf_manifest_sha256":file_sha(original),
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":file_sha(args.checkpoint/"state.pt"),"cuda_allocator_cap_gib":26,
        "exposure":"Shop internal dev has historical full-pool predictions; ALF official unseen excludes earlier SCOPE pilot IDs"})
    return configs, ids


def audit_arm(output, expected, disabled):
    records = list(read_jsonl(output/"results.jsonl"))
    summary = read(output/"summary.json")
    if [r["task_id"] for r in records] != expected or file_sha(output/"results.jsonl") != summary["results_sha256"]:
        raise ContractError("Paired evaluation IDs or result digest changed")
    operations, diagnoses = Counter(), Counter()
    attempts, repeated = {}, 0
    for row in records:
        public = {e["id"]:e for e in row["public_evidence"]}
        for op in row["manager_operations"]:
            result = op["result"]
            if result.get("accepted") and result.get("operation"):
                operations[result["operation"]] += 1
            diagnosis = result.get("gap_diagnostic")
            if diagnosis:
                diagnoses[diagnosis["class"]] += 1
            for omission in result.get("menu_diagnostics",[]):
                diagnoses[omission["class"]] += 1
        rejected = None
        for step in row["history"]:
            if step["phase"] != "execute":
                continue
            if step.get("action_status") == "environment_invalid":
                key = (step["public_state_before"],step["command"])
                repeated += key == rejected
                rejected = key
            else:
                rejected = None
            confirmation = step.get("public_confirmation")
            if not confirmation:
                continue
            if any(public[eid]["revision"] <= confirmation["attempt_revision"] for eid in confirmation["evidence_ids"]):
                raise ContractError("A confirmation cites a pre-action observation")
            if confirmation["status"]=="confirmed" and not confirmation["evidence_ids"]:
                raise ContractError("Unsupported confirmation in the actor trace")
            attempts[(row["task_id"],confirmation["attempt_id"])] = confirmation["status"]
    if any(operations[tool] for tool in disabled):
        raise ContractError("Disabled tool executed")
    return {"success_count":summary["success_count"],"n":summary["n"],"mean_total_tokens":summary["mean_total_tokens"],
            "illegal_rate":summary["illegal_action_rate_per_turn"],"operations":dict(operations),
            "gap_diagnostics":dict(diagnoses),"confirmation_attempts":dict(Counter(attempts.values())),
            "evidence_usage":summary["evidence_usage"],"total_actor_turns":sum(r["actor_turns"] for r in records),
            "by_task":{r["task_id"]:r["metrics"]["success"] for r in records},
            "public_state_changes":sum(step.get("progress")=="state_changed" for row in records for step in row["history"]),
            "repeated_environment_rejections":repeated}


def action_contrast(left, right):
    """First action divergence before either policy visits a different public state."""
    changes = []
    a = list(read_jsonl(left/"results.jsonl"))
    b = list(read_jsonl(right/"results.jsonl"))
    for old,new in zip(a,b,strict=True):
        if old["task_id"]!=new["task_id"]:
            raise ContractError("Action comparison crossed task IDs")
        steps = ([h for h in row["history"] if h["phase"]=="execute"] for row in (old,new))
        for turn,(before,after) in enumerate(zip(*steps),1):
            if before["public_state_before"]!=after["public_state_before"]:
                break
            if before["command"]!=after["command"]:
                changes.append({"task_id":old["task_id"],"turn":turn,"before":before["command"],"after":after["command"],
                                "before_status":before["action_status"],"after_status":after["action_status"],
                                "before_progress":before["progress"],"after_progress":after["progress"]})
                break
    return changes


def native_probe(policy, config, output):
    from scope.credit import paired_local_samples
    from scope.rollout import EpisodeRunner
    from scope.training import paired_rollouts

    config = copy.deepcopy(config)
    runtime = config["runtime"]
    runtime.update(decision_evidence=True)
    registry = SplitRegistry(config["data_root"],233)
    tasks = sorted(registry.ids("rl_train"),key=lambda task:digest(["decision-credit-probe",task]))[:2]
    env_conf = config["environment"]
    env = ShopEnvironment(env_conf["base_url"],tasks,file_sha(env_conf["manifest"]))
    env.health()
    runner = EpisodeRunner(policy,env,runtime,"train")
    attempted = []
    initial = policy.fingerprint()
    for task in tasks:
        trajectory = runner.run(task,"decision-credit-probe",greedy=True,capture_phase="context",
                                capture_opportunity=True,probe_opportunities=("context_threshold",))
        attempted.append(task)
        if not trajectory.boundaries:
            continue
        boundary = trajectory.boundaries[0]
        try:
            baselines,candidates,seeds = paired_rollouts(runner,boundary,2,1,True,["decision-credit-probe",task])
            weighted,stats = paired_local_samples(boundary,candidates,baselines,.05)
            if policy.fingerprint()!=initial:
                raise ContractError("Native probe modified the frozen policy")
            atomic_json(output/"native_probe.json",{"complete":True,"attempted_tasks":attempted,
                "boundary_hash":boundary.fingerprint(),"seeds":seeds,"native_snapshot":True,
                "weighted_sample_count":len(weighted),"credit":stats})
            return
        finally:
            env.drop_snapshot(boundary.environment)
    atomic_json(output/"native_probe.json",{"complete":True,"attempted_tasks":attempted,"native_snapshot":True,
                                            "reason":"no_eligible_boundary"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--tasks-per-domain",type=int,choices=range(1,7),default=6)
    p.add_argument("--decision-comparison",action="store_true")
    p.add_argument("--native-probe",action="store_true")
    args = p.parse_args()
    args.conditions = ({"legacy_retrieval":{"decision_evidence":False},
                        "decision_evidence":{"decision_evidence":True},
                        "decision_no_retrieval":{"decision_evidence":True,"disabled_memory_tools":["RETRIEVE"]}}
                       if args.decision_comparison else
                       {arm:{"disabled_memory_tools":disabled} for arm,disabled in ARMS.items()})
    random.seed(233)
    np.random.seed(233)
    torch.manual_seed(233)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / total))
    configs, ids = prepare(args)
    model = configs["shopsim"]["model"]
    identity = model_identity(model["path"],args.project/"runs/.identity-cache")
    verify_checkpoint_base(args.checkpoint,identity)
    policy = HFPolicy(model["path"],rank=model["rank"],alpha=model["alpha"])
    load_checkpoint(args.checkpoint,policy,restore_random=False)
    report = {}
    for domain, cfg in configs.items():
        e = cfg["environment"]
        env = (ShopEnvironment(e["base_url"],ids[domain],file_sha(e["manifest"])) if domain=="shopsim"
               else ALFWorldEnvironment(e["base_url"],ids[domain],e["manifest"]))
        service = own_service(cfg,args.output/"shop_service",True) if domain=="shopsim" else alf_service(cfg,args.output)
        report[domain] = {}
        with service:
            env.health()
            for arm, settings in args.conditions.items():
                atomic_json(args.output/"status.json",{"phase":"evaluate","domain":domain,"arm":arm})
                effective = copy.deepcopy(cfg)
                effective["runtime"].update(settings)
                out = args.output/domain/arm
                evaluate(policy,env,FixedTasks(ids[domain]),effective,out,"internal_dev" if domain=="shopsim" else "test")
                report[domain][arm] = audit_arm(out,ids[domain],settings.get("disabled_memory_tools",[]))
                atomic_json(args.output/"partial_summary.json",report)
            if domain=="shopsim" and args.native_probe:
                atomic_json(args.output/"status.json",{"phase":"native_probe"})
                native_probe(policy,cfg,args.output)
        reference = next(iter(args.conditions))
        full = report[domain][reference]["by_task"]
        for arm in list(args.conditions)[1:]:
            control = report[domain][arm]["by_task"]
            report[domain][arm]["paired_vs_full"] = {
                "full_only":sum(full[t]>control[t] for t in full),"ablation_only":sum(full[t]<control[t] for t in full)}
            if args.decision_comparison:
                report[domain][arm]["first_action_changes"] = action_contrast(args.output/domain/reference,args.output/domain/arm)
    atomic_json(args.output/"summary.json",{"complete":True,"domains":report,"parameters_frozen":True})
    atomic_json(args.output/"status.json",{"phase":"complete"})


if __name__ == "__main__":
    main()
