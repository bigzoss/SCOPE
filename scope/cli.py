from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import signal
from .contracts import ContractError
from .io import atomic_json,append_jsonl,canonical,digest,file_sha,read_jsonl


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    files=sorted((root/"scope").glob("*.py"))+[root/"pyproject.toml"]
    return {str(p.relative_to(root)):file_sha(p) for p in files}


def model_identity(model_path,cache_root):
    root=Path(model_path)
    files=sorted(p for p in root.iterdir() if p.is_file())
    signature={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in files}
    cache=Path(cache_root)/(digest([str(root),signature])+".json")
    if cache.exists():return json.loads(cache.read_text())
    result={"path":str(root),"stat_signature":signature,"sha256":{p.name:file_sha(p) for p in files}}
    atomic_json(cache,result)
    return result


def make_receipt(config,registry,output,command,identity_cache,extra=None):
    import torch,transformers,peft
    receipt={"method":"SCOPE-GRPO","config":config,"command":command,"extra":extra or {},
             "source_sha256":source_hashes(),"dataset_sha256":registry.hashes,
             "rl_train_ids_sha256":digest(sorted(registry.ids("rl_train"))),
             "internal_dev_ids_sha256":digest(sorted(registry.ids("internal_dev"))),
             "model":model_identity(config["model"]["path"],identity_cache),
             "environment":{"python":sys.version,"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__},
             "evaluation_exposure":("official target splits; external prior training/exposure is not certified"
                  if config["environment"].get("kind")=="alfworld_http" else
                  "dev previously used; original test900 and original task pool have historical predictions"),
             "resume_boundary":"completed_update","information_protocol":"full_instruction"}
    receipt["environment_manifest_sha256"]=file_sha(config["environment"]["manifest"])
    for key,path in (("warmstart",config.get("warmstart")),("experience_bank",config.get("experience_bank"))):
        if path:
            value=Path(path)
            receipt[key+"_sha256"]=file_sha(value/"state.pt" if value.is_dir() else value)
    path=Path(output)/"receipt.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old!=receipt:raise ContractError("Immutable execution receipt changed")
    else:atomic_json(path,receipt)
    return receipt


def verify_checkpoint_base(path,current_model):
    path=Path(path)
    metadata=json.loads((path/"metadata.json").read_text())
    if file_sha(path/"state.pt")!=metadata["state_sha256"]:raise ContractError("Checkpoint content hash mismatch")
    old=json.loads((path.parent.parent/"receipt.json").read_text())
    if digest(old)!=metadata["receipt_hash"]:raise ContractError("Originating checkpoint receipt hash mismatch")
    if old["model"]["sha256"]!=current_model["sha256"]:
        raise ContractError("Checkpoint was trained from a different frozen base model")


def main(argv=None):
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    parser=argparse.ArgumentParser(description="SCOPE evidence-grounded context GRPO")
    parser.add_argument("command",choices=["preflight","train","evaluate","collect","warmup-build","warmup","compare"])
    parser.add_argument("--config",default="configs/shopsim.json")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--manager-checkpoint")
    parser.add_argument("--manage-service",action="store_true")
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--split",choices=["rl_train","internal_dev","dev","test"],default="dev")
    parser.add_argument("--public-file")
    parser.add_argument("--examples")
    parser.add_argument("--experience-output",help="Optional training-only experience JSONL for warmup-build")
    parser.add_argument("--teacher-model", help="Local HF teacher checkpoint for offline warmup-build")
    parser.add_argument("--teacher-api-config", type=Path, help="Local USTC credential config; never logged")
    parser.add_argument("--warmup-role", choices=["manager", "actor"], default="manager")
    parser.add_argument("--teacher-intervention", choices=["recovery", "all"], default="recovery")
    parser.add_argument("--probe-output", type=Path, help="Export offline teacher probe hints from warmup-build")
    parser.add_argument("--teacher-max-calls", type=int)
    parser.add_argument("--teacher-eval", action="store_true", help="Evaluate with bounded USTC teacher recovery")
    parser.add_argument("--teacher-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--allow-interaction-contrast",action="store_true",
                        help="For compare: allow only action_mode and loop_policy to differ")
    parser.add_argument("--allow-teacher-contrast", action="store_true",
                        help="For compare: allow teacher assistance as the only evaluation contrast")
    args=parser.parse_args(argv)
    if args.experience_output and args.command!="warmup-build":parser.error("--experience-output requires warmup-build")
    if args.teacher_model and args.teacher_api_config:
        parser.error("Select one teacher backend")
    if args.teacher_model and (args.command != "warmup-build" or args.split != "rl_train"):
        parser.error("--teacher-model requires warmup-build --split rl_train")
    if args.teacher_eval and (args.command != "evaluate" or args.split != "test" or not args.teacher_api_config
                              or args.teacher_intervention != "recovery"):
        parser.error("--teacher-eval requires evaluate --split test, --teacher-api-config and recovery intervention")
    if args.teacher_api_config and not args.teacher_eval and (args.command not in {"warmup-build", "collect"} or args.split != "rl_train"):
        parser.error("--teacher-api-config requires teacher evaluation or warmup-build/collect --split rl_train")
    if args.warmup_role != "manager" and args.command != "warmup":
        parser.error("--warmup-role actor requires warmup")
    if args.probe_output and (args.command != "warmup-build" or not (args.teacher_model or args.teacher_api_config)):
        parser.error("--probe-output requires teacher-assisted warmup-build")
    if args.probe_output and args.probe_output.exists():
        parser.error("Probe output already exists")
    if args.teacher_max_calls is not None and args.teacher_max_calls < 1:parser.error("--teacher-max-calls must be >= 1")
    if args.allow_interaction_contrast and args.command!="compare":parser.error("--allow-interaction-contrast requires compare")
    if args.allow_teacher_contrast and (args.command != "compare" or args.allow_interaction_contrast):
        parser.error("--allow-teacher-contrast requires compare without an interaction contrast")
    if args.resume and args.command not in {"train","warmup"}:parser.error("Only train/warmup accept --resume; evaluation resumes complete sequence caches automatically")
    if args.stop_after is not None and args.command not in {"train","warmup"}:parser.error("--stop-after requires train/warmup")
    if args.command in {"train","warmup"}:args.split="rl_train"
    if args.limit is not None and args.limit<1:parser.error("--limit must be >= 1")
    if args.epochs<1 or args.batch_size<1:parser.error("epochs and batch size must be >= 1")
    if args.stop_after is not None and args.stop_after<0:parser.error("--stop-after must be >= 0")
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    from .config import validate_config
    config=validate_config(config)
    kind=config["environment"].get("kind","shopsim")
    if kind=="shopsim" and args.split=="test" and args.command in {"evaluate","collect"}:
        config["environment"]["manifest"]=str(Path(config["data_root"])/"manifests/test_unseen.json")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.get("gpu",2))
    if config.get("deterministic",True):os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    from .data import SplitRegistry
    if kind=="alfworld_http":
        from .alfworld import ALFWorldRegistry
        registry=ALFWorldRegistry(config["environment"]["manifest"],config.get("split_seed",233))
        if args.manage_service:raise ContractError("ALFWorld service is externally managed; SCOPE will not start or stop it")
        if args.command=="train" and config["training"]["method"]!="global":
            raise ContractError("ALFWorld public-v1 cannot support native local comparisons; select an explicit global-only recipe")
    else:registry=SplitRegistry(config["data_root"],config.get("split_seed",233))
    if args.teacher_eval:
        from .teacher_online import TEACHER
        config["evaluation"]["teacher"] = TEACHER
        if args.teacher_max_calls is None:
            n = len(registry.ids("test"))
            args.teacher_max_calls = min(n, args.limit or n) * TEACHER["max_proposals_per_task"] * 3
    elif args.teacher_max_calls is None:
        args.teacher_max_calls = 24
    output=Path(args.output or "runs/"+args.command)
    if args.command == "collect" and args.teacher_api_config and (output/"public.jsonl").exists():
        raise ContractError("Teacher collection already has records; use a distinct output to avoid duplicate demonstrations")
    if args.command=="preflight":
        report={"split_sizes":{k:len(v) for k,v in registry.rows.items()},"source_files":source_hashes(),
                "model_exists":Path(config["model"]["path"]).is_dir(),"gpu":config.get("gpu",2),
                "split_guard":"passed","service_port":config["environment"]["base_url"]}
        print(canonical(report));return
    if args.command=="warmup-build":
        from .warmup import build_examples,graph_adapter
        if not args.public_file or not args.examples:parser.error("warmup-build requires --public-file and --examples")
        if args.experience_output and args.split != "rl_train":
            raise ContractError("Experience extraction requires --split rl_train")
        teacher = None
        if args.teacher_model or args.teacher_api_config:
            if not config["runtime"]["state_graph"]:
                raise ContractError("Teacher distillation requires runtime.state_graph")
            from .teacher import LocalTeacher,TeacherDistillation
            if args.teacher_api_config:
                from .teacher_api import APITeacher
                backend = APITeacher(args.teacher_api_config, output / "teacher_cache", args.teacher_max_calls)
            else:
                backend = LocalTeacher(args.teacher_model, args.teacher_device)
            teacher = TeacherDistillation(backend, args.teacher_max_calls)
        try:
            report = build_examples(args.public_file,args.examples,registry.ids(args.split),config,teacher)
        finally:
            if teacher and hasattr(teacher.generate, "close"):
                teacher.generate.close()
        if teacher:
            report["teacher"]["api_usage"] = dict(getattr(teacher.generate, "counts", {}))
            report["teacher"]["public_corpus_sha256"] = file_sha(args.public_file)
            report["teacher"]["examples_sha256"] = file_sha(args.examples)
        if args.probe_output:
            from .io import atomic_jsonl
            hints = [{"source":"teacher_public_probe", "task_id":r["task_id"], "phase":r["phase"],
                      "prefix_hash":r["teacher_probe"]["prefix_hash"]} for r in read_jsonl(args.examples)
                     if r.get("teacher_probe", {}).get("worthy")]
            atomic_jsonl(args.probe_output, hints)
            report["teacher_probes"] = {"count":len(hints), "sha256":file_sha(args.probe_output)}
        if args.experience_output:
            from .experience import build_experiences
            report["experience"] = build_experiences(args.public_file,args.experience_output,
                registry.ids("rl_train"),graph_adapter(kind))
        print(canonical(report));return
    if args.command=="compare":
        from .evaluation import paired_comparison
        if not args.left or not args.right:parser.error("compare requires --left and --right")
        report=paired_comparison(args.left,args.right,allow_interaction_contrast=args.allow_interaction_contrast,
                                 allow_teacher_contrast=args.allow_teacher_contrast)
        atomic_json(output/"paired.json",report);print(canonical(report));return
    import random
    import numpy as np
    import torch
    random.seed(config.get("seed",233));np.random.seed(config.get("seed",233));torch.manual_seed(config.get("seed",233))
    torch.set_num_threads(4)
    if config.get("deterministic",True):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    from .policy import HFPolicy
    from .checkpoint import load_checkpoint
    from .experience import ExperienceBank
    from .environments import ShopEnvironment
    from .launcher import own_service
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        resumed=Path(args.resume).resolve()
        latest_path=output/"checkpoints/latest.json"
        if resumed.parent!=(output/"checkpoints").resolve() or not latest_path.exists():raise ContractError("Resume must use this run's checkpoint directory; use warmstart for another run")
        latest=json.loads(latest_path.read_text())
        if resumed.name!=latest["path"]:raise ContractError("Resume requires the latest committed checkpoint; older checkpoints need a distinct warmstart run")
    import atexit
    import fcntl
    lock=(output/".run.lock").open("a")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ContractError("Another process owns this run directory")
    atexit.register(lock.close)
    if args.command in {"train","warmup","collect"} and not args.resume and ((output/"summary.json").exists() or (output/"checkpoints/latest.json").exists()):
        raise ContractError("Run already has results/checkpoints; use resume or a distinct output")
    extra={"split":args.split,"limit":args.limit,"epochs":args.epochs,"batch_size":args.batch_size}
    extra["warmup_role"] = args.warmup_role
    if args.teacher_eval:
        extra["teacher"] = {**TEACHER, "request_budget": args.teacher_max_calls}
    elif args.teacher_api_config:
        # Include identity, never credential path/content, in the reproducibility receipt.
        teacher_cfg = json.loads(args.teacher_api_config.read_text(encoding="utf-8-sig"))
        extra["teacher"] = {"provider":"USTC", "model":teacher_cfg.get("model", "qwen3.8-flash-next"),
                            "request_budget":args.teacher_max_calls,"intervention":args.teacher_intervention}
    for field in ("checkpoint","actor_checkpoint","manager_checkpoint","examples","public_file"):
        value=getattr(args,field)
        if value:
            path=Path(value);extra[field]={"path":str(path),"sha256":file_sha(path/"state.pt" if path.is_dir() else path)}
    receipt=make_receipt(config,registry,output,args.command,Path("runs/.identity-cache"),extra)
    try:
        for checkpoint in (args.checkpoint,args.actor_checkpoint,args.manager_checkpoint,config.get("warmstart")):
            if checkpoint:verify_checkpoint_base(checkpoint,receipt["model"])
        policy=HFPolicy(config["model"]["path"],rank=config["model"].get("rank",16),alpha=config["model"].get("alpha",32),
                        gradient_checkpointing=config["model"].get("gradient_checkpointing",True))
        if args.checkpoint:load_checkpoint(args.checkpoint,policy,restore_random=False)
        for role in ("actor","manager"):
            selected=getattr(args,role+"_checkpoint")
            if selected:
                load_checkpoint(selected,policy,restore_random=False,roles={role})
        if args.command=="warmup":
            from .warmup import warmup
            if not args.examples:parser.error("warmup requires --examples")
            optimizers={r:torch.optim.AdamW(policy.parameters(r),lr=config["training"].get("lr",1e-5),weight_decay=0.) for r in ("manager","actor")}
            report=warmup(policy,args.examples,optimizers,output,digest(receipt),args.epochs,args.batch_size,
                          registry.ids("rl_train"),args.resume,args.stop_after,
                          graph_token_budget=(config["runtime"][args.warmup_role+"_max_input"], config["runtime"][args.warmup_role+"_max_new"]),
                          memory_choice=config["runtime"]["memory_choice"],role=args.warmup_role)
            print(canonical(report));return
        experience=ExperienceBank(config.get("experience_bank"),registry.ids("rl_train"))
        with own_service(config,output,args.manage_service):
            allowed=registry.ids("rl_train") if args.command=="train" else registry.ids(args.split)
            if kind=="alfworld_http":
                from .alfworld import ALFWorldEnvironment
                env=ALFWorldEnvironment(config["environment"]["base_url"],allowed,config["environment"]["manifest"])
            else:env=ShopEnvironment(config["environment"]["base_url"],allowed,receipt["environment_manifest_sha256"])
            env.health()
            try:
                if args.command=="train":
                    from .training import train
                    train(policy,env,registry,config,output,receipt,experience,args.resume,args.stop_after)
                elif args.command=="evaluate":
                    from .evaluation import evaluate
                    if args.teacher_eval:
                        from .teacher_online import OnlineRecovery
                        teacher = OnlineRecovery(args.teacher_api_config, output / "teacher_cache", args.teacher_max_calls)
                        try:
                            report = evaluate(policy, env, registry, config, output, args.split, experience,
                                              args.limit, actor_teacher=teacher)
                        finally:
                            teacher.close()
                    else:
                        report=evaluate(policy,env,registry,config,output,args.split,experience,args.limit)
                    print(canonical(report))
                elif args.command=="collect":
                    from .rollout import EpisodeRunner
                    tasks=registry.ids(args.split)[:args.limit or 10]
                    backend, guide = None, None
                    if args.teacher_api_config:
                        from .teacher_api import APITeacher
                        from .teacher_execution import ExecutionTeacher
                        backend=APITeacher(args.teacher_api_config,output/"teacher_cache",args.teacher_max_calls)
                        guide=ExecutionTeacher(backend,args.teacher_intervention)
                    runner=EpisodeRunner(policy,env,config["runtime"],args.split,experience,actor_teacher=guide)
                    demonstrations, successes = 0, 0
                    try:
                        for index,task in enumerate(tasks):
                            outcome=runner.run(task,f"collect{index}",greedy=True)
                            evidence=[json.loads(r[1]) for r in outcome.final_atlas["tables"]["evidence"]]
                            append_jsonl(output/"public.jsonl",{"task_id":task,"evidence":evidence})
                            successes += int(outcome.metrics["success"])
                            if guide:
                                append_jsonl(output/"teacher_rollouts.jsonl",{"task_id":task,"metrics":outcome.metrics,
                                    "events":outcome.events,"costs":outcome.costs,"termination_reason":outcome.termination_reason})
                                for event in outcome.events:
                                    label=event.get("teacher_demonstration")
                                    if label and event["action_status"] == "accepted" and outcome.metrics["success"] == 1:
                                        append_jsonl(output/"actor_examples.jsonl",{"task_id":task,"phase":"execute",
                                            "source":"teacher_actor_public_prefix","messages":event["student_messages"],
                                            "completion":label["text"],"environment_accepted":True,"task_success":1,
                                            "subgoal":label["subgoal"],"evidence_ids":label["evidence_ids"]})
                                        demonstrations += 1
                            atomic_json(output/"status.json",{"phase":"collecting","completed":index+1,"expected":len(tasks)})
                        atomic_json(output/"summary.json",{"complete":True,"n":len(tasks),"public_only":True,
                            "teacher_assisted":bool(guide),"success_count":successes,"actor_examples":demonstrations,
                            "teacher_usage":dict(backend.counts) if backend else {},
                            "student_performance_claim":False if guide else None})
                    finally:
                        if backend:backend.close()
            finally:env.close()
    except BaseException as exc:
        atomic_json(output/"error.json",{"type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        cancelled=isinstance(exc,KeyboardInterrupt) or (isinstance(exc,SystemExit) and exc.code==128+signal.SIGTERM)
        atomic_json(output/"status.json",{"phase":"cancelled" if cancelled else "needs_attention","error":str(exc)})
        raise


if __name__=="__main__":main()
