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
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--allow-interaction-contrast",action="store_true",
                        help="For compare: allow only action_mode and loop_policy to differ")
    args=parser.parse_args(argv)
    if args.allow_interaction_contrast and args.command!="compare":parser.error("--allow-interaction-contrast requires compare")
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
    output=Path(args.output or "runs/"+args.command)
    if args.command=="preflight":
        report={"split_sizes":{k:len(v) for k,v in registry.rows.items()},"source_files":source_hashes(),
                "model_exists":Path(config["model"]["path"]).is_dir(),"gpu":config.get("gpu",2),
                "split_guard":"passed","service_port":config["environment"]["base_url"]}
        print(canonical(report));return
    if args.command=="warmup-build":
        from .warmup import build_examples
        if not args.public_file or not args.examples:parser.error("warmup-build requires --public-file and --examples")
        print(canonical(build_examples(args.public_file,args.examples,registry.ids(args.split))));return
    if args.command=="compare":
        from .evaluation import paired_comparison
        if not args.left or not args.right:parser.error("compare requires --left and --right")
        report=paired_comparison(args.left,args.right,allow_interaction_contrast=args.allow_interaction_contrast)
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
            report=warmup(policy,args.examples,optimizers,output,digest(receipt),args.epochs,args.batch_size,registry.ids("rl_train"),args.resume,args.stop_after)
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
                    report=evaluate(policy,env,registry,config,output,args.split,experience,args.limit)
                    print(canonical(report))
                elif args.command=="collect":
                    from .rollout import EpisodeRunner
                    tasks=registry.ids(args.split)[:args.limit or 10]
                    runner=EpisodeRunner(policy,env,config["runtime"],args.split,experience)
                    for index,task in enumerate(tasks):
                        outcome=runner.run(task,f"collect{index}",greedy=True)
                        evidence=[json.loads(r[1]) for r in outcome.final_atlas["tables"]["evidence"]]
                        append_jsonl(output/"public.jsonl",{"task_id":task,"evidence":evidence})
                    atomic_json(output/"summary.json",{"complete":True,"n":len(tasks),"public_only":True})
            finally:env.close()
    except BaseException as exc:
        atomic_json(output/"error.json",{"type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        cancelled=isinstance(exc,KeyboardInterrupt) or (isinstance(exc,SystemExit) and exc.code==128+signal.SIGTERM)
        atomic_json(output/"status.json",{"phase":"cancelled" if cancelled else "needs_attention","error":str(exc)})
        raise


if __name__=="__main__":main()
