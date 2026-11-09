"""Bounded real-model engineering validation, not a benchmark experiment."""
import copy
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
from scope.policy import HFPolicy
from scope.data import SplitRegistry
from scope.environments import ShopEnvironment
from scope.launcher import own_service
from scope.rollout import EpisodeRunner,manager_messages
from scope.contracts import Budget,Sample
from scope.atlas import StateAtlas
from scope.contracts import Scope
from scope.credit import local_samples
from scope.training import local_rollouts
from scope.checkpoint import save_checkpoint,load_checkpoint
from scope.io import atomic_json,canonical,digest


def main():
    torch.manual_seed(233);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    config=json.loads(Path("configs/shopsim.json").read_text())
    registry=SplitRegistry(config["data_root"])
    task=registry.ids("rl_train")[0]
    output=Path(os.environ.get("SCOPE_VALIDATION_OUTPUT","runs/validation_hf"))
    if (output/"result.json").exists():raise RuntimeError("Validation already completed; do not silently repeat")
    output.mkdir(parents=True,exist_ok=True)
    policy=HFPolicy(config["model"]["path"])
    optimizers={r:torch.optim.AdamW(policy.parameters(r),lr=1e-5,weight_decay=0.) for r in ("manager","actor")}
    atomic_json(output/"status.json",{"phase":"model_loaded"})
    with own_service(config,output,True):
        env=ShopEnvironment(config["environment"]["base_url"],registry.ids("rl_train"))
        try:
            goal=env.reset(task)
            atlas=StateAtlas(Scope("train",task,"checked_warmup"));atlas.append_public(goal,"requirement")
            messages=manager_messages(atlas,goal,{},goal,"state",Budget())
            text=canonical({"operation":"FINISH","expected_revision":atlas.revision})
            ids=policy.encode(messages);completion=policy.tokenizer.encode(text,add_special_tokens=False)+[policy.tokenizer.eos_token_id]
            sample=Sample("manager","state",ids,completion,[0.]*len(completion),text,0,"checked")
            before=policy.fingerprint("manager");actor=policy.fingerprint("actor")
            sft=policy.sft_update([sample],optimizers["manager"])
            checkpointing_active=bool(policy.model.training and policy.model.is_gradient_checkpointing)
            assert checkpointing_active,"HF gradient checkpointing is not active during replay"
            assert before!=policy.fingerprint("manager") and sft["grad_norm"]>0
            assert actor==policy.fingerprint("actor")
            checkpoint=save_checkpoint(output/"checkpoints",1,policy,optimizers,"engineering-validation")
            saved_parameter_hash=policy.fingerprint("manager")
            saved_optimizer=copy.deepcopy(optimizers["manager"].state_dict())
            captured=[]
            def capture_grad(opt,args,kwargs):
                captured.append([p.grad.detach().cpu().clone() if p.grad is not None else None for p in policy.parameters("manager")])
            hook=optimizers["manager"].register_step_pre_hook(capture_grad)
            # Exact next-step replay exercises real 8B adapter+optimizer restoration.
            policy.sft_update([sample],optimizers["manager"])
            uninterrupted=policy.fingerprint("manager")
            expected_state=policy.adapter_state("manager")
            load_checkpoint(checkpoint,policy,optimizers,"engineering-validation")
            roundtrip_parameters=policy.fingerprint("manager")==saved_parameter_hash
            restored_optimizer=optimizers["manager"].state_dict()
            moment_diff=max(float((v.cpu()-restored_optimizer["state"][k][name].cpu()).abs().max())
                            for k,vals in saved_optimizer["state"].items() for name,v in vals.items() if torch.is_tensor(v))
            policy.sft_update([sample],optimizers["manager"])
            observed=policy.adapter_state("manager")
            max_diff=max(float((observed[k]-expected_state[k]).abs().max()) for k in observed)
            grad_diff=max(float((a-b).abs().max()) for a,b in zip(captured[0],captured[1]) if a is not None)
            hook.remove()
            atomic_json(output/"resume_validation.json",{"bitwise_equal":uninterrupted==policy.fingerprint("manager"),"max_abs_diff":max_diff,
                        "roundtrip_parameters_equal":roundtrip_parameters,"optimizer_moment_max_abs_diff":moment_diff,"clipped_gradient_max_abs_diff":grad_diff,
                        "step_device_before":str(next(iter(saved_optimizer["state"].values()))["step"].device),
                        "step_device_after":str(next(iter(restored_optimizer["state"].values()))["step"].device)})
            assert uninterrupted==policy.fingerprint("manager"),f"Resumed next update diverged: max_abs_diff={max_diff}"
            atlas.close();env.close()
            runtime={**config["runtime"],"max_turns":2,"actor_max_new":64,"manager_max_new":32,"probe_query_horizon":1}
            runner=EpisodeRunner(policy,env,runtime,"train")
            atomic_json(output/"status.json",{"phase":"real_rollout"})
            outcome=runner.run(task,"hf_real",capture_phase="context",capture_turn=1)
            if not outcome.boundaries:raise RuntimeError("No local boundary captured")
            boundary=outcome.boundaries[0]
            try:
                candidates=local_rollouts(runner,boundary,[],2,"conditional")
                weighted,credit=local_samples(boundary,candidates)
                changed_before=policy.fingerprint("manager")
                update=policy.grpo_update(weighted,"manager",optimizers["manager"])
                changed_after=policy.fingerprint("manager")
            finally:env.drop_snapshot(boundary.environment)
            assert actor==policy.fingerprint("actor")
            atomic_json(output/"result.json",{"passed":True,"real_model":config["model"]["path"],"physical_gpu":2,
                "task_id":task,"task_split":"rl_train","engineering_only":True,
                "runtime_overrides":{"max_turns":2,"actor_max_new":64,"manager_max_new":32,"local_candidates":2},
                "sft_gradient":sft,"manager_parameters_changed_after_sft":True,"actor_parameters_unchanged":True,
                "gradient_checkpointing_active_during_replay":checkpointing_active,
                "next_update_resume_exact":True,"native_restore_verified":True,"local_credit":credit,
                "local_grpo_update":update,"manager_parameters_changed_after_local_grpo":changed_before!=changed_after,
                "local_reward_source":"real terminal metrics plus actual invalid/token costs; zero task success is not evidence of benefit",
                "metrics":outcome.metrics,"global_queries":1,"local_query_continuations":2,
                "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated()})
            atomic_json(output/"status.json",{"phase":"complete"})
        finally:env.close()


if __name__=="__main__":main()
