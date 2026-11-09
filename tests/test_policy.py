import copy
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import Qwen2Config,Qwen2ForCausalLM
from peft import LoraConfig,get_peft_model
from scope.policy import HFPolicy
from scope.contracts import Sample
from scope.checkpoint import save_checkpoint,load_checkpoint


def tiny_policy():
    torch.manual_seed(17)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128))
    config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
    model=get_peft_model(base,config,adapter_name="manager")
    for role in ("actor","manager_ref","actor_ref"):model.add_adapter(role,copy.deepcopy(config))
    policy=HFPolicy.__new__(HFPolicy);policy.model=model;policy.device=torch.device("cpu");policy.torch=torch
    policy.versions={"manager":0,"actor":0}
    policy.reset_reference("manager");policy.reset_reference("actor");policy.activate("manager")
    return policy


def action(policy):
    sample=Sample("manager","context",[1,2,3],[4,5],[0.,0.],"tool",policy.versions["manager"],"boundary",advantage=1.)
    sample.old_logprobs=policy.logprobs(sample).detach().tolist()
    return sample


def test_policy_gradient_role_isolation_and_exact_optimizer_resume(tmp_path):
    policy=tiny_policy()
    optimizers={r:torch.optim.AdamW(policy.parameters(r),lr=.001,weight_decay=0.) for r in ("manager","actor")}
    before=policy.fingerprint("manager");actor=policy.fingerprint("actor")
    info=policy.grpo_update([(action(policy),1.)],"manager",optimizers["manager"])
    assert info["grad_norm"]>0 and before!=policy.fingerprint("manager")
    assert actor==policy.fingerprint("actor")
    assert all(not p.requires_grad for name,p in policy.model.named_parameters() if "lora_" not in name)
    path=save_checkpoint(tmp_path,1,policy,optimizers,"receipt")
    policy.grpo_update([(action(policy),1.)],"manager",optimizers["manager"])
    expected=policy.fingerprint("manager")
    load_checkpoint(path,policy,optimizers,"receipt")
    policy.grpo_update([(action(policy),1.)],"manager",optimizers["manager"])
    assert policy.fingerprint("manager")==expected


def test_action_logprob_mask_matches_manual_cross_entropy():
    policy=tiny_policy();sample=action(policy)
    got=policy.logprobs(sample)
    ids=torch.tensor([[1,2,3,4,5]])
    output=policy.model(input_ids=ids,use_cache=False).logits[0]
    reference=torch.stack([output[2].log_softmax(-1)[4],output[3].log_softmax(-1)[5]])
    assert got.shape==(2,) and torch.allclose(got,reference,atol=1e-6)


def test_training_replay_enables_gradient_checkpointing_and_rejects_broadcast_logprobs():
    from scope.contracts import ContractError
    policy=tiny_policy();policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    sample=action(policy)
    logps=policy.logprobs(sample)
    assert policy.model.training and policy.model.is_gradient_checkpointing
    (-logps.mean()).backward()
    assert any(p.grad is not None for p in policy.parameters("manager"))
    sample.old_logprobs=[0.]
    opt=torch.optim.AdamW(policy.parameters("manager"),lr=.001)
    with pytest.raises(ContractError):policy.grpo_update([(sample,1.)],"manager",opt)


def test_checkpoint_rejects_alpha_change_before_mutating_weights(tmp_path):
    from scope.contracts import ContractError
    policy=tiny_policy()
    path=save_checkpoint(tmp_path,0,policy,{},"receipt")
    before=policy.fingerprint()
    policy.model.peft_config["manager"].lora_alpha=8
    with pytest.raises(ContractError,match="LoRA configuration"):load_checkpoint(path,policy,receipt_hash="receipt")
    assert policy.fingerprint()==before


def test_manager_sft_resume_has_exact_cursor_and_frozen_actor(tmp_path):
    from scope.warmup import warmup
    from scope.io import append_jsonl
    class Tokenizer:
        eos_token_id=6
        def encode(self,text,add_special_tokens=False):return [4,5]
    def make():
        p=tiny_policy();p.tokenizer=Tokenizer();p.encode=lambda messages:[1,2,3]
        opts={r:torch.optim.AdamW(p.parameters(r),lr=.001,weight_decay=0.) for r in ("manager","actor")}
        return p,opts
    examples=tmp_path/"data.jsonl"
    for _ in range(2):append_jsonl(examples,{"task_id":"1","phase":"state","messages":[],"completion":'{"operation":"FINISH","expected_revision":0}'})
    resumed,opts=make()
    first=warmup(resumed,examples,opts,tmp_path/"resume","receipt",allowed_tasks=["1"],stop_after=1)
    assert first["updates"]==1 and not first["complete"]
    # A crash after a non-checkpointed update must not duplicate the journal on resume.
    append_jsonl(tmp_path/"resume/warmup_metrics.jsonl",{"epoch":0,"update":2,"uncommitted":True})
    result=warmup(resumed,examples,opts,tmp_path/"resume","receipt",allowed_tasks=["1"],resume=first["checkpoint"])
    assert result["updates"]==2 and result["complete"] and result["actor_unchanged"]
    from scope.io import read_jsonl
    assert [r["update"] for r in read_jsonl(tmp_path/"resume/warmup_metrics.jsonl")]==[1,2]
    assert len(list((tmp_path/"resume").glob("uncommitted_warmup_metrics_*.jsonl")))==1
    full,full_opts=make()
    warmup(full,examples,full_opts,tmp_path/"full","receipt",allowed_tasks=["1"])
    assert resumed.fingerprint()==full.fingerprint()
