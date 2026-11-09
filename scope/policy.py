"""One HF backbone, independent manager/actor LoRA and fixed reference adapters."""
from __future__ import annotations
import copy
import hashlib
import math
from .contracts import ContractError, Sample


class HFPolicy:
    def __init__(self, model_path, device="cuda", rank=16, alpha=32, gradient_checkpointing=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                   torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                   attn_implementation="sdpa").to(self.device)
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.model = get_peft_model(base, cfg, adapter_name="manager")
        for name in ("actor", "manager_ref", "actor_ref"):
            self.model.add_adapter(name, copy.deepcopy(cfg))
        self.versions = {"manager": 0, "actor": 0}
        self.model_path = str(model_path)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.reset_reference("manager")
        self.reset_reference("actor")
        self.activate("manager")

    def parameters(self, role):
        return [p for name, p in self.model.named_parameters() if "lora_" in name and f".{role}." in name]

    def activate(self, role,training=False):
        if role not in {"manager","actor","manager_ref","actor_ref"}:raise ContractError("Unknown policy role")
        self.model.set_adapter(role)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("lora_" in name and f".{role}." in name and not role.endswith("_ref"))
        # HF gradient checkpointing is gated on training mode. Keep that path active for
        # replay, but disable stochastic dropout so the sampled policy is unchanged.
        self.model.train(training)
        if training:
            for module in self.model.modules():
                if isinstance(module,self.torch.nn.Dropout) or hasattr(module,"attention_dropout"):module.eval()

    def reset_reference(self, role):
        source = self.adapter_state(role)
        params = dict(self.model.named_parameters())
        with self.torch.no_grad():
            for name, value in source.items():
                target = name.replace(f".{role}.", f".{role}_ref.")
                params[target].copy_(value.to(params[target].device))

    def adapter_state(self, role):
        return {name: p.detach().cpu().clone() for name, p in self.model.named_parameters()
                if "lora_" in name and f".{role}." in name}

    def fingerprint(self, role=None):
        h = hashlib.sha256()
        for name, p in sorted(self.model.named_parameters()):
            if "lora_" in name and (role is None or f".{role}." in name):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def frozen_base_signature(self):
        return tuple((name,tuple(p.shape),str(p.dtype),p.data_ptr(),p._version,p.requires_grad)
                     for name,p in self.model.named_parameters() if "lora_" not in name)

    def encode(self, messages):
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False))

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def choice_ids(self, labels):
        tokens = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(ids) != 1 for ids in tokens) or len({tuple(ids) for ids in tokens}) != len(labels):
            raise ContractError("Choice labels must map to distinct single tokens")
        return [ids[0] for ids in tokens]

    def sample_choice(self, messages, phase, boundary, labels, max_input, greedy=False):
        torch = self.torch
        prompt = self.encode(messages)
        if not prompt or len(prompt) > max_input:
            raise ContractError("Choice prompt exceeds input budget")
        allowed = self.choice_ids(labels)
        sample = Sample("manager", phase, prompt, [allowed[0]], [], "", self.versions["manager"],
                        boundary, "choice", choice_token_ids=allowed)
        with torch.no_grad():
            probabilities = self.choice_logprobs(sample, "manager")
            index = int(probabilities.argmax()) if greedy else int(torch.multinomial(probabilities.exp(), 1))
        sample.text = labels[index]
        sample.completion_ids = [allowed[index]]
        sample.old_logprobs = [float(probabilities[index])]
        return sample

    def choice_logprobs(self, sample, role):
        self.activate(role, training=self.torch.is_grad_enabled() and not role.endswith("_ref"))
        ids = self.torch.tensor([sample.prompt_ids], device=self.device)
        logits = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids), use_cache=False,
                            logits_to_keep=1).logits[0, -1].float()
        return logits[sample.choice_token_ids].log_softmax(-1)

    def sample(self, messages, role, phase, boundary, max_input, max_new, greedy=False):
        import torch
        if role not in self.versions:raise ContractError("Reference adapters are not rollout policies")
        ids = self.encode(messages)
        if len(ids) > max_input:
            raise ContractError(f"Prompt exceeds {max_input} tokens; no unsafe token slicing")
        if max_new < 1:
            raise ContractError("Generation budget exhausted")
        self.activate(role)
        with torch.no_grad():
            result = self.model.generate(
                input_ids=torch.tensor([ids], device=self.device),
                attention_mask=torch.ones((1,len(ids)), dtype=torch.long, device=self.device),
                max_new_tokens=int(max_new), do_sample=not greedy, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, num_beams=1, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=True)
            tokens = result.sequences[0, len(ids):].tolist()
            logps = [float(score[0].float().log_softmax(-1)[token]) for score, token in zip(result.scores, tokens)]
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        termination="eos" if tokens and tokens[-1] in eos else ("length" if len(tokens)==max_new else "other_stop")
        return Sample(role, phase, ids, tokens, logps,
                      self.tokenizer.decode(tokens, skip_special_tokens=True), self.versions[role], boundary,
                      termination)

    def logprobs(self, sample, role=None):
        import torch
        role = role or sample.role
        if sample.choice_token_ids:
            allowed = sample.choice_token_ids
            if len(set(allowed)) != len(allowed) or len(sample.completion_ids) != 1 or sample.completion_ids[0] not in allowed:
                raise ContractError("Invalid categorical manager action")
            index = allowed.index(sample.completion_ids[0])
            return self.choice_logprobs(sample, role)[index:index+1]
        self.activate(role,training=torch.is_grad_enabled() and not role.endswith("_ref"))
        prompt = sample.prompt_ids
        completion = sample.completion_ids
        if not completion:
            raise ContractError("Empty action cannot train")
        if not prompt:raise ContractError("Empty prompt cannot train")
        ids = torch.tensor([prompt + completion], device=self.device)
        output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = output.logits[0, len(prompt)-1:len(prompt)+len(completion)-1].float()
        targets = torch.tensor(completion, device=self.device)
        # Only completion positions enter the loss; prompt/tool observations have zero action mask.
        return logits.log_softmax(-1).gather(-1, targets[:,None]).squeeze(-1)

    def grpo_update(self, weighted_samples, role, optimizer, clip=.2, beta=.01, max_norm=1.0):
        import torch
        usable = [(s,float(w)) for s,w in weighted_samples if s.role == role and s.completion_ids and abs(w)>0]
        optimizer.zero_grad(set_to_none=True)
        losses, kls = [], []
        for sample, weight in usable:
            if sample.origin != "policy":
                raise ContractError("Teacher demonstrations cannot enter on-policy GRPO")
            if weight<0 or not math.isfinite(weight) or not math.isfinite(sample.advantage) or len(sample.old_logprobs)!=len(sample.completion_ids):
                raise ContractError("Invalid loss weight or old-logprob mask")
            if sample.version != self.versions[role]:
                raise ContractError("Off-policy sample version is not the current collection version")
            with torch.no_grad():
                ref = self.logprobs(sample, role + "_ref")
            current = self.logprobs(sample, role)
            old = torch.tensor(sample.old_logprobs, device=self.device)
            ratio = (current-old).clamp(-20,20).exp()
            advantage = sample.advantage
            objective = torch.minimum(ratio*advantage, ratio.clamp(1-clip,1+clip)*advantage)
            delta = (ref-current).clamp(-20,20)
            kl = delta.exp()-delta-1
            loss = (-objective.mean()+beta*kl.mean())*weight
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GRPO loss")
            loss.backward()
            losses.append(float(loss.detach())); kls.append(float(kl.detach().mean()))
        params = self.parameters(role)
        norm = float(torch.nn.utils.clip_grad_norm_(params,max_norm)) if usable else 0.
        if not math.isfinite(norm):raise FloatingPointError("Non-finite policy gradient norm")
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError("Non-finite policy gradient")
        if usable:
            optimizer.step()
            self.versions[role] += 1
        optimizer.zero_grad(set_to_none=True)
        return {"loss":sum(losses), "kl":sum(kls)/max(1,len(kls)), "grad_norm":norm,
                "trained_actions":len(usable), "optimizer_step":bool(usable)}

    def sft_update(self, samples, optimizer, role="manager", max_norm=1.):
        import torch
        optimizer.zero_grad(set_to_none=True)
        if not samples:
            raise ContractError("No SFT examples")
        if any(sample.role!=role for sample in samples):raise ContractError("SFT role mismatch")
        loss_sum = 0.
        for sample in samples:
            loss = -self.logprobs(sample, role).mean()/len(samples)
            loss.backward(); loss_sum += float(loss.detach())
        norm = float(torch.nn.utils.clip_grad_norm_(self.parameters(role), max_norm))
        if not torch.isfinite(torch.tensor(norm)):
            raise FloatingPointError("Non-finite SFT gradient")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.versions[role] += 1
        return {"loss":loss_sum,"grad_norm":norm}
