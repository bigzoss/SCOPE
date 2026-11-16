"""Compute reuse and batched categorical loss for the full HotpotQA run."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
import math

from scope.contracts import ContractError
from scope.io import digest
from scope.policy import HFPolicy


class OptimizedPolicy(HFPolicy):
    choice_batch_size = 4
    batch_grpo = True

    @contextmanager
    def task_cache(self):
        self.cache = {}
        self.cache_stats = Counter()
        try:
            yield
        finally:
            self.cache = None

    def choice_logprobs(self, sample, role):
        cache = getattr(self, "cache", None)
        if cache is None or self.torch.is_grad_enabled():
            return super().choice_logprobs(sample, role)
        key = ("choice", role, self.versions.get(role), tuple(sample.prompt_ids), tuple(sample.choice_token_ids))
        if key in cache:
            self.cache_stats["manager_forward_hits"] += 1
            self.cache_stats["reused_input_tokens"] += len(sample.prompt_ids)
        else:
            cache[key] = super().choice_logprobs(sample, role)
            self.cache_stats["manager_forwards"] += 1
        return cache[key]

    def sample(self, messages, role, phase, boundary, max_input, max_new, greedy=False):
        cache = getattr(self, "cache", None)
        if cache is None or not greedy:
            return super().sample(messages, role, phase, boundary, max_input, max_new, greedy)
        key = ("generation", role, self.versions[role], phase, digest(messages), max_input, max_new)
        if key in cache:
            result = cache[key]
            self.cache_stats["reader_generation_hits"] += 1
            self.cache_stats["reused_input_tokens"] += len(result.prompt_ids)
            self.cache_stats["reused_output_tokens"] += len(result.completion_ids)
            return replace(result, boundary=boundary)
        self.cache_stats["reader_generations"] += 1
        result = super().sample(messages, role, phase, boundary, max_input, max_new, greedy)
        cache[key] = result
        return result

    def batch_choice_logprobs(self, keys, role):
        torch = self.torch
        self.activate(role, training=torch.is_grad_enabled() and not role.endswith("_ref"))
        width = max(len(prompt) for prompt, _ in keys)
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(prompt)) + list(prompt) for prompt, _ in keys],
                           device=self.device)
        mask = torch.tensor([[0] * (width - len(prompt)) + [1] * len(prompt) for prompt, _ in keys],
                            device=self.device)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        logits = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False, logits_to_keep=1).logits[:, -1].float()
        return [row[list(allowed)].log_softmax(-1) for row, (_, allowed) in zip(logits, keys, strict=True)]

    def grpo_update(self, weighted_samples, role, optimizer, clip=.2, beta=.01, max_norm=1.0):
        weighted_samples = list(weighted_samples)
        usable = [(sample, float(weight)) for sample, weight in weighted_samples
                  if sample.role == role and sample.completion_ids and abs(weight) > 0]
        if not self.batch_grpo or not usable or any(not sample.choice_token_ids for sample, _ in usable):
            return super().grpo_update(weighted_samples, role, optimizer, clip, beta, max_norm)

        torch = self.torch
        groups = defaultdict(list)
        for sample, weight in usable:
            if (sample.origin != "policy" or sample.version != self.versions[role]
                    or weight < 0 or not math.isfinite(weight) or not math.isfinite(sample.advantage)
                    or len(sample.completion_ids) != 1 or len(sample.old_logprobs) != 1
                    or not math.isfinite(sample.old_logprobs[0])):
                raise ContractError("Invalid on-policy categorical GRPO sample")
            allowed = tuple(sample.choice_token_ids)
            if len(set(allowed)) != len(allowed) or sample.completion_ids[0] not in allowed:
                raise ContractError("Invalid categorical GRPO action mask")
            groups[(tuple(sample.prompt_ids), allowed)].append(
                (allowed.index(sample.completion_ids[0]), sample.old_logprobs[0], sample.advantage, weight))
        keys = sorted(groups, key=lambda key: len(key[0]))
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, device=self.device)
        for start in range(0, len(keys), self.choice_batch_size):
            chunk = keys[start:start + self.choice_batch_size]
            with torch.no_grad():
                reference = self.batch_choice_logprobs(chunk, role + "_ref")
            current = self.batch_choice_logprobs(chunk, role)
            ref_selected = []
            current_selected = []
            targets = []
            for key, ref, logits in zip(chunk, reference, current, strict=True):
                items = groups[key]
                indices = [index for index, *_ in items]
                ref_selected.append(ref[indices])
                current_selected.append(logits[indices])
                targets.extend((old, advantage, weight) for _, old, advantage, weight in items)
            old, advantage, weights = torch.tensor(targets, device=self.device).unbind(-1)
            current_selected = torch.cat(current_selected)
            ratio = (current_selected - old).clamp(-20, 20).exp()
            objective = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
            delta = (torch.cat(ref_selected) - current_selected).clamp(-20, 20)
            kl = delta.exp() - delta - 1
            policy_loss = (-objective * weights).sum()
            kl_loss = (beta * kl * weights).sum()
            loss = policy_loss + kl_loss
            loss.backward()
            totals += torch.stack((loss.detach(), kl.detach().sum(), policy_loss.detach(), kl_loss.detach()))
        params = self.parameters(role)
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
        loss, kl, policy_loss, kl_loss = totals.tolist()
        if not all(math.isfinite(value) for value in (loss, kl, policy_loss, kl_loss)):
            raise FloatingPointError("Non-finite categorical GRPO loss")
        optimizer.step()
        self.versions[role] += 1
        optimizer.zero_grad(set_to_none=True)
        return {"loss": loss, "kl": kl / len(usable), "grad_norm": float(norm),
                "policy_loss": policy_loss, "kl_loss": kl_loss,
                "trained_actions": len(usable), "optimizer_step": True,
                "unique_forward_prompts": len(groups), "forward_batch_size": self.choice_batch_size}
