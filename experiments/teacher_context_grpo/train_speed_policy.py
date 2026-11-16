"""Lower overhead for categorical GRPO while preserving serial training forwards."""
import math
from time import perf_counter

from optimized_policy import OptimizedPolicy
from scope.contracts import ContractError
from scope.policy import HFPolicy


class TrainingSpeedPolicy(OptimizedPolicy):
    batch_grpo = False
    short_no_checkpointing_tokens = 0
    fast_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_layers = [layer for layer in self.model.modules()
                                   if getattr(layer, "gradient_checkpointing", False)]
        self._checkpointing_enabled = True

    def set_activation_checkpointing(self, enabled):
        if enabled != self._checkpointing_enabled:
            for layer in self._checkpoint_layers:
                layer.gradient_checkpointing = enabled
            self._checkpointing_enabled = enabled

    def activate(self, role, training=False):
        active = (role, training)
        if getattr(self, "_update_activation", None) == active:
            return
        super().activate(role, training)
        if hasattr(self, "_update_activation"):
            self._update_activation = active

    def grpo_update(self, weighted_samples, role, optimizer, clip=.2, beta=.01, max_norm=1.):
        weighted_samples = list(weighted_samples)
        usable = [(sample, float(weight)) for sample, weight in weighted_samples
                  if sample.role == role and sample.completion_ids and abs(weight) > 0]
        if not self.fast_update or not usable or any(not sample.choice_token_ids for sample, _ in usable):
            return HFPolicy.grpo_update(self, weighted_samples, role, optimizer, clip, beta, max_norm)

        started = perf_counter()
        torch = self.torch
        for sample, weight in usable:
            if (sample.origin != "policy" or sample.version != self.versions[role]
                    or weight < 0 or not math.isfinite(weight) or not math.isfinite(sample.advantage)
                    or len(sample.completion_ids) != 1 or len(sample.old_logprobs) != 1
                    or not math.isfinite(sample.old_logprobs[0])):
                raise ContractError("Invalid on-policy categorical GRPO sample")
            if (len(set(sample.choice_token_ids)) != len(sample.choice_token_ids)
                    or sample.completion_ids[0] not in sample.choice_token_ids):
                raise ContractError("Invalid categorical GRPO action mask")

        optimizer.zero_grad(set_to_none=True)
        references = {}
        self._update_activation = None
        try:
            with torch.no_grad():
                for sample, _ in usable:
                    key = (tuple(sample.prompt_ids), tuple(sample.choice_token_ids))
                    if key not in references:
                        references[key] = self.choice_logprobs(sample, role + "_ref")
            rows = []
            no_checkpointing_actions = 0
            for sample, weight in usable:
                key = (tuple(sample.prompt_ids), tuple(sample.choice_token_ids))
                index = sample.choice_token_ids.index(sample.completion_ids[0])
                ref = references[key][index:index+1]
                keep_checkpointing = len(sample.prompt_ids) > self.short_no_checkpointing_tokens
                self.set_activation_checkpointing(keep_checkpointing)
                no_checkpointing_actions += not keep_checkpointing
                current = self.logprobs(sample, role)
                old = torch.tensor(sample.old_logprobs, device=self.device)
                ratio = (current-old).clamp(-20, 20).exp()
                advantage = sample.advantage
                objective = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
                delta = (ref-current).clamp(-20, 20)
                kl = delta.exp()-delta-1
                policy_loss = -objective.mean()*weight
                kl_loss = beta*kl.mean()*weight
                loss = (-objective.mean()+beta*kl.mean())*weight
                loss.backward()
                rows.append(torch.stack((loss.detach(), kl.detach().mean(),
                                         policy_loss.detach(), kl_loss.detach())))

            values = torch.stack(rows).cpu().tolist()
            if any(not math.isfinite(value) for row in values for value in row):
                raise FloatingPointError("Non-finite categorical GRPO loss or KL")
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(role), max_norm, error_if_nonfinite=True)
            optimizer.step()
            self.versions[role] += 1
            loss, kl, policy_loss, kl_loss = map(sum, zip(*values, strict=True))
            return {"loss": loss, "kl": kl / len(usable), "grad_norm": float(norm),
                    "policy_loss": policy_loss, "kl_loss": kl_loss,
                    "trained_actions": len(usable), "optimizer_step": True,
                    "reference_forward_count": len(references),
                    "optimizer_seconds": perf_counter() - started,
                    "actions_without_activation_checkpointing": no_checkpointing_actions}
        finally:
            self.set_activation_checkpointing(True)
            optimizer.zero_grad(set_to_none=True)
            del self._update_activation
