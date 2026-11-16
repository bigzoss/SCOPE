"""Same-prefix, student-sampled HotpotQA management GRPO.

The public-stream seed fixes documents/IDs/menus. Candidate RNG and downstream
RNG are separate. Only the sampled boundary action enters the local loss.
"""
from dataclasses import asdict
import math
import statistics
import time

from .contracts import ContractError
from .credit import standardize
from .hotpot import NOOP, STAGES, rollout, score
from .hotpot_intervention import continuation_tokens
from .io import digest


def seed_for(*parts):
    return int(digest(parts)[:8], 16)


def opportunities(trace):
    """Public menu opportunities; no scores, gold supports, or teacher rewards."""
    result = []
    for index, action in enumerate(trace["actions"]):
        menu = action["menu"]
        if len(menu) < 2:
            continue
        stage = action["stage"]
        operations = {option["operation"] for option in menu.values()}
        if stage.startswith("write") and "STORE" in operations:
            reason = "public_information_to_store"
        elif stage == "compress" and "COMPRESS" in operations:
            reason = "working_context_can_be_compressed"
        elif stage.startswith("retrieve") and "RETRIEVE" in operations:
            reason = "unseen_public_excerpt_available"
        elif "UPDATE_STATE" in operations:
            reason = "public_evidence_supports_state_update"
        else:
            continue
        result.append({"stage_index": index, "stage": stage, "boundary": action["boundary"],
                       "reason": reason, "menu_size": len(menu)})
    return result


def select_opportunity(trace, preferred_stage):
    candidates = opportunities(trace)
    # Rotate the public opportunity type, rather than selecting with gold rewards.
    return min(candidates, key=lambda c: (c["stage_index"] - preferred_stage) % len(STAGES), default=None)


def paired_estimate(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ContractError("Paired returns must be finite and nonempty")
    deviation = statistics.stdev(values) if len(values) > 1 else None
    return {"mean": statistics.mean(values), "n": len(values), "sample_std": deviation,
            "standard_error": deviation / math.sqrt(len(values)) if deviation is not None else None}


def _decision_tokens(sample):
    return len(sample.prompt_ids) + len(sample.completion_ids)


def local_weights(candidates, controls, *, cost_weight=0., objective="group_relative", full_choice_ids=None):
    """Strict EM is primary. Cost is an unnormalized, successful compression tie-break.

    All candidates retain their on-policy sampling multiplicity. A forced noop
    is a measurement control, never an extra action in the GRPO sample group.
    """
    if len(candidates) < 2 or not controls or not 0 <= cost_weight <= .25:
        raise ContractError("Local GRPO requires >=2 student candidates and paired controls")
    if objective not in {"group_relative", "paired_delta"}:
        raise ContractError("Unknown local credit objective")
    anchor = candidates[0]["sample"]
    stage = candidates[0]["stage_index"]
    if type(stage) is not int or not 0 <= stage < len(STAGES):
        raise ContractError("Invalid local stage")
    seeds = [c["seed"] for c in controls]
    if len(set(seeds)) != len(seeds):
        raise ContractError("Continuation seeds must be unique")
    values, deltas, savings = [], [], []
    all_success = all(c["score"]["em"] == 1 and c["score"]["support_visible_recall"] == 1 for c in controls)
    for candidate in candidates:
        sample = candidate["sample"]
        if full_choice_ids is None:
            valid_mask = sample.choice_token_ids == anchor.choice_token_ids
        else:
            valid_mask = (bool(sample.choice_token_ids) and len(set(sample.choice_token_ids)) == len(sample.choice_token_ids)
                          and set(sample.choice_token_ids) <= set(full_choice_ids)
                          and set(sample.completion_ids) <= set(sample.choice_token_ids))
        if (sample.origin != "policy" or sample.role != "manager" or sample.phase != STAGES[stage]
                or sample.boundary != anchor.boundary or sample.prompt_ids != anchor.prompt_ids
                or sample.version != anchor.version or not valid_mask
                or candidate["stage_index"] != stage or not sample.completion_ids
                or len(sample.old_logprobs) != len(sample.completion_ids)):
            raise ContractError("Local credit crossed prompt, boundary, role, origin, mask, or policy version")
        if [run["seed"] for run in candidate["runs"]] != seeds:
            raise ContractError("Every candidate must share the control continuation seeds")
        strict = [run["score"]["em"] for run in candidate["runs"]]
        values.append(statistics.mean(strict))
        deltas.append(paired_estimate([a - b["score"]["em"] for a, b in zip(strict, controls, strict=True)]))
        retained = all(run["score"]["em"] == 1 and run["score"]["support_visible_recall"] == 1
                       for run in candidate["runs"])
        all_success = all_success and retained
        saving = 0.
        if candidate["operation"] == "COMPRESS" and retained:
            fractions = []
            for run, control in zip(candidate["runs"], controls, strict=True):
                # Both arms pay the SAME boundary inference cost. The forced
                # control has no actual call; its deployment cost is imputed.
                left = run["suffix_tokens"] + _decision_tokens(sample)
                right = control["suffix_tokens"] + _decision_tokens(sample)
                fractions.append((right-left) / max(1, left, right))
            saving = statistics.mean(fractions)
        savings.append(saving)
    group_advantages = standardize(values)
    # A shared intervention baseline cancels under within-group centering.
    # Preserve the signed paired effect, including homogeneous positive/negative
    # groups. The forced control is never an on-policy loss sample.
    task_advantages = ([d["mean"] for d in deltas] if objective == "paired_delta"
                       else group_advantages)
    constant = not any(task_advantages)
    advantages = list(task_advantages)
    if constant and all_success:
        advantages = [cost_weight * saving for saving in savings]
    weighted = []
    for candidate, advantage in zip(candidates, advantages, strict=True):
        sample = candidate["sample"]
        sample.advantage = advantage
        sample.credit_scope = "hotpot:local:" + STAGES[stage]
        # A singleton conditional draw has probability one for every parameter
        # setting. Its outcome helps compare the group but cannot train a choice.
        if advantage and (full_choice_ids is None or len(sample.choice_token_ids) > 1):
            weighted.append((sample, 1 / len(candidates)))
    return weighted, {"strict_returns": values, "paired_strict_deltas": deltas,
        "objective": objective, "group_relative_advantages": group_advantages,
        "group_zero_variance": not any(group_advantages),
        "task_advantages": task_advantages, "advantages": advantages,
        "task_zero_variance": not any(group_advantages), "task_zero_signal": constant,
        "cost_only_signal": constant and any(advantages),
        "all_failed": not any(values) and not any(c["score"]["em"] for c in controls),
        "compression_savings": savings, "cost_eligible_group": constant and all_success,
        "unique_choices": len({c["sample"].text for c in candidates}),
        "loss_owner": "sampled_boundary_manager_tokens_only", "teacher_rewards_used": False,
        "zero_signal_behavior": "skip_optimizer_and_kl_when_no_nonzero_advantage"}


def operation_effect_key(option):
    """Collapse labels whose tool execution and working-memory effect are identical."""
    operation = option['operation']
    if operation in {'STORE', 'RETRIEVE'}:
        payload = option['evidence']
    elif operation == 'UPDATE_STATE':
        payload = option['proposal']
    elif operation == 'COMPRESS':
        payload = option.get('refs')
    else:
        payload = None
    return digest([operation, payload])


def collect_local_group(policy, public, gold, anchor, stage, *, candidate_count=4, repeats=1,
                        seed=233, continuation="greedy", cost_weight=0., rollout_options=None,
                        objective="group_relative", candidate_sampling="independent"):
    started = time.perf_counter()
    if candidate_count < 2 or repeats < 1 or continuation not in {"greedy", "stochastic"}:
        raise ContractError("Invalid local group sampling configuration")
    if continuation == "greedy" and repeats != 1:
        raise ContractError("Repeated deterministic continuations are not independent evidence")
    if candidate_sampling not in {'independent', 'distinct_effects'}:
        raise ContractError('Unknown local candidate sampling strategy')
    if anchor["split"] != "rl_train" or anchor["task_id"] != public["id"]:
        raise ContractError("Local optimization is restricted to the source training split")
    if any(action.get("teacher_decision") is not None or
           (action.get("sample") and action["sample"].get("origin", "policy") != "policy")
           for action in anchor["actions"]):
        raise ContractError("Local anchor must come from a student rollout, not a teacher demonstration")
    if stage not in [item["stage_index"] for item in opportunities(anchor)]:
        raise ContractError("No public management opportunity at the selected stage")
    options = dict(rollout_options or {})
    reserved = {"factual", "intervention", "intervention_label", "resample_intervention", "decision_seed",
                "continuation_seed", "greedy_continuation", "chooser", "greedy", "decision_labels"}
    if reserved & options.keys():
        raise ContractError("Rollout options cannot override local sampling ownership")
    before_versions = dict(policy.versions)
    prefix = anchor["actions"][stage]
    effect_keys = {label: operation_effect_key(option) for label, option in prefix['menu'].items()}
    seen_effects = set()
    full_choice_ids = policy.choice_ids(list(prefix['menu'])) if candidate_sampling == 'distinct_effects' else None
    seeds = [seed_for(seed, "continuation", i) for i in range(repeats)]
    controls, candidates = [], []

    def replay(*, continuation_seed, **overrides):
        trace, samples = rollout(policy, public, "rl_train", anchor["seed"], factual=anchor,
            intervention=stage, continuation_seed=continuation_seed,
            greedy_continuation=continuation == "greedy", **options, **overrides)
        action = trace["actions"][stage]
        if (action["boundary"] != prefix["boundary"] or action["menu"] != prefix["menu"]
                or action["budget_before"] != prefix["budget_before"]
                or action["policy_versions"] != prefix["policy_versions"]):
            raise ContractError("Local branch did not preserve exact pre-action state/menu")
        return {"seed": continuation_seed, "score": score(trace, gold), "trace": trace,
                "suffix_tokens": sum(continuation_tokens(trace, stage).values())}, samples

    for continuation_seed in seeds:
        control, _ = replay(continuation_seed=continuation_seed)
        if control["trace"]["actions"][stage]["operation"] != NOOP[STAGES[stage]]:
            raise ContractError("Local control must preserve the current context")
        controls.append(control)
    for index in range(candidate_count):
        labels = None
        if candidate_sampling == 'distinct_effects':
            labels = [label for label, effect in effect_keys.items() if effect not in seen_effects] or list(effect_keys)
        first, samples = replay(continuation_seed=seeds[0], resample_intervention=True,
                                decision_seed=seed_for(seed, "candidate", index), decision_labels=labels)
        sample = samples[stage]
        if sample is None or sample.version != before_versions["manager"]:
            raise ContractError("Local candidate was not sampled by the frozen collection policy")
        if labels is not None and sample.choice_token_ids != policy.choice_ids(labels):
            raise ContractError('Sample log probabilities must use the recorded conditional action mask')
        seen_effects.add(effect_keys[sample.text])
        runs = [first]
        for continuation_seed in seeds[1:]:
            run, _ = replay(continuation_seed=continuation_seed, intervention_label=sample.text)
            runs.append(run)
        candidates.append({"sample": sample, "stage_index": stage,
            "operation": first["trace"]["actions"][stage]["operation"], "runs": runs})
    if policy.versions != before_versions:
        raise ContractError("Policy changed during a paired continuation group")
    weighted, metrics = local_weights(candidates, controls, cost_weight=cost_weight, objective=objective,
                                      full_choice_ids=full_choice_ids)
    metrics.update(unique_effects=len(seen_effects), candidate_sampling=candidate_sampling,
                   unique_final_views=len({digest(c['runs'][0]['trace']['answer_view']) for c in candidates}),
                   trainable_nonzero_credit=bool(weighted))
    for candidate in candidates:
        candidate["runs"][0]["trace"]["actions"][stage]["sample"] = asdict(candidate["sample"])
    actual_candidate_tokens = sum(run["suffix_tokens"] for c in candidates for run in c["runs"])
    actual_candidate_tokens += sum(_decision_tokens(c["sample"]) for c in candidates)
    # Full executed continuations retained for replay/audit; never feed scorer
    # fields or teacher scores back into the manager prompts.
    record = {"stage_index": stage, "stage": STAGES[stage], "boundary": prefix["boundary"],
        "public_prefix_sha256": digest(anchor["actions"][:stage]), "policy_versions": before_versions,
        "budget_before": prefix["budget_before"],
        "continuation": continuation, "continuation_seeds": seeds, "reader_greedy": True,
        "candidate_count": candidate_count, "repeats": repeats, "metrics": metrics,
        "controls": controls,
        "candidates": [{**c, "sample": asdict(c["sample"])} for c in candidates],
        "actual_sampling_tokens": {"candidates": actual_candidate_tokens,
                                   "controls": sum(c["suffix_tokens"] for c in controls)},
        "elapsed_seconds": time.perf_counter()-started,
        "cost_contract": "actual inference charged separately; both deployment arms include boundary decision cost"}
    return weighted, record
