"""Controlled mechanism tests; these are not benchmark performance results."""
import copy
import json
import random
from types import SimpleNamespace

import pytest

from scope.contracts import ContractError, Sample
from scope.hotpot import rollout, split_row
from scope.hotpot_local_credit import collect_local_group, local_weights, select_opportunity
from scope.hotpot_provenance import memory_flow


class SeededReader:
    def __init__(self):
        self.rng = random.Random()
        self.torch = SimpleNamespace(manual_seed=self.rng.seed)
        self.versions = {"manager": 0, "actor": 0}
        self.calls = []

    def choice_ids(self, labels):
        return [ord(label) for label in labels]

    def sample_choice(self, messages, stage, boundary, labels, max_input, greedy):
        payload = json.loads(messages[-1]["content"])
        menu = payload.get("actions", payload.get("verified_options"))
        # A fixed downstream manager deliberately does not rescue a lost write.
        if greedy:
            op = "STORE" if stage.startswith("write") else "KEEP" if stage == "compress" else "FINISH"
            label = next(k for k, v in menu.items() if v["operation"] == op)
        else:
            label = self.rng.choice(labels)
        self.calls.append((stage, boundary, label, greedy))
        prompt = list(messages[-1]["content"].encode())
        return Sample("manager", stage, prompt, [ord(label)], [-1.], label, 0, boundary,
                      choice_token_ids=[ord(k) for k in labels])

    def sample(self, messages, role, phase, boundary, max_input, max_new, greedy):
        payload = json.loads(messages[-1]["content"])
        answer = "Glasgow" if any("Glasgow" in row["text"] for row in payload["evidence"]) else "unknown"
        return Sample(role, phase, list(messages[-1]["content"].encode()), [1], [0.], answer, 0, boundary)


def task():
    return split_row({"id": "local-test", "question": "Which city?", "answer": "Glasgow",
        "supporting_facts": [["Primary", 0]], "context": [["Primary", ["The city is Glasgow."]]]})


def group(seed=233, **kwargs):
    policy = SeededReader()
    public, gold = task()
    anchor, _ = rollout(policy, public, "rl_train", 1, greedy=True)
    weighted, record = collect_local_group(policy, public, gold, anchor, 1,
                                           seed=seed, candidate_count=8, **kwargs)
    return policy, anchor, weighted, record


def test_same_prefix_local_group_trains_only_sampled_boundary_and_counts_costs():
    policy, anchor, weighted, record = group()
    assert record["metrics"]["unique_choices"] == 2
    assert record["metrics"]["strict_returns"] == [
        int(c["operation"] == "STORE") for c in record["candidates"]]
    assert weighted and all(s.phase == "write_2" and s.origin == "policy" for s, _ in weighted)
    assert all(s.boundary == anchor["actions"][1]["boundary"] for s, _ in weighted)
    assert len({tuple(s.prompt_ids) for s, _ in weighted}) == 1
    assert sum(s.advantage for s, _ in weighted) == pytest.approx(0.)
    for candidate in record["candidates"]:
        sampled = candidate["sample"]
        assert (sampled["advantage"] > 0) == (candidate["operation"] == "STORE")
        assert candidate["runs"][0]["trace"]["actions"][1]["sample"] == sampled
    controls = record["controls"]
    assert all(c["trace"]["actions"][1]["sample"] is None for c in controls)
    measured = sum(run["suffix_tokens"] for c in record["candidates"] for run in c["runs"])
    measured += sum(len(c["sample"]["prompt_ids"]) + 1 for c in record["candidates"])
    assert record["actual_sampling_tokens"]["candidates"] == measured
    assert policy.versions == {"manager": 0, "actor": 0}


def test_distinct_effect_sampling_replays_the_conditional_action_mask():
    _, anchor, weighted, record = group(candidate_sampling='distinct_effects')
    first, second = [r['sample'] for r in record['candidates'][:2]]
    assert first['completion_ids'] != second['completion_ids']
    assert second['choice_token_ids'] == [t for t in first['choice_token_ids'] if t not in first['completion_ids']]
    assert first['boundary'] == second['boundary'] == anchor['actions'][1]['boundary']
    assert first['prompt_ids'] == second['prompt_ids']
    assert record['metrics']['unique_effects'] == 2
    assert record['metrics']['unique_final_views'] == 2
    assert all(len(sample.choice_token_ids) > 1 for sample, _ in weighted)


def test_stochastic_continuation_seeds_are_paired_and_candidate_rng_is_independent():
    _, _, _, one = group(continuation="stochastic", repeats=3)
    _, _, _, two = group(continuation="stochastic", repeats=2)
    assert [c["sample"]["text"] for c in one["candidates"]] == [c["sample"]["text"] for c in two["candidates"]]
    for candidate in one["candidates"]:
        assert [r["seed"] for r in candidate["runs"]] == one["continuation_seeds"]
        assert all(r["trace"]["actions"][1]["choice"] == candidate["sample"]["text"] for r in candidate["runs"])
        assert all(r["trace"]["actions"][1]["sample"] is None for r in candidate["runs"][1:])
    # Identical sampled choices must yield identical continuations on each paired seed.
    seen = {}
    for candidate in one["candidates"]:
        for run in candidate["runs"]:
            key = (candidate["sample"]["text"], run["seed"])
            decisions = ([a["choice"] for a in run["trace"]["actions"][2:]], run["trace"]["answer"])
            assert seen.setdefault(key, decisions) == decisions
    assert all(delta["standard_error"] is not None for delta in one["metrics"]["paired_strict_deltas"])
    with pytest.raises(ContractError, match="deterministic"):
        group(repeats=2)


def synthetic_group(operations, returns, *, baseline=1, support=1, costs=None):
    controls = [{"seed": 3, "score": {"em": baseline, "support_visible_recall": 1}, "suffix_tokens": 100}]
    candidates = []
    for index, (op, result) in enumerate(zip(operations, returns, strict=True)):
        sample = Sample("manager", "compress", [1, 2], [index+3], [-1.], str(index), 0, "same",
                        choice_token_ids=list(range(3, 3+len(operations))))
        candidates.append({"sample": sample, "stage_index": 2, "operation": op,
            "runs": [{"seed": 3, "score": {"em": result, "support_visible_recall": support},
                      "suffix_tokens": costs[index] if costs else 50}]})
    return candidates, controls


def test_all_failed_or_tied_noncompression_does_not_manufacture_reward():
    candidates, controls = synthetic_group(["KEEP", "COMPRESS"], [0, 0], baseline=0)
    weighted, info = local_weights(candidates, controls, cost_weight=.1)
    assert weighted == [] and info["all_failed"] and info["task_zero_variance"]
    candidates, controls = synthetic_group(["STORE", "STORE"], [1, 1])
    assert local_weights(candidates, controls, cost_weight=.1)[0] == []
    controls[0]["score"]["em"] = 0
    weighted, info = local_weights(candidates, controls, cost_weight=.1)
    assert weighted == [] and all(d["mean"] == 1 for d in info["paired_strict_deltas"])


def test_compression_is_small_successful_tiebreak_with_equal_boundary_cost():
    candidates, controls = synthetic_group(["KEEP", "COMPRESS"], [1, 1], costs=[100, 50])
    weighted, info = local_weights(candidates, controls, cost_weight=.1)
    assert len(weighted) == 1 and weighted[0][0].text == "1"
    assert weighted[0][0].advantage == pytest.approx(.1*50/103)
    assert info["cost_only_signal"] and not any(info["task_advantages"])
    candidates[1]["runs"][0]["score"]["support_visible_recall"] = .5
    assert local_weights(candidates, controls, cost_weight=.1)[0] == []
    candidates[1]["runs"][0]["score"]["em"] = 0
    weighted, info = local_weights(candidates, controls, cost_weight=.1)
    assert info["advantages"][1] < 0 and not info["cost_only_signal"]


@pytest.mark.parametrize("field,value", [("origin", "teacher"), ("version", 1),
    ("boundary", "other"), ("role", "actor"), ("prompt_ids", [9]), ("old_logprobs", [])])
def test_local_loss_rejects_teacher_and_mismatched_samples(field, value):
    candidates, controls = synthetic_group(["KEEP", "COMPRESS"], [1, 0])
    setattr(candidates[1]["sample"], field, value)
    with pytest.raises(ContractError):
        local_weights(candidates, controls)


def test_wrong_continuation_seed_or_validation_split_is_rejected():
    candidates, controls = synthetic_group(["KEEP", "COMPRESS"], [1, 0])
    candidates[1]["runs"][0]["seed"] = 99
    with pytest.raises(ContractError, match="seeds"):
        local_weights(candidates, controls)
    public, gold = task()
    policy = SeededReader()
    trace, _ = rollout(policy, public, "validation", 1, greedy=True)
    with pytest.raises(ContractError, match="training split"):
        collect_local_group(policy, public, gold, trace, 1)
    trace["split"] = "rl_train"
    trace["actions"][1]["teacher_decision"] = {"choice": "A"}
    with pytest.raises(ContractError, match="teacher demonstration"):
        collect_local_group(policy, public, gold, trace, 1)


def test_public_opportunity_and_provenance_do_not_claim_causation():
    _, anchor, _, _ = group()
    opportunity = select_opportunity(anchor, 1)
    assert opportunity["reason"] == "public_information_to_store"
    flow = anchor["memory_flow"]
    assert not flow["contains_gold"]
    assert any(e["relation"] == "DELIVERED_TO" for e in flow["edges"])
    assert all(e["relation"] != "CAUSES_SUCCESS" for e in flow["edges"])
    broken = copy.deepcopy(anchor)
    broken["answer_view"][0]["end"] = 100000
    with pytest.raises(ContractError):
        memory_flow(broken)


def test_alternating_driver_routes_local_and_global_losses():
    pytest.importorskip("pyarrow")
    from experiments.memory_curriculum.hotpot_grpo import step_data
    policy = SeededReader()
    local, record = step_data(policy, task(), 2, "alternating", 1, {}, 0., local_candidates=8)
    assert record["update_kind"] == "local" and not record["global_advantages"]
    assert record["local_group"]["candidate_count"] == 8
    assert all(s.credit_scope.startswith("hotpot:local:") for s, _ in local)
    _, record = step_data(policy, task(), 3, "alternating", 1, {}, 0., trajectory_objective="strict")
    assert record["update_kind"] == "trajectory" and not record["controls"]
    assert all(g["score"]["reward"] == g["score"]["em"] for g in record["primary"])


def test_local_group_backprop_updates_manager_and_exactly_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    from test_policy import tiny_policy
    from scope.checkpoint import load_checkpoint, save_checkpoint
    torch.set_num_threads(1)
    policy = tiny_policy()
    optimizer = torch.optim.AdamW(policy.parameters("manager"), lr=.001, weight_decay=0.)

    def batch():
        candidates, controls = synthetic_group(["KEEP", "COMPRESS"], [1, 0])
        for c in candidates:
            c["sample"].version = policy.versions["manager"]
            c["sample"].old_logprobs = policy.logprobs(c["sample"]).detach().tolist()
        return local_weights(candidates, controls)[0]

    before, actor = policy.fingerprint("manager"), policy.fingerprint("actor")
    metrics = policy.grpo_update(batch(), "manager", optimizer)
    assert metrics["loss"] == pytest.approx(metrics["policy_loss"] + metrics["kl_loss"])
    assert metrics["grad_norm"] > 0 and policy.fingerprint("manager") != before
    assert policy.fingerprint("actor") == actor
    checkpoint = save_checkpoint(tmp_path, 1, policy, {"manager": optimizer}, "local-receipt")
    policy.grpo_update(batch(), "manager", optimizer)
    expected = policy.fingerprint("manager")
    load_checkpoint(checkpoint, policy, {"manager": optimizer}, "local-receipt")
    policy.grpo_update(batch(), "manager", optimizer)
    assert policy.fingerprint("manager") == expected
    assert policy.fingerprint("actor") == actor
