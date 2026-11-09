from experiments.memory_curriculum.choice_study import fixture


def test_choice_menus_bind_current_entity_and_keep_uncertainty():
    for case in ("wrong_entity", "stale_state", "pending", "preserve_constraints", "sufficient_state"):
        for variant in (0, 3):
            row, menu, gold = fixture("public-test", "internal_dev", case, variant)
            assert len(menu) >= 2 and gold
            if row["phase"] == "state":
                assert all(p.get("entity", row["working"]["current_entity"]) == row["working"]["current_entity"]
                           for p in menu.values())
            if case == "pending":
                assert all(p.get("support_status") != "supported" for p in menu.values()
                           if p.get("predicate") == "selected_color")


def test_choice_sampling_and_replay_use_the_same_restricted_distribution():
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from test_policy import tiny_policy

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(text) - 60]

    policy = tiny_policy()
    policy.tokenizer = Tokenizer()
    policy.encode = lambda messages: [1, 2, 3]
    labels = ["A", "B", "C"]
    sample = policy.sample_choice([], "state", "same", labels, 16)
    assert sample.text in labels and len(sample.completion_ids) == 1
    with torch.no_grad():
        assert policy.logprobs(sample).item() == pytest.approx(sample.old_logprobs[0], abs=1e-7)
        assert policy.choice_logprobs(sample, "manager").exp().sum().item() == pytest.approx(1.)
    actor = policy.fingerprint("actor")
    manager = policy.fingerprint("manager")
    optimizer = torch.optim.AdamW(policy.parameters("manager"),lr=.001)
    sample.advantage = 1.
    assert policy.grpo_update([(sample,1.)],"manager",optimizer)["grad_norm"] > 0
    assert policy.fingerprint("manager") != manager and policy.fingerprint("actor") == actor
