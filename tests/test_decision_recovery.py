import copy

import pytest

from scope.alfworld import ALFWorldActions
from scope.atlas import StateAtlas
from scope.candidate_actions import candidate_actions
from scope.contracts import Budget, ContractError, Scope
from scope.decision_evidence import review
from scope.interaction import execute_action
from scope.io import digest
from scope.recovery import prepare_recovery, record_recovery
from scope.retrieval import Retriever
from scope.tools import ToolExecutor, context_view


@pytest.fixture
def scene():
    atlas = StateAtlas(Scope("internal_dev", "recovery", "1"))
    tools = ToolExecutor(adapter=ALFWorldActions())
    working = {"state_graph": True, "shared_memory": True, "decision_evidence": True,
               "decision_recovery": True, "current_entity": "task", "public_state": {"held": []},
               "recovery_slots": [("alf:apple 1", "held")]}
    yield atlas, tools, working, Budget()
    atlas.close()


def test_raw_recall_without_claim_then_validated_commit(scene):
    atlas, tools, working, budget = scene
    old = atlas.append_public("You pick up the apple 1 from the desk 1.", "observation")
    working["observation_id"] = atlas.append_public("You arrive at cabinet 1.", "observation")["id"]
    original = atlas.fingerprint()
    menu = candidate_actions(atlas, working, "put the apple", "state", budget, tools)
    raw = next(p for p in menu.values() if p.get("source") == "raw")
    result = tools.execute(atlas, working, raw, "state", budget, "put the apple")
    assert atlas.fingerprint() == original  # Read-only lookup never confirms a fact.
    assert result["gap_diagnostic"]["class"] == "extractor_gap"
    assert review(atlas, working, "put the apple")[0]["status"] == "pending"
    assert context_view(atlas, working, "put the apple")["selected_evidence"][0]["id"] == old["id"]
    menu = candidate_actions(atlas, working, "put the apple", "state", budget, tools)
    commit = next(p for p in menu.values() if p.get("predicate") == "held" and p.get("operation") == "PROPOSE_STATE_UPDATE")
    tools.execute(atlas, working, commit, "state", budget, "put the apple")
    assert review(atlas, working, "put the apple")[0]["status"] == "supported"


def test_raw_seen_object_does_not_confirm_holding_and_expires(scene):
    atlas, tools, working, budget = scene
    atlas.append_public("You see an apple 1 on desk 1.", "observation")
    working["observation_id"] = atlas.append_public("You arrive at cabinet 1.", "observation")["id"]
    menu = candidate_actions(atlas, working, "goal", "state", budget, tools)
    raw = next(p for p in menu.values() if p.get("source") == "raw")
    result = tools.execute(atlas, working, raw, "state", budget, "goal")
    assert result["gap_diagnostic"]["class"] == "unverified_raw_evidence"
    assert not atlas.nodes("Claim")
    working["state_lookup_used"] = False
    raw["expected_revision"] = atlas.revision
    with pytest.raises(ContractError, match="NO_NEW_DECISION_EVIDENCE"):
        tools.execute(atlas, working, raw, "state", budget, "goal")
    working["public_state"] = {"held": ["apple 1"]}
    assert not context_view(atlas, working, "goal")["selected_evidence"]


def test_raw_boundaries_revision_and_public_only(scene):
    atlas, _, _, _ = scene
    old = atlas.append_public("You take the apple 1 from desk 1.", "observation")
    atlas.append_public("You take the apple 10 from desk 1.", "observation")
    atlas.append_public("take apple 1 from desk 1", "action_receipt")
    atlas.append_public("hold apple 1", "requirement")
    atlas.append_public("Room.\nYour task is to find apple 1", "observation")
    retriever = Retriever()
    found = retriever.search_raw(atlas, "apple 1 held", "alf:apple 1")
    assert [r["id"] for r in found] == [old["id"]]
    assert not retriever.search_raw(atlas, "apple 1 held", "alf:apple 1", min_revision=old["revision"] + 1)


def test_post_action_pending_does_not_reuse_pre_action_confirmation(scene):
    atlas, tools, working, budget = scene
    atlas.append_public("You take the apple 1 from desk 1.", "observation")
    attempt = atlas.append_public("take apple 1 from desk 1", "action_receipt")
    working["confirmation"] = {"entity": "alf:apple 1", "predicate": "held", "status": "pending",
                               "attempt_revision": attempt["revision"]}
    working["observation_id"] = atlas.append_public("Nothing happens.", "observation")["id"]
    menu = candidate_actions(atlas, working, "goal", "state", budget, tools)
    assert not any(p.get("source") == "raw" or p.get("predicate") == "held" for p in menu.values())


def test_recovery_candidates_public_and_guard_resets_on_state_change(scene):
    atlas, _, working, _ = scene
    adapter = ALFWorldActions()
    obs = "You see a desk 1, a drawer 1.\nYour task is to examine the pencil with the desklamp."
    first = atlas.append_public(obs, "observation")
    working["public_state"] = adapter.observe(obs, {}, None, [])
    for _ in range(2):
        record_recovery(working, {"normalized_command": "look", "action_status": "accepted",
            "progress": "legal_no_progress", "public_state_before": digest(working["public_state"]),
            "observation_before_id": first["id"]})
    plan = prepare_recovery(adapter, atlas, working, "examine the pencil with the desklamp")
    assert plan["blocked_commands"] == ["look"]
    assert {c["command"] for c in plan["candidates"]} == {"go to desk 1", "go to drawer 1"}
    assert all(c["evidence_ids"] == [first["id"]] for c in plan["candidates"])

    class Env:
        def step_command(self, command):
            pytest.fail("A blocked command must not reach the environment")
    _, done, _, action = execute_action(Env(), adapter, "Action: look", obs, working["public_state"],
                                       "grounded", plan["blocked_commands"])
    assert not done and not action["sent"] and action["action_status"] == "no_progress_blocked"
    working["public_state"] = adapter.observe("You arrive at desk 1.", working["public_state"], "go to desk 1", [])
    assert not prepare_recovery(adapter, atlas, working, "goal")["blocked_commands"]


def test_receptacle_normalization_preserves_ambiguity():
    adapter = ALFWorldActions()
    state = {"entities": ["fridge 1", "fridge 2"]}
    assert adapter.normalize("open receptacle fridge 1.") == "open fridge 1"
    assert adapter.bind(adapter.normalize("open receptacle fridge 1."), state).status == "bound"
    assert adapter.bind(adapter.normalize("open fridge"), state).status == "ambiguous"


def test_raw_ablation_and_budget(scene):
    atlas, tools, working, budget = scene
    atlas.append_public("You see apple 1 on desk 1.", "observation")
    working["observation_id"] = atlas.append_public("At cabinet 1.", "observation")["id"]
    working["disabled_memory_tools"] = ["RETRIEVE"]
    assert not any(p["operation"] == "RETRIEVE" for p in candidate_actions(atlas, working, "goal", "state", budget, tools).values())
    working["disabled_memory_tools"] = []
    budget.retrievals = 0
    assert not any(p["operation"] == "RETRIEVE" for p in candidate_actions(atlas, working, "goal", "state", budget, tools).values())


def test_recovery_rollout_exposes_candidates_and_logs_blocked_dispatch():
    from scope.contracts import Sample
    from scope.rollout import EpisodeRunner
    from scope.recovery import recovery_diagnostics
    from scope.io import canonical

    class Env:
        action_adapter = ALFWorldActions()
        actor_system = "Select a public command."
        task_goal = "inspect desk 1"
        def reset(self, task):
            self.metrics = dict(success=0, r_loose=0, r_strict=0, purchased=0)
            self.commands = []
            return "You see a desk 1."
        def step_command(self, command):
            self.commands.append(command)
            done = command == "go to desk 1"
            if done:
                self.metrics.update(success=1, r_loose=1, r_strict=1)
            return "You arrive at desk 1." if done else "You see a desk 1.", done, {"is_error":False}
        def close(self):
            pass

    class Policy:
        versions = {"actor":0, "manager":0}
        calls = 0
        def encode(self, messages):
            return [1] * (len(canonical(messages))//10 + 1)
        def sample(self, messages, role, phase, boundary, max_input, max_new, greedy):
            self.calls += 1
            assert '"go to desk 1"' in messages[-1]["content"]
            text = "Action: look" if self.calls < 4 else "Action: go to desk 1"
            return Sample(role, phase, self.encode(messages), [2], [0.], text, 0, boundary)

    env = Env()
    cfg = dict(max_turns=4, state_graph=True, shared_memory=True, decision_evidence=True,
               decision_recovery=True, memory_choice=True, manager_mode="rule", action_mode="grounded", loop_policy="state")
    outcome = EpisodeRunner(Policy(), env, cfg, split="internal_dev").run("fixture", "1", greedy=True)
    assert outcome.metrics["success"] == 1
    assert env.commands == ["look", "look", "go to desk 1"]
    stats = recovery_diagnostics(outcome.events)
    assert stats["repeated_actions_blocked"] == 1 and stats["recovery_candidate_selected"] == 1


def test_shared_teacher_can_choose_raw_lookup_without_claim(scene):
    from scope.teacher import TeacherDistillation
    from scope.rollout import manager_messages
    import json
    atlas, tools, working, budget = scene
    atlas.append_public("You see apple 1 on desk 1.", "observation")
    working["observation_id"] = atlas.append_public("At cabinet 1.", "observation")["id"]
    menu = candidate_actions(atlas, working, "find apple", "state", budget, tools)
    rule = next(p for p in menu.values() if p.get("source") == "raw")
    def teacher(request):
        payload = json.loads(request[-1]["content"])
        chosen = next(p for p in payload["verified_options"] if p["proposal"].get("source") == "raw")
        return json.dumps({"case":"evidence_gap", "candidate_id":chosen["candidate_id"]})
    distill = TeacherDistillation(teacher, 1)
    result = distill.annotate(atlas, working, "find apple", "state", budget,
                            manager_messages(atlas, "At cabinet 1.", working, "find apple", "state", budget), tools, rule)
    assert result and result["proposal"]["source"] == "raw"
    assert distill.counts["accepted"] == 1 and not atlas.nodes("Claim")
