import json

import pytest

from scope.alfworld import ALFWorldActions
from scope.atlas import StateAtlas
from scope.contracts import Budget, ContractError, Sample, Scope
from scope.credit import paired_local_samples
from scope.environments import DeterministicTestEnvironment, ShopActions
from scope.experience import ExperienceBank, build_experiences
from scope.io import atomic_jsonl, read_jsonl
from scope.rollout import EpisodeRunner
from scope.state_graph import apply_observation
from scope.tools import ToolExecutor, context_view
from scope.training import paired_rollouts
from scope.warmup import build_examples


def commit_goal(atlas, tools, working, goal):
    record = atlas.append_public(goal, "requirement")
    tools.execute(atlas, working, {"operation": "PROPOSE_STATE_UPDATE", "expected_revision": atlas.revision,
        "predicate": "requirement", "value": goal, "support_status": "supported",
        "evidence_refs": [{"id": record["id"], "start": 0, "end": len(goal)}]}, "state", Budget(), goal)
    return record


def test_shop_checklist_needs_confirmation_and_keeps_required_state_under_compression():
    atlas = StateAtlas(Scope("train", "1", "shop"))
    tools = ToolExecutor(adapter=ShopActions())
    entity = "product:12345678"
    working = {"state_graph": True, "current_entity": entity}
    goal = "颜色是黑色，预算不超过400元。"
    requirement = commit_goal(atlas, tools, working, goal)
    page = atlas.append_public("商品详情 [SEP] 价格：358元 [SEP] 黑色 [SEP] Buy Now", entity=entity,
                               clickables=["黑色", "Buy Now"])
    apply_observation(tools, atlas, working, page, Budget(), goal)
    view = context_view(atlas, working, goal)
    assert [r["status"] for r in view["constraint_checklist"]] == ["confirmed", "pending", "pending"]
    attempted = atlas.append_public("click[黑色]", "action_receipt", entity)
    with pytest.raises(ContractError, match="EVIDENCE_DOES_NOT_SUPPORT"):
        tools.execute(atlas, working, {"operation":"PROPOSE_STATE_UPDATE", "expected_revision":atlas.revision,
            "entity":entity, "predicate":"selected_color", "value":"黑色", "support_status":"supported",
            "evidence_refs":[{"id":attempted["id"], "start":0, "end":len(attempted["text"])}]}, "state", Budget(), goal)
    selected = atlas.append_public("已选颜色：黑色\nBuy Now", entity=entity, clickables=["Buy Now"])
    apply_observation(tools, atlas, working, selected, Budget(), goal)
    budget = Budget()
    lookup = {"operation":"RETRIEVE", "expected_revision":atlas.revision, "query":"selection",
              "entity":entity, "predicate":"selected_color"}
    result = tools.execute(atlas, working, lookup, "state", budget, goal)
    assert [r["id"] for r in result["retrieved"]] == [selected["id"]]
    with pytest.raises(ContractError, match="ALREADY_USED"):
        tools.execute(atlas, working, lookup, "state", budget, goal)
    tools.execute(atlas, working, {"operation":"SUMMARIZE", "expected_revision":atlas.revision,
        "items":[{"text":selected["text"][:7], "evidence_ref":{"id":selected["id"], "start":0, "end":7}}]},
        "context", budget, goal)
    view = context_view(atlas, working, goal)
    assert view["summary"] and not view["selected_evidence"]
    assert view["constraint_checklist"][1]["status"] == "confirmed"
    assert requirement["id"] in view["constraint_checklist"][1]["evidence_ids"]
    working["candidate_since"] = atlas.revision + 1
    assert context_view(atlas, working, goal)["constraint_checklist"][1]["status"] == "pending"
    atlas.close()


def test_alfworld_state_slots_are_versioned_and_retrieval_respects_time():
    atlas = StateAtlas(Scope("train", "1", "alf"))
    tools = ToolExecutor(adapter=ALFWorldActions())
    working = {"state_graph": True}
    opened = atlas.append_public("You open the cabinet 1.")
    apply_observation(tools, atlas, working, opened, Budget(), "goal")
    old_revision = atlas.revision
    closed = atlas.append_public("You close the cabinet 1.")
    apply_observation(tools, atlas, working, closed, Budget(), "goal")
    current = atlas.nodes("Claim")
    assert [(c["entity"], c["predicate"], c["value"]) for c in current] == [("alf:cabinet 1", "container_open", False)]
    assert tools.retriever.search(atlas, "cabinet", "alf:cabinet 1", predicate="container_open")[0]["id"] == closed["id"]
    assert tools.retriever.search(atlas, "cabinet", "alf:cabinet 1", predicate="container_open",
                                  as_of_revision=old_revision)[0]["id"] == opened["id"]
    with pytest.raises(ContractError, match="STALE_SLOT_EVIDENCE"):
        tools.execute(atlas, working, tools.validator.public_facts(opened)[0].proposal(opened, atlas.revision),
                      "state", Budget(), "goal")
    pickup = atlas.append_public("You pick up the book 1 from the cabinet 1.")
    apply_observation(tools, atlas, working, pickup, Budget(), "goal")
    assert {c["predicate"]:c["value"] for c in atlas.nodes("Claim") if c["entity"] == "alf:book 1"} == {
        "held": True, "object_location": "agent"}
    atlas.close()


def test_stateful_warmup_and_experience_have_training_provenance(tmp_path):
    def task_records(task):
        return {"task_id":task,"evidence":[
            {"id":f"{task}-goal","kind":"requirement","text":"Take the book."},
            {"id":f"{task}-look","kind":"observation","text":"You are facing the desk 1. You see a book 1."},
            {"id":f"{task}-take","kind":"action_receipt","text":"take book 1 from desk 1"},
            {"id":f"{task}-held","kind":"observation","text":"You pick up the book 1 from the desk 1."},
            {"id":f"{task}-look2","kind":"action_receipt","text":"look"},
            {"id":f"{task}-look3","kind":"observation","text":"You are facing the desk 1."},
        ]}
    corpus = tmp_path / "public.jsonl"
    atomic_jsonl(corpus, [task_records("1"), task_records("2")])
    output = tmp_path / "examples.jsonl"
    report = build_examples(corpus, output, ["1", "2"], {"environment":{"kind":"alfworld_http"},
                                "runtime":{"state_graph":True, "compression_after":2}})
    assert report["operations"]["SUMMARIZE"] > 0
    examples = read_jsonl(output)
    assert any(any(c["predicate"] == "held" for c in json.loads(r["messages"][-1]["content"])["claims"]) for r in examples)
    experiences = tmp_path / "experiences.jsonl"
    assert build_experiences(corpus, experiences, ["1", "2"], ALFWorldActions())["experiences"] == 1
    assert ExperienceBank(experiences, ["1", "2"]).records[0]["support_tasks"] == ["1", "2"]
    with pytest.raises(ContractError, match="training-only"):
        build_experiences(corpus, tmp_path / "leak.jsonl", ["1"], ALFWorldActions())


class BranchShop(DeterministicTestEnvironment):
    def reset(self, task):
        super().reset(task)
        self.task_goal = "颜色是黑色。"
        self.observation = self.task_goal
        return self.observation

    def step_command(self, command):
        done = command in {"click[Buy Now]", "click[back to search]"}
        if done and command == "click[Buy Now]":
            self.metrics.update(success=1., r_strict=1., r_loose=1., purchased=1.)
        self.observation = "商品详情 [SEP] 黑色 [SEP] Buy Now"
        return self.observation, done, {"is_error":False,"clickables":["黑色", "Buy Now", "back to search"]}


class BranchPolicy:
    versions = {"manager":0, "actor":0}

    def __init__(self):
        self.draws = 0

    def encode(self, messages):
        return list(messages[-1]["content"].encode())

    def sample(self, messages, role, phase, boundary, max_input, max_new, greedy=False):
        if role == "manager":
            request = json.loads(messages[-1]["content"])
            op = {"operation":"FINISH", "expected_revision":request["revision"]}
            if not greedy and self.draws % 2 == 1:
                op.update(operation="RETRIEVE",query="Buy Now",top_k=1)
            if not greedy:
                self.draws += 1
            text = json.dumps(op)
        else:
            content = messages[-1]["content"]
            if "商品详情" not in content:
                text = "Action: click[12345678]"
            elif '"selected_evidence":[{' in content:
                text = "Action: click[Buy Now]"
            else:
                text = "Action: click[back to search]"
        return Sample(role, phase, self.encode(messages), [1], [0.], text, 0, boundary)


def test_paired_credit_replays_one_sample_and_excludes_rule_finish():
    runner = EpisodeRunner(BranchPolicy(), BranchShop(), {"state_graph":True,"max_turns":3,"manager_max_input":8192})
    rollout = runner.run("1", "one", greedy=True, capture_phase="state", capture_opportunity=True)
    boundary, = rollout.boundaries
    assert boundary.opportunity == "missing_evidence"
    baselines, candidates, seeds = paired_rollouts(runner, boundary, group_size=2, repeats=2, greedy=True, seed=233)
    assert len(set(seeds)) == 2
    weighted, info = paired_local_samples(boundary, candidates, baselines, .05)
    assert info["strict_deltas"] == [0., 1.]
    assert info["operations"] == ["FINISH", "RETRIEVE"]
    assert [s.role for s, _ in weighted] == ["manager", "manager"]
    assert all(not b.events[0]["sampled"] and b.events[0]["intervention"] for b in baselines)
    assert all(c["outcomes"][0].events[0]["replayed_sample"] for c in candidates)


def test_paired_training_step_updates_only_manager_lora(tmp_path):
    from test_policy import tiny_policy
    from scope.training import train

    policy = tiny_policy()
    scripted = BranchPolicy()
    scripted.encode = policy.encode = lambda messages: [1, 2, 3]

    def sample(messages, role, phase, boundary, max_input, max_new, greedy=False):
        action = scripted.sample(messages, role, phase, boundary, max_input, max_new, greedy)
        action.version = policy.versions[role]
        action.completion_ids = [5, 6] if '"RETRIEVE"' in action.text else [4, 5]
        action.old_logprobs = policy.logprobs(action).detach().tolist()
        return action

    policy.sample = sample

    class Registry:
        def training_sequence(self, length):
            return ["1"]

    cfg = {"runtime":{"state_graph":True,"max_turns":3},
           "training":{"method":"conditional","local_credit":"paired","updates":1,
                       "group_size":2,"local_group_size":2,"probe_updates":[1]}}
    before_actor, before_manager = policy.fingerprint("actor"), policy.fingerprint("manager")
    train(policy, BranchShop(), Registry(), cfg, tmp_path, {"test":"paired"})
    row, = read_jsonl(tmp_path / "train_metrics.jsonl")
    assert row["local"]["global_manager_credit_applied"] is False
    assert row["trained_actions"] == 2 and row["task_policy_gradient"] and row["grad_norm"] > 0
    assert policy.fingerprint("manager") != before_manager and policy.fingerprint("actor") == before_actor


def test_compression_cost_needs_success_and_required_evidence():
    from types import SimpleNamespace

    boundary = SimpleNamespace(phase="context", opportunity="context_threshold", role_versions={"manager":0},
                               fingerprint=lambda: "boundary")
    def outcome(success, tokens, evidence):
        return SimpleNamespace(metrics={"success":success}, costs={"charged_tokens":tokens},
            events=[{"phase":"execute","required_evidence_ids":["required"],"context_evidence_ids":evidence}])
    baseline = outcome(1, 100, ["required"])
    candidates = [{"sample":Sample("manager","context",[1],[2],[0.],op,0,"boundary"),
        "accepted":True,"operation":op,"boundary_hash":"boundary","outcomes":[value]} for op,value in (
            ("FINISH", baseline), ("SUMMARIZE", outcome(1, 70, ["required"])))]
    weighted, report = paired_local_samples(boundary, candidates, [baseline], .1)
    assert report["task_zero_variance"] and weighted[0][0].advantage == pytest.approx(.03)
    candidates[1]["outcomes"] = [outcome(1, 70, [])]
    assert not paired_local_samples(boundary, candidates, [baseline], .1)[0]
    baseline.metrics["success"] = 0
    candidates[1]["outcomes"] = [outcome(0, 10, ["required"])]
    assert not paired_local_samples(boundary, candidates, [baseline], .1)[0]
