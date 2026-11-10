import copy
import json
from types import SimpleNamespace

import pytest

from scope.alfworld import ALFWorldActions
from scope.atlas import StateAtlas
from scope.contracts import Budget, ContractError, InfrastructureError, Scope
from scope.interaction import record_progress
from scope.io import canonical, digest
from scope.recovery import information_progress, prepare_recovery
from scope.rollout import EpisodeRunner, manager_messages
from scope.state_graph import apply_observation, alfworld_facts
from scope.teacher import TeacherDistillation
from scope.teacher_execution import ExecutionTeacher
from scope.tools import ToolExecutor


def test_room_listing_binds_location_without_asserting_possession():
    text = "You arrive at desk 1. On the desk 1, you see a alarmclock 2, a book 1, and a desklamp 1."
    adapter = ALFWorldActions()
    state = adapter.observe(text, {}, "go to desk 1", [])
    assert state["object_locations"]["alarmclock 2"] == "desk 1"
    assert not state.get("held")
    assert "take alarmclock 2 from desk 1" in {x["command"] for x in adapter.recovery_candidates(state, "look at alarmclock")}
    atlas = StateAtlas(Scope("train", "room", "1"))
    try:
        obs = atlas.append_public(text)
        tools = ToolExecutor(adapter=adapter)
        apply_observation(tools, atlas, {"state_graph":True,"public_state":state}, obs, Budget(), "look at alarmclock")
        assert any(c["entity"] == "alf:alarmclock 2" and c["predicate"] == "object_location" for c in atlas.nodes("Claim"))
        assert not any(c["predicate"] == "held" and c["value"] is True for c in atlas.nodes("Claim"))
        assert not list(alfworld_facts({"text":"Room.\nYour task is to: on desk 1, you see a secretobject 9."}))
    finally:
        atlas.close()


def test_processing_candidates_require_public_possession_and_location():
    adapter = ALFWorldActions()
    state = adapter.observe("You arrive at microwave 1.", {}, None, [])
    assert not any(x["command"].startswith("heat") for x in adapter.recovery_candidates(state, "heat plate"))
    state = adapter.observe("You pick up the plate 1 from the microwave 1.", state, None, [])
    assert "heat plate 1 with microwave 1" in {x["command"] for x in adapter.recovery_candidates(state, "heat plate")}
    assert adapter.normalize("check inventory.") == "inventory"
    assert adapter.bind("look at plate 1", state).status == "invalid_syntax"


def test_twenty_rejected_actions_are_not_zero_loops():
    window = []
    loops = []
    for _ in range(20):
        progress, cycle = record_progress(window, "room", "room", "look at alarmclock 2", "obs", False, False)
        assert progress == "invalid_no_progress"
        loops.append(cycle)
    assert sum(bool(x) for x in loops) == 19
    assert loops[-1]["kind"] == "repeated_rejection"
    assert record_progress(window, "room", "holding", "take alarmclock 2 from desk 1", "obs2", True, False) == ("state_changed", None)


def test_revisited_page_is_not_new_information_or_verified_task_progress():
    working = {}
    first = {"sent":True,"action_status":"accepted","progress":"state_changed","loop":None}
    information_progress(working, first, "detail", "description")
    assert first["information_gain"] == "new_observation"
    second = copy.deepcopy(first)
    information_progress(working, second, "description", "detail")
    assert second["state_change_without_new_observation"]
    assert second["task_progress_verified"] is None


def test_teacher_history_discloses_missing_coverage():
    atlas = StateAtlas(Scope("train", "t", "1"))
    try:
        for i in range(15):
            atlas.append_public(f"Observation {i}: apple on desk.")
        records, coverage = TeacherDistillation.history(atlas, {}, "apple", 10000)
        assert len(records) == 15 and coverage["complete"]
        _, limited = TeacherDistillation.history(atlas, {}, "apple", 20)
        assert not limited["complete"]
    finally:
        atlas.close()


def test_teacher_can_add_a_valid_operation_outside_menu(monkeypatch):
    atlas = StateAtlas(Scope("train", "t", "1"))
    tools = ToolExecutor(adapter=ALFWorldActions())
    try:
        obs = atlas.append_public("You pick up the apple 1 from the desk 1.")
        working = {"state_graph":True,"current_entity":"task","interaction_feedback":{"message":"missing held state"}}
        messages = manager_messages(atlas, obs["text"], working, "put apple", "state", Budget())
        proposal = next(f.proposal(obs, atlas.revision) for f in alfworld_facts(obs) if f.predicate == "held")
        teacher = TeacherDistillation(lambda request:canonical({"case":"wrong_or_stale_link", "proposal":proposal,
            "target_ids":[],"diagnosis":"menu_gap"}), 1)
        monkeypatch.setattr(teacher, "options", lambda *args:[])
        before = atlas.fingerprint()
        result = teacher.annotate(atlas, working, "put apple", "state", Budget(), messages, tools,
                                  {"operation":"FINISH","expected_revision":atlas.revision})
        assert result["menu_addition"] and result["proposal"]["predicate"] == "held"
        assert result["diagnosis_verified"] is False and atlas.fingerprint() == before
    finally:
        atlas.close()


def test_execution_teacher_rejects_hidden_citation_and_bad_binding():
    messages = [{"role":"user","content":"obs_000001_abc: You see desk 1."}]
    state = ALFWorldActions().observe("You see desk 1.", {}, None, [])
    def teacher(command, ids):
        return ExecutionTeacher(lambda _:canonical({"command":command,"evidence_ids":ids,"subgoal":"inspect desk"}))
    result = teacher("go to desk 1", ["obs_000001_abc"]).propose(messages,state,ALFWorldActions())
    assert result["grammar_verified"] and not result["utility_verified"]
    with pytest.raises(ContractError, match="visible"):
        teacher("go to desk 1", ["obs_000002_abc"]).propose(messages,state,ALFWorldActions())
    with pytest.raises(ContractError, match="bind"):
        teacher("go to desk 9", ["obs_000001_abc"]).propose(messages,state,ALFWorldActions())
    guide=teacher("go to desk 1", ["obs_000001_abc"])
    assert not guide.should_intervene({})
    assert guide.should_intervene({"warning":"repeated invalid action"})


def test_teacher_rollout_provenance_and_split_guard():
    class Env:
        action_adapter = ALFWorldActions()
        task_goal = "inspect desk"
        actor_system = "Action: go to desk 1"
        def reset(self, task):
            self.metrics = dict(success=0,r_loose=0,r_strict=0,purchased=0)
            return "You see desk 1."
        def step_command(self, command):
            assert command == "go to desk 1"
            self.metrics.update(success=1,r_loose=1,r_strict=1)
            return "You arrive at desk 1.", True, {"is_error":False}
        def close(self): pass
    class Policy:
        versions = {"actor":0,"manager":0}
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs:list(text.encode()))
        def encode(self, messages): return [1] * (len(canonical(messages))//10 + 1)
        def sample(self, *args): pytest.fail("Teacher collection must not label a student sample as teacher")
    class Teacher:
        def propose(self, messages, state, adapter):
            return {"text":"Action: go to desk 1","subgoal":"inspect desk","utility_verified":False}
    cfg = {"manager_mode":"rule","action_mode":"grounded","max_turns":2,"actor_max_new":256}
    with pytest.raises(ContractError, match="training partition"):
        EpisodeRunner(Policy(),Env(),cfg,"dev",actor_teacher=Teacher())
    result=EpisodeRunner(Policy(),Env(),cfg,"rl_train",actor_teacher=Teacher()).run("1","teacher")
    assert result.metrics["success"] == 1
    assert result.samples[0].origin == "teacher" and not result.samples[0].old_logprobs
    assert result.events[-1]["teacher_demonstration"]["utility_verified"] is False


def test_api_cache_and_secret_redaction(tmp_path, monkeypatch):
    from scope.teacher_api import APITeacher
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"base_url":"https://api.llm.ustc.edu.cn/v1","api_key":"test-secret"}))
    teacher = APITeacher(config,tmp_path/"cache",1)
    calls=[]
    def post(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200,json=lambda:{"model":"qwen3.8-flash-next",
            "choices":[{"finish_reason":"stop","message":{"content":"{\"case\":\"sufficient_state\"}"}}],
            "usage":{"prompt_tokens":12,"completion_tokens":4}})
    monkeypatch.setattr(teacher.http,"post",post)
    messages=[{"role":"user","content":"public prefix"}]
    assert teacher(messages)==teacher(messages)
    assert len(calls)==1 and teacher.counts["cache_hits"]==1 and calls[0]["allow_redirects"] is False
    assert "test-secret" not in next((tmp_path/"cache").glob("*.json")).read_text()
    with pytest.raises(InfrastructureError,match="budget"):
        teacher([{"role":"user","content":"different prefix"}])
    teacher.close()


def test_teacher_samples_cannot_receive_grpo_updates():
    from test_policy import tiny_policy
    from scope.contracts import Sample
    import torch
    policy = tiny_policy()
    sample=Sample("manager","state",[1,2],[3],[0.],"teacher",policy.versions["manager"],"prefix",origin="teacher")
    with pytest.raises(ContractError,match="on-policy"):
        policy.grpo_update([(sample,1.)],"manager",torch.optim.AdamW(policy.parameters("manager")))


def test_online_teacher_falls_back_on_invalid_completion_and_resumes_from_cache(tmp_path, monkeypatch):
    from scope.teacher_online import OnlineRecovery

    config = tmp_path / "provider.json"
    config.write_text(json.dumps({"base_url": "https://api.llm.ustc.edu.cn/v1",
                                  "api_key": "test-secret", "model": "different-default"}))
    teacher = OnlineRecovery(config, tmp_path / "cache", max_requests=4)
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        return SimpleNamespace(status_code=200, json=lambda: {
            "model": teacher.identity["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": "invalid JSON"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })

    monkeypatch.setattr(teacher.api.http, "post", post)
    messages = [{"role": "user", "content": "obs_000001_abc: You see desk 1."}]
    adapter = ALFWorldActions()
    state = adapter.observe("You see desk 1.", {}, None, [])
    warning = {"warning": "previous command was rejected"}
    assert not teacher.should_intervene({})
    for _ in range(2):
        assert teacher.should_intervene(warning)
        assert teacher.propose(messages, state, adapter) is None
    assert not teacher.should_intervene(warning)
    assert calls == ["qwen3.8-flash-next"]
    usage = teacher.task_usage()
    assert usage["rejected_proposals"] == 2 and usage["cache_hits"] == 1
    assert usage["logical_prompt_tokens"] == 20 and usage["prompt_tokens"] == 10
    teacher.begin_task()
    assert teacher.should_intervene(warning)
    monkeypatch.setattr(teacher.api.http, "post", lambda *a, **kw: SimpleNamespace(
        status_code=200, json=lambda: {"model": "wrong-model", "choices": []}))
    with pytest.raises(InfrastructureError, match="response envelope"):
        teacher.propose([*messages, {"role": "user", "content": "new prompt"}], state, adapter)
    teacher.close()


def test_actor_warmup_updates_only_actor_and_resumes(tmp_path):
    import torch
    from test_policy import tiny_policy
    from scope.warmup import warmup
    from scope.io import atomic_jsonl
    policy = tiny_policy()
    policy.encode = lambda messages:[1,2,3]
    policy.tokenizer = SimpleNamespace(encode=lambda text,**kw:[4,5],eos_token_id=6)
    optimizers={r:torch.optim.AdamW(policy.parameters(r),lr=.001) for r in ("actor","manager")}
    examples=tmp_path/"examples.jsonl"
    row={"task_id":"train1","phase":"execute","source":"teacher_actor_public_prefix",
         "messages":[{"role":"user","content":"public state"}],"completion":"Action: look",
         "environment_accepted":True,"task_success":1}
    atomic_jsonl(examples,[row])
    manager=policy.fingerprint("manager")
    initial=policy.fingerprint("actor")
    report=warmup(policy,examples,optimizers,tmp_path/"run","receipt",epochs=2,
                  allowed_tasks=["train1"],stop_after=1,role="actor")
    assert report["actor_changed"] and report["manager_unchanged"] and not report["complete"]
    report=warmup(policy,examples,optimizers,tmp_path/"run","receipt",epochs=2,
                  allowed_tasks=["train1"],resume=report["checkpoint"],role="actor")
    assert report["complete"] and report["updates"]==2
    assert policy.fingerprint("manager")==manager and policy.fingerprint("actor")!=initial
    row["task_success"]=0
    atomic_jsonl(examples,[row])
    with pytest.raises(ContractError,match="successful"):
        warmup(policy,examples,optimizers,tmp_path/"bad","receipt",allowed_tasks=["train1"],role="actor")


def test_teacher_probe_requires_matching_public_state_and_training_split(tmp_path):
    from scope.probe_selection import TeacherProbeSelector, prefix_signature
    from scope.io import atomic_jsonl, file_sha
    atlas=StateAtlas(Scope("train","train1","episode1"))
    try:
        working={"public_state":{"held":[]}}
        key=prefix_signature(atlas,working,"put apple","desk","state")
        path=tmp_path/"hints.jsonl"
        atomic_jsonl(path,[{"source":"teacher_public_probe","task_id":"train1","phase":"state","prefix_hash":key}])
        spec={"path":str(path),"sha256":file_sha(path)}
        selector=TeacherProbeSelector(spec,["train1"])
        assert selector(atlas,working,"put apple","desk","state")
        assert not selector(atlas,{"public_state":{"held":["apple 1"]}},"put apple","desk","state")
        with pytest.raises(ContractError,match="partition"):
            TeacherProbeSelector(spec,["different-task"])
        with pytest.raises(ContractError,match="checksum"):
            TeacherProbeSelector({**spec,"sha256":"invalid"},["train1"])
    finally:
        atlas.close()


def test_compact_view_keeps_goal_claims_and_provenance():
    from scope.environments import ShopActions
    atlas=StateAtlas(Scope("train","shop","1"))
    try:
        goal="A black item under 100 yuan"
        req=atlas.append_public(goal,"requirement")
        tools=ToolExecutor(adapter=ShopActions())
        tools.execute(atlas,{}, {"operation":"PROPOSE_STATE_UPDATE","expected_revision":atlas.revision,
            "entity":"task","predicate":"requirement","value":goal,"support_status":"supported",
            "evidence_refs":[{"id":req["id"],"start":0,"end":len(goal)}]},"state",Budget(),goal)
        obs=atlas.append_public("Public page")
        working={"state_graph":True,"decision_recovery":True,"current_entity":"task","observation_id":obs["id"],
                 "recovery_plan":{"goal":goal,"known_entities":["x"]*100,"candidates":[],"held":[],"unresolved_slots":[]}}
        env=SimpleNamespace(action_adapter=ShopActions())
        policy=SimpleNamespace(encode=lambda messages:list(canonical(messages).encode()))
        history=[{"response":"Action: search[x]","observation":"Past page "*20} for _ in range(6)]
        full=EpisodeRunner(policy,env,{"actor_max_input":16000})._actor_messages(atlas,working,goal,"Public page",history)[0]
        compact=EpisodeRunner(policy,env,{"actor_max_input":16000,"compact_actor_view":True,"actor_history_turns":2})._actor_messages(
            atlas,working,goal,"Public page",history)[0]
        assert len(canonical(compact)) < len(canonical(full))
        assert goal in canonical(compact) and req["id"] in canonical(compact)
        assert "known_entities" not in canonical(compact)
    finally:
        atlas.close()


def test_freeform_warmup_retains_teacher_action_missing_from_menu(monkeypatch):
    from scope.warmup import _decision_example
    import scope.warmup as warmup_module
    atlas=StateAtlas(Scope("train","train1","e1"))
    try:
        obs=atlas.append_public("You pick up the apple 1 from the desk 1.")
        tools=ToolExecutor(adapter=ALFWorldActions())
        working={"state_graph":True,"decision_recovery":True,"freeform_manager":True,"current_entity":"task",
                 "observation_id":obs["id"],"public_state":{},"interaction_feedback":{"message":"missing state"}}
        finish={"operation":"FINISH","expected_revision":atlas.revision}
        monkeypatch.setattr(warmup_module,"candidate_actions",lambda *args:{"A":finish})
        proposal=next(f.proposal(obs,atlas.revision) for f in alfworld_facts(obs) if f.predicate=="held")
        teacher=TeacherDistillation(lambda _:canonical({"case":"wrong_or_stale_link","proposal":proposal,
                                                       "target_ids":[],"probe_worthy":True}),1)
        label=_decision_example(atlas,working,"put apple","state",Budget(),tools,obs["text"],None,teacher)
        assert label["source"]=="teacher_state_graph_public_prefix"
        assert json.loads(label["completion"])["predicate"]=="held"
        assert label["teacher_probe"]["worthy"] and "verified_options" not in json.loads(label["messages"][-1]["content"])
        assert not atlas.nodes("Claim")
    finally:
        atlas.close()
