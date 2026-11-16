"""Mechanism checks, not downstream benchmark scores."""
from copy import deepcopy
import json

import pytest

from scope.contracts import ContractError, Sample, TeacherBudgetExhausted
from scope.hotpot import document_text, rollout, split_row
from scope.hotpot_memory import MentionAdapter, bridge_options, relation_compressions, state_options
from scope.hotpot_teacher_repair import repair_packet, repair_prefix, check_proposal, proposal_messages
from scope.hotpot_local_credit import local_weights
from scope.atlas import StateAtlas
from scope.contracts import Scope
from scope.tools import ToolExecutor
from test_hotpot_local_credit import SeededReader, synthetic_group, task


class RepairReader(SeededReader):
    def encode(self, messages):
        return list(messages[-1]["content"].encode())

    def choice_ids(self, labels):
        return [ord(x) for x in labels]


def test_training_entry_records_teacher_anchor_without_caller_record_prompts():
    from experiments.memory_curriculum.hotpot_grpo import step_data
    policy = RepairReader()
    public, gold = task()
    gold = {**gold, "answer": "deliberately unmatched scoring target"}
    options = {}
    requests = []
    def teacher(messages):
        requests.append(messages)
        return json.dumps({"choice": None, "diagnosis": "no_repair", "evidence_refs": []})
    _, record = step_data(policy, (public, gold), 1, "alternating", 0, options, 0.,
                          local_objective="paired_delta", trajectory_objective="strict", teacher=teacher)
    assert requests and record["teacher_repair"]["reason"] == "no_repair"
    assert options == {}
    actions = record["primary"][0]["trace"]["actions"]
    chosen = actions[record["teacher_repair"]["stage_index"]]
    assert chosen["student_messages"] and chosen["sample"]
    assert all(action.get("student_messages") for action in actions if action.get("sample"))


def reference(option, packet):
    required = ([option["evidence"]] if option.get("evidence") else []) + option.get("refs", [])
    return [{k: row[k] for k in ("id", "start", "end", "quote")}
            for ref in required for row in packet["sources"]
            if all(row[k] == ref[k] for k in ("id", "start", "end"))]


def failing_anchor():
    policy = RepairReader()
    public, gold = task()
    # Find a genuinely sampled failure, rather than editing a trace into one.
    for seed in range(100):
        trace, _ = rollout(policy, public, "rl_train", seed, record_prompts=True)
        if trace["actions"][1]["operation"] == "SKIP" and trace["answer"] == "unknown":
            return policy, public, gold, trace
    raise AssertionError("No sampled failing prefix")


def test_valid_teacher_repair_replays_student_then_builds_separate_supervision():
    policy, public, gold, trace = failing_anchor()
    before = deepcopy(trace)
    packet = repair_packet(public, trace, 1)
    label, option = next((k,v) for k,v in packet["menu"].items() if v["operation"] == "STORE")
    reply = {"choice":label,"diagnosis":"missing_retrieval","evidence_refs":reference(option,packet)}
    record, sample = repair_prefix(policy, public, gold, trace, 1, None,
                                  rollout_options={"record_prompts":True}, response=reply)
    assert trace == before and record["accepted"]
    assert record["effects"]["student"]["mean"] == 1
    assert record["branches"]["student"][0]["trace"]["answer"] == "unknown"
    assert record["branches"]["teacher"][0]["trace"]["answer"] == "Glasgow"
    assert sample.origin == "teacher" and sample.old_logprobs == []
    assert sample.prompt_ids == trace["actions"][1]["sample"]["prompt_ids"]
    assert all(row["trace"]["actions"][1]["sample"] is None for runs in record["branches"].values() for row in runs)
    assert len({row["seed"] for runs in record["branches"].values() for row in runs}) == 1


def test_teacher_never_sees_gold_future_question_or_future_observation():
    policy = RepairReader()
    public, gold = split_row({"id":"prefix", "question":"FUTURE_QUERY_SENTINEL", "answer":"HIDDEN_GOLD_SENTINEL",
        "supporting_facts":[["One",0]], "context":[["One",["first public fact"]],["Two",["second public fact"]]]})
    trace, _ = rollout(policy,public,"rl_train",3,record_prompts=True)
    packet = repair_packet(public,trace,0)
    text = json.dumps(proposal_messages(packet))
    assert public["question"] not in text and gold["answer"] not in text and "supporting_facts" not in text
    available = {r["id"] for r in packet["sources"]}
    later = {op["evidence"]["id"] for op in trace["actions"][1]["menu"].values() if op.get("evidence")}
    assert not available.intersection(later)


def test_bad_citation_or_diagnostic_only_response_cannot_become_training_label():
    policy, public, gold, trace = failing_anchor()
    packet = repair_packet(public,trace,1)
    label, op = next((k,v) for k,v in packet["menu"].items() if v["operation"] == "STORE")
    reply = {"choice":label,"diagnosis":"missing_retrieval","evidence_refs":reference(op,packet)}
    reply["evidence_refs"][0]["quote"] = "fabricated fact"
    record, sample = repair_prefix(policy,public,gold,trace,1,None,response=reply)
    assert sample is None and record["reason"] == "invalid_proposal"
    reply = {"choice":None,"diagnosis":"menu_gap","evidence_refs":[]}
    record, sample = repair_prefix(policy,public,gold,trace,1,None,response=reply)
    assert sample is None and record["reason"] == "menu_gap"
    bad = deepcopy(trace);bad["split"] = "validation"
    with pytest.raises(ContractError):repair_packet(public,bad,1)
    policy.versions["manager"] += 1
    with pytest.raises(ContractError,match="version"):
        repair_prefix(policy,public,gold,trace,1,None,response=reply)


def test_paired_credit_preserves_homogeneous_positive_negative_and_zero_effects():
    for reward, baseline, expected in [(1,0,1),(0,1,-1),(1,1,0),(0,0,0)]:
        candidates, controls = synthetic_group(["KEEP","KEEP"],[reward,reward],baseline=baseline)
        weighted, info = local_weights(candidates,controls,objective="paired_delta")
        assert info["group_zero_variance"] and info["task_zero_variance"]
        assert info["advantages"] == [expected,expected]
        assert len(weighted) == (2 if expected else 0)
        assert all(weight == .5 for _,weight in weighted)
        assert local_weights(candidates,controls,objective="group_relative")[0] == []


def memory_fixture():
    atlas = StateAtlas(Scope("rl_train","memory","0"));archive = {}
    for title, sentences_ in [("Alpha",["Decorative background.","Alpha works with Beta, but not before 1999.","Unrelated filler."]),
                              ("Beta",["Beta is based in Glasgow.","More background."])]:
        text, offsets = document_text(title,sentences_)
        r = atlas.append_public(text,entity=title)
        archive[r["id"]] = {**r,"title":title,"sentence_offsets":offsets}
    return atlas,archive


def test_state_update_checks_entity_and_cross_document_retrieval_reaches_reader():
    atlas, archive = memory_fixture()
    tools = ToolExecutor(adapter=MentionAdapter(archive))
    first, second = list(archive.values())
    full = {"id":first["id"],"start":len(first["title"])+1,"end":len(first["text"])}
    options = state_options(atlas,archive,tools,{first["id"]})
    assert options and options[0]["proposal"]["predicate"] == "mentions:Beta"
    checked = tools.validator.check(atlas,options[0]["proposal"])
    assert checked["support_status"] == "supported"
    broken = deepcopy(options[0]["proposal"]);broken["entity"] = "Beta"
    with pytest.raises(ContractError):tools.validator.check(atlas,broken)
    bridges = bridge_options(archive,[full],"Where is Alpha's partner based?")
    assert bridges[0]["evidence"]["id"] == second["id"]
    assert "Glasgow" in second["text"][bridges[0]["evidence"]["start"]:bridges[0]["evidence"]["end"]]
    atlas.close()


def test_compression_retains_bridge_negation_and_date_with_exact_sources():
    atlas, archive = memory_fixture();first = next(iter(archive.values()))
    ref = {"id":first["id"],"start":len(first["title"])+1,"end":len(first["text"])}
    options = relation_compressions(archive,[ref])
    assert options and options[0]["protected_refs"]
    content = " ".join(archive[r["id"]]["text"][r["start"]:r["end"]] for r in options[0]["refs"])
    assert "Beta" in content and "not before 1999" in content and "Decorative" not in content
    assert sum(r["end"]-r["start"] for r in options[0]["refs"]) < ref["end"]-ref["start"]
    atlas.close()


class EnhancedReader(RepairReader):
    def sample_choice(self, messages, stage, boundary, labels, max_input, greedy):
        payload = json.loads(messages[-1]["content"])
        menu = payload["verified_options"]
        want = ("PROPOSE_STATE_UPDATE" if stage.startswith("write") else
                "SUMMARIZE" if stage == "compress" else "FETCH_EVIDENCE")
        # Prefer a supported entity-mention update when it exists.
        label = next((k for k,v in menu.items() if str(v.get("predicate","")).startswith("mentions:")),None)
        label = label or next((k for k,v in menu.items() if v["operation"] == want),labels[0])
        return Sample("manager",stage,self.encode(messages),self.choice_ids([label]),[-1.],label,
                      self.versions["manager"],boundary,choice_token_ids=self.choice_ids(labels))


def test_enhanced_menu_state_commit_and_repair_replay_share_the_same_prefix():
    from scope.hotpot_local_credit import collect_local_group
    policy = EnhancedReader()
    public,gold = split_row({"id":"enhanced", "question":"Where is Alpha's partner based?", "answer":"Glasgow",
        "supporting_facts":[["Alpha",0],["Beta",0]],
        "context":[["Alpha",["Alpha works with Beta, but not before 1999.","Background noise."]],
                   ["Beta",["Beta is based in Glasgow.","Unimportant filler."]]]})
    options = dict(sentence_candidates=True,bridge_candidates=True,shared_interface=True,
                   enhanced_memory=True,record_prompts=True)
    anchor,_ = rollout(policy,public,"rl_train",3,**options)
    assert anchor["memory_protocol"] == "public_relation_candidates_v1"
    assert any(a["operation"] == "UPDATE_STATE" for a in anchor["actions"])
    nodes = [json.loads(row[2]) for row in anchor["atlas"]["tables"]["nodes"] if row[1] == "Claim"]
    assert any(n["predicate"].startswith("mentions:") and n["support_status"] == "supported" for n in nodes)
    mention_ids = {n["id"] for n in nodes if n.get("mention_target")}
    assert mention_ids and any(src in mention_ids and relation == "TARGETS"
                               for src,relation,dst in anchor["atlas"]["tables"]["edges"])
    assert all(len(a["menu"]) <= 8 for a in anchor["actions"])
    weighted,group = collect_local_group(policy,public,gold,anchor,1,objective="paired_delta",rollout_options=options)
    assert group["metrics"]["objective"] == "paired_delta"
    assert all(c["sample"]["boundary"] == anchor["actions"][1]["boundary"] for c in group["candidates"])


def test_verified_teacher_label_has_real_sft_gradient_and_is_rejected_by_grpo(tmp_path):
    torch = pytest.importorskip("torch")
    from test_policy import tiny_policy
    policy,public,gold,trace = failing_anchor()
    packet = repair_packet(public,trace,1)
    label,op = next((k,v) for k,v in packet["menu"].items() if v["operation"] == "STORE")
    record,sample = repair_prefix(policy,public,gold,trace,1,None,response={"choice":label,
        "diagnosis":"missing_retrieval","evidence_refs":reference(op,packet)},rollout_options={"record_prompts":True})
    assert record["accepted"]
    # Token IDs are remapped only for this tiny CPU gradient check.
    learner = tiny_policy();sample.prompt_ids = [1,2];sample.completion_ids = [3];sample.choice_token_ids = [3,4]
    optimizer = torch.optim.AdamW(learner.parameters("manager"),lr=.001,weight_decay=0.)
    before,actor = learner.fingerprint("manager"),learner.fingerprint("actor")
    with pytest.raises(ContractError,match="Teacher"):
        learner.grpo_update([(sample,1.)],"manager",optimizer)
    metrics = learner.sft_update([sample],optimizer)
    assert metrics["grad_norm"] > 0 and learner.fingerprint("manager") != before
    assert learner.fingerprint("actor") == actor
    from scope.checkpoint import save_checkpoint, load_checkpoint
    policy_optimizer = torch.optim.AdamW(learner.parameters("manager"),lr=.0003,weight_decay=0.)
    optimizers = {"manager":policy_optimizer,"teacher_manager":optimizer}
    def cycle():
        sampled = deepcopy(sample);sampled.origin = "policy";sampled.version = learner.versions["manager"]
        sampled.advantage = -1.
        sampled.old_logprobs = learner.logprobs(sampled).detach().tolist()
        learner.grpo_update([(sampled,1.)],"manager",policy_optimizer)
        learner.sft_update([sample],optimizer)
    cycle();path = save_checkpoint(tmp_path,1,learner,optimizers,"teacher-and-policy")
    cycle();expected = learner.fingerprint("manager")
    load_checkpoint(path,learner,optimizers,"teacher-and-policy")
    cycle();assert learner.fingerprint("manager") == expected
    assert learner.fingerprint("actor") == actor


def test_training_entrypoint_collects_teacher_repairs_without_putting_them_in_grpo():
    pytest.importorskip("pyarrow")
    from collections import Counter
    from experiments.memory_curriculum.hotpot_grpo import step_data
    class KeepReader(RepairReader):
        def sample_choice(self,messages,stage,boundary,labels,max_input,greedy):
            payload = json.loads(messages[-1]["content"])
            menu = payload.get("actions",payload.get("verified_options"))
            noop = "SKIP" if stage.startswith("write") else "KEEP" if stage == "compress" else "FINISH"
            label = next(k for k,v in menu.items() if v["operation"] == noop)
            return Sample("manager",stage,self.encode(messages),[ord(label)],[-1.],label,0,boundary,
                          choice_token_ids=self.choice_ids(labels))
    class Teacher:
        counts = Counter()
        limit = 1
        identity = {"provider":"fixture"}
        def __call__(self,messages):
            if self.counts["requests"] >= self.limit:
                raise TeacherBudgetExhausted("budget")
            self.counts["requests"] += 1
            packet = json.loads(messages[-1]["content"])
            label,option = next((k,v) for k,v in packet["menu"].items() if v["operation"] == "RETRIEVE")
            return json.dumps({"choice":label,"diagnosis":"missing_retrieval","evidence_refs":reference(option,packet)})
    teacher = Teacher();policy = KeepReader()
    weighted,record = step_data(policy,task(),2,"alternating",1,{"record_prompts":True},0.,
                               trajectory_objective="strict",local_objective="paired_delta",teacher=teacher)
    assert record["teacher_repair"]["accepted"]
    assert record["teacher_repair"]["supervised_sample"]["origin"] == "teacher"
    assert all(s.origin == "policy" for s,_ in weighted)
    assert record["sampling_tokens"]["teacher_verification"] > 0
    _,record = step_data(policy,task(),2,"alternating",1,{"record_prompts":True},0.,teacher=teacher)
    assert record["teacher_repair"]["reason"] == "teacher_request_budget_exhausted"


def test_format_only_gain_is_reported_but_not_used_as_memory_supervision():
    class FormatReader(RepairReader):
        def sample(self,*args,**kwargs):
            result = super().sample(*args,**kwargs)
            if result.text == "unknown":
                result.text = "Glasgow is the answer."
            return result
    policy = FormatReader();public,gold = task()
    for seed in range(100):
        trace,_ = rollout(policy,public,"rl_train",seed,record_prompts=True)
        if trace["actions"][1]["operation"] == "SKIP":break
    packet = repair_packet(public,trace,1)
    label,op = next((k,v) for k,v in packet["menu"].items() if v["operation"] == "STORE")
    record,sample = repair_prefix(policy,public,gold,trace,1,None,response={"choice":label,
        "diagnosis":"missing_retrieval","evidence_refs":reference(op,packet)},rollout_options={"record_prompts":True})
    assert record["observed_strict_gain"] and record["format_compatible_candidate"]
    assert sample is None and not record["accepted"] and record["routing"] == "review_or_actor_data"


def test_teacher_cache_remains_reusable_after_request_budget_exhaustion(tmp_path,monkeypatch):
    from scope.teacher_api import APITeacher
    config = tmp_path/"fixture.json"
    config.write_text(json.dumps({"base_url":"https://api.llm.ustc.edu.cn/v1","api_key":"test-placeholder"}),encoding="utf-8")
    teacher = APITeacher(config,tmp_path/"cache",max_requests=1)
    class Response:
        status_code = 200
        def json(self):
            return {"model":"qwen3.8-flash-next","choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}
    monkeypatch.setattr(teacher.http,"post",lambda *args,**kwargs:Response())
    try:
        messages = [{"role":"user","content":"public prefix"}]
        assert teacher(messages) == "{}" and teacher.counts["requests"] == 1
        assert teacher(messages) == "{}" and teacher.counts["cache_hits"] == 1
        with pytest.raises(TeacherBudgetExhausted):teacher([{"role":"user","content":"another prefix"}])
    finally:
        teacher.close()


def test_paired_signed_credit_really_updates_manager_and_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    from test_policy import tiny_policy
    from scope.checkpoint import save_checkpoint, load_checkpoint
    torch.set_num_threads(1)
    policy = tiny_policy();optimizer = torch.optim.AdamW(policy.parameters("manager"),lr=.001,weight_decay=0.)
    def update():
        candidates, controls = synthetic_group(["KEEP","KEEP"],[0,0],baseline=1)
        # A repeated sampled action still retains its original multiplicity.
        candidates[1]["sample"] = deepcopy(candidates[0]["sample"])
        for c in candidates:
            s = c["sample"];s.version = policy.versions["manager"]
            s.old_logprobs = policy.logprobs(s).detach().tolist()
        weighted, _ = local_weights(candidates,controls,objective="paired_delta")
        old = float(policy.logprobs(weighted[0][0]).detach()[0])
        metrics = policy.grpo_update(weighted,"manager",optimizer,beta=0.)
        new = float(policy.logprobs(weighted[0][0]).detach()[0])
        return metrics,old,new
    actor = policy.fingerprint("actor")
    metrics,old,new = update()
    assert metrics["grad_norm"] > 0 and new < old and policy.fingerprint("actor") == actor
    path = save_checkpoint(tmp_path,1,policy,{"manager":optimizer},"paired")
    update();expected = policy.fingerprint("manager")
    load_checkpoint(path,policy,{"manager":optimizer},"paired")
    update();assert policy.fingerprint("manager") == expected
