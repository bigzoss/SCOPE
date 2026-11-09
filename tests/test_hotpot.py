import json
import re
from types import SimpleNamespace

from scope.contracts import Sample
from scope.hotpot import rollout, score, split_row, weights
from scope.hotpot_intervention import aligned_credit, compression_efficiency, coverage, factorial, public_support_inventory, subtract


class FixedPolicy:
    torch = SimpleNamespace(manual_seed=lambda seed:None)
    versions = {"manager":0,"actor":0}

    def sample_choice(self, messages, stage, boundary, labels, max_input, greedy):
        payload = json.loads(messages[-1]["content"])
        op = "STORE" if stage.startswith("write") else "KEEP" if stage=="compress" else "FINISH"
        menu = payload.get("actions",payload.get("verified_options"))
        label = next(k for k,v in menu.items() if v["operation"]==op)
        return Sample("manager",stage,[1],[2],[0.],label,0,boundary,choice_token_ids=[2,3])

    def sample(self, messages, role, phase, boundary, max_input, max_new, greedy):
        payload = json.loads(messages[-1]["content"])
        answer = "Glasgow" if any("Glasgow" in row["text"] for row in payload["evidence"]) else "unknown"
        return Sample("actor",phase,[1],[2],[0.],answer,0,boundary)


def test_hotpot_replay_keeps_prefix_and_never_supplies_grader_fields():
    row = {"id":"test","question":"Which city?","answer":"Glasgow",
           "supporting_facts":[["Primary",0]],"context":[["Primary",["The city is Glasgow."]]]}
    public,gold = split_row(row)
    assert set(public)=={"id","question","paragraphs"}
    policy = FixedPolicy()
    fact,samples = rollout(policy,public,"rl_train",1)
    control,_ = rollout(policy,public,"rl_train",1,factual=fact,intervention=1)
    assert fact["actions"][1]["boundary"]==control["actions"][1]["boundary"]
    assert control["actions"][1]["operation"]=="SKIP"
    assert score(fact,gold)["em"]==1 and score(control,gold)["em"]==0
    group = {"samples":samples,"score":score(fact,gold),"control_score":score(control,gold)}
    weighted,global_adv,delta = weights([group],1,.75)
    assert global_adv==[0.] and delta==[1.]
    assert len(weighted)==1 and weighted[0][0].phase=="write_2" and weighted[0][0].advantage==.75


def test_teacher_branch_can_export_a_verified_student_preference():
    public = {"id":"demo","question":"Which city?",
              "paragraphs":[["Page A",["The city is Glasgow."]],["Page B",["Unrelated sentence."]]]}

    def teacher(messages,menu,public_options,working_sources):
        prompt = json.loads(messages[-1]["content"])
        assert ("question" in prompt)==prompt["stage"].startswith("retrieve")
        label = next((k for k,v in menu.items() if v["operation"]=="STORE"),next(iter(menu)))
        return {"choice":label,"alternative":next(k for k in menu if k!=label),
                "evidence_id":public_options.get(label,{}).get("source_id")}

    policy = FixedPolicy()
    expert,_ = rollout(policy,public,"rl_train",1,chooser=teacher,sentence_candidates=True,record_prompts=True)
    action = expert["actions"][0]
    forced = action["teacher_decision"]["alternative"]
    branch,_ = rollout(policy,public,"rl_train",1,factual=expert,intervention=0,intervention_label=forced,
                       sentence_candidates=True)
    assert branch["actions"][0]["choice"]==forced and branch["actions"][0]["sample"] is None
    assert branch["actions"][0]["boundary"]==action["boundary"]
    assert forced in json.loads(action["student_messages"][-1]["content"])["verified_options"]


def test_factorial_masks_only_source_interval_and_assigns_credit_to_sample():
    public,gold = split_row({"id":"test","question":"Which city?","answer":"Glasgow",
                             "supporting_facts":[["Primary",0]],"context":[["Primary",["The city is Glasgow."]]]})
    policy = FixedPolicy()
    options = {"sentence_candidates":True,"bridge_candidates":True}
    factual,samples = rollout(policy,public,"rl_train",1,**options)
    control,control_samples = rollout(policy,public,"rl_train",1,factual=factual,intervention=1,**options)
    contrast = factorial(policy,public,gold,factual,control,1)
    assert contrast["eligible"] and contrast["interaction"]==1
    assert contrast["cells"]["a_masked"]["answer_changed"]
    assert contrast["cells"]["b_masked"]["reused_visible"]
    assert coverage(factual,public,gold)["delivered"]==1
    assert coverage(control,public,gold)["offered_not_delivered"]==1
    assert control_samples[1] is None
    weighted,_,_ = weights([{"samples":samples,"score":score(factual,gold),"control_score":score(control,gold),
                             "evidence_credit":contrast["evidence_credit"]}],1,.75)
    assert len(weighted)==1 and weighted[0][0].credit_scope=="hotpot:evidence"


def test_factorial_uses_stable_sources_when_later_graph_ids_shift():
    public = {"id":"prefix","question":"Which city?","paragraphs":[["A",["Glasgow"]],["B",["Dundee"]]]}
    gold = {"answer":"Glasgow","supporting_facts":[["A",0]]}
    policy = FixedPolicy()
    factual,_ = rollout(policy,public,"rl_train",1)
    control,_ = rollout(policy,public,"rl_train",1,factual=factual,intervention=0)
    assert factual["actions"][1]["boundary"]!=control["actions"][1]["boundary"]
    contrast = factorial(policy,public,gold,factual,control,0)
    assert contrast["eligible"] and contrast["cells"]["a_masked"]["removed_chars"]>0
    assert len(contrast["cells"]["a_masked"]["answer_view"])==1


def test_interval_mask_preserves_unaffected_text_and_discordant_credit_is_zero():
    span = {"id":"obs","start":5,"end":30}
    assert subtract(span,{"id":"obs","start":10,"end":15})==[
        {"id":"obs","start":5,"end":10},{"id":"obs","start":15,"end":30}]
    assert subtract(span,{"id":"other","start":10,"end":15})==[span]
    assert aligned_credit(.3,-.5)==0 and aligned_credit(-.3,-.5)==-.3


def test_compression_reward_requires_both_answers_and_all_support():
    def trace(operation, prompt_tokens):
        sample = {"prompt_ids":[1]*prompt_tokens,"completion_ids":[2]}
        return {"actions":[{"operation":operation},{"sample":sample}],"answer_sample":sample}

    compressed, kept = trace("COMPRESS",3), trace("KEEP",7)
    valid = {"em":1,"support_visible_recall":1}
    saving = compression_efficiency(compressed,kept,0,valid,valid)
    assert saving == {"eligible":True,"ineligible_reasons":[],"includes_operation_cost":True,"tokens_a":{"manager":4,"reader":4},
                      "tokens_b":{"manager":8,"reader":8},"relative_saving":.5}
    missing = {**valid,"support_visible_recall":.5}
    assert compression_efficiency(compressed,kept,0,missing,valid)["relative_saving"]==0.
    assert compression_efficiency(trace("CLEAR",1),kept,0,valid,valid)["relative_saving"]==0.


def test_compression_cost_includes_management_operation():
    a = {"actions":[{"operation":"COMPRESS", "sample":{"prompt_ids":[1]*20,"completion_ids":[2]}}],
         "answer_sample":{"prompt_ids":[1],"completion_ids":[2]}}
    b = {"actions":[{"operation":"KEEP", "sample":{"prompt_ids":[1],"completion_ids":[2]}}],
         "answer_sample":{"prompt_ids":[1]*10,"completion_ids":[2]}}
    result = compression_efficiency(a,b,0,{"em":1,"support_visible_recall":1},{"em":1,"support_visible_recall":1})
    assert result["eligible"] and result["relative_saving"] < 0


def test_shared_interface_preserves_prefix_and_offers_compression_content():
    public = {"id":"shared", "question":"SECRET_FUTURE_QUESTION", "paragraphs":[
        ["First", ["Early fact.", "Late necessary fact."]], ["Second", ["Other early fact.", "Other late fact."]]]}
    def choose(messages, menu, public_options, working_sources):
        payload = json.loads(messages[-1]["content"])
        assert "verified_options" in payload and "phase" in payload
        if not payload["stage"].startswith("retrieve"):
            assert "SECRET_FUTURE_QUESTION" not in json.dumps(messages)
        op = "STORE" if payload["stage"].startswith("write") else "COMPRESS" if payload["stage"]=="compress" else "FINISH"
        label = next(k for k,v in menu.items() if v["operation"]==op and
                     (op != "COMPRESS" or v["strategy"]=="last_sentences"))
        return {"choice":label}
    factual, _ = rollout(FixedPolicy(),public,"rl_train",1,chooser=choose,record_prompts=True,
                         sentence_candidates=True,shared_interface=True)
    control, _ = rollout(FixedPolicy(),public,"rl_train",1,chooser=choose,factual=factual,intervention=2,
                         sentence_candidates=True,shared_interface=True)
    assert factual["actions"][2]["boundary"] == control["actions"][2]["boundary"]
    assert all("late" in r["text"].lower() for r in factual["answer_view"])
    assert control["actions"][2]["operation"] == "KEEP"


def test_public_support_inventory_distinguishes_missing_evidence_from_retrieval():
    public = {"paragraphs":[["Present",["Available sentence."]]]}
    gold = {"supporting_facts":[["Present",0],["Present",1],["Missing",0]]}
    assert public_support_inventory(public,gold)=={"expected":3,"outside_public":2}


def test_candidate_preview_stops_at_the_exact_source_interval():
    public = {"id":"preview","question":"Which city?",
              "paragraphs":[["Primary",["The city is Glasgow.","Beyond the permitted span " + "x"*800]]]}
    trace,_ = rollout(FixedPolicy(),public,"rl_train",1,sentence_candidates=True,record_prompts=True)
    content = json.loads(trace["actions"][1]["student_messages"][-1]["content"])
    previews = [op["preview"] for op in content["verified_options"].values() if op["operation"]=="STORE"]
    assert previews==["The city is Glasgow."]


def test_length_matched_counterfactual_preserves_reader_budget_and_hides_support():
    class TokenizingPolicy(FixedPolicy):
        def encode(self, messages):
            return [len(token) for token in re.findall(r"\w+|[^\w\s]",messages[-1]["content"])]

        def sample(self, messages, *args, **kwargs):
            sample = super().sample(messages,*args,**kwargs)
            sample.prompt_ids = self.encode(messages)
            return sample

    public,gold = split_row({"id":"matched","question":"Which city?","answer":"Glasgow",
                             "supporting_facts":[["Primary",0]],"context":[["Primary",["The city is Glasgow."]]]})
    policy = TokenizingPolicy()
    factual,_ = rollout(policy,public,"rl_train",1,sentence_candidates=True)
    control,_ = rollout(policy,public,"rl_train",1,factual=factual,intervention=1,sentence_candidates=True)
    contrast = factorial(policy,public,gold,factual,control,1,mask_mode="length_matched")
    assert contrast["eligible"] and contrast["evidence_credit"]==1
    original,masked = contrast["cells"]["a_visible"],contrast["cells"]["a_masked"]
    assert original["prompt_tokens"]==masked["prompt_tokens"]
    assert "Glasgow" not in masked["answer_view"][0]["text"] and masked["score"]["support_visible_recall"]==0
    assert contrast["cells"]["b_masked"]["reused_visible"]
