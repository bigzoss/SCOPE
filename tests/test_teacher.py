import json

from scope.atlas import StateAtlas
from scope.contracts import Budget, Scope
from scope.environments import ShopActions
from scope.rollout import manager_messages
from scope.state_graph import apply_observation
from scope.teacher import TeacherDistillation
from scope.tools import ToolExecutor, graph_tool_prompt


def test_teacher_uses_public_prefix_and_validates_student_visible_labels():
    atlas = StateAtlas(Scope("train", "1", "teacher"))
    tools = ToolExecutor(adapter=ShopActions())
    entity = "product:12345678"
    working = {"state_graph":True, "current_entity":entity}
    goal = "颜色是黑色。"
    budget = Budget()
    req = atlas.append_public(goal, "requirement")
    tools.execute(atlas, working, {"operation":"PROPOSE_STATE_UPDATE", "expected_revision":atlas.revision,
        "predicate":"requirement", "value":goal, "support_status":"supported",
        "evidence_refs":[{"id":req["id"],"start":0,"end":len(goal)}]}, "state", budget, goal)
    historical = atlas.append_public("Found 12345678 with several colors.", clickables=["12345678"])
    atlas.append_public("click[12345678]", "action_receipt", entity)
    text = "Black is listed. " + "x" * 240 + "old selected option" + "x" * 240 + " Buy Now"
    page = atlas.append_public(text, entity=entity, clickables=["Buy Now"])
    apply_observation(tools, atlas, working, page, budget, goal)
    messages = manager_messages(atlas, text, working, goal, "state", budget)
    original = atlas.fingerprint()
    received = []
    rule = {"operation":"FINISH","expected_revision":atlas.revision}

    def mock(proposal, targets, case="evidence_gap"):
        def response(request):
            received.append(json.loads(request[-1]["content"]))
            return json.dumps({"case":case,"proposal":proposal,"target_ids":targets})
        return TeacherDistillation(response, 1)

    teacher = mock({"operation":"RETRIEVE","query":"12345678","top_k":1,"entity":"task"}, [historical["id"]])
    label = teacher.annotate(atlas, working, goal, "state", budget, messages, tools, rule)
    assert label["verified_target_ids"] == [historical["id"]] and label["teacher_trigger"] == "historical_candidate_evidence"
    assert historical["id"] not in json.dumps(label["proposal"])
    assert historical["id"] in {e["id"] for e in received[0]["public_history"]}
    bad = mock({"operation":"RETRIEVE","query":historical["id"],"top_k":1}, [historical["id"]])
    assert bad.annotate(atlas, working, goal, "state", budget, messages, tools, rule) is None
    wrong = mock({"operation":"RETRIEVE","query":"Buy Now","top_k":1}, [historical["id"]])
    assert wrong.annotate(atlas, working, goal, "state", budget, messages, tools, rule) is None
    def choose(request):
        options = json.loads(request[-1]["content"])["verified_options"]
        selected = next(o for o in options if any(r["id"] == historical["id"] for r in o["retrieved"]))
        return json.dumps({"case":"evidence_gap","candidate_id":selected["candidate_id"]})
    assert TeacherDistillation(choose, 1).annotate(atlas, working, goal, "state", budget, messages, tools, rule)[
        "verified_target_ids"] == [historical["id"]]
    working["observations_since_summary"] = 6
    hidden_quote = "old selected option"
    start = text.index(hidden_quote)
    summary = mock({"operation":"SUMMARIZE","items":[{"text":hidden_quote,
        "evidence_ref":{"id":page["id"],"start":start,"end":start+len(hidden_quote)}}]}, [], "compression")
    context = manager_messages(atlas, text, working, goal, "context", budget)
    assert summary.annotate(atlas, working, goal, "context", budget, context, tools, rule) is None
    assert atlas.fingerprint() == original and budget.retrievals == 6
    atlas.close()


def test_graph_prompt_lists_only_current_phase_operations():
    state = graph_tool_prompt("state", 5)
    used = graph_tool_prompt("state", 5, lookup_used=True)
    context = graph_tool_prompt("context", 5)
    assert '"expected_revision":5' in state and "SUMMARIZE" not in state
    assert "RETRIEVE" in state and "RETRIEVE" not in used
    assert "SUMMARIZE" in context and "PROPOSE_STATE_UPDATE" not in context
