import pytest

from scope.alfworld import ALFWorldActions
from scope.atlas import StateAtlas
from scope.candidate_actions import candidate_actions
from scope.contracts import Budget, Scope
from scope.environments import ShopActions
from scope.decision_evidence import review
from scope.memory_gaps import confirmation_status
from scope.state_graph import apply_observation
from scope.tools import ToolExecutor, context_view


@pytest.mark.parametrize(("adapter","command","entity","before","after"), [
    (ShopActions(), "click[蓝色]", "product:12345678", "可选颜色：蓝色", "Selected color: 蓝色"),
    (ALFWorldActions(), "take book 1 from desk 1", "task", "You see a book 1 on the desk 1.",
     "You pick up the book 1 from the desk 1."),
])
def test_post_action_confirmation_is_shared(adapter, command, entity, before, after):
    atlas = StateAtlas(Scope("dev","case","1"))
    tools, budget = ToolExecutor(adapter=adapter), Budget()
    working = {"state_graph":True,"shared_memory":True,"current_entity":entity}
    atlas.append_public(before,"observation",entity)
    receipt = atlas.append_public(command,"action_receipt",entity)
    subject, predicate, value = adapter.confirmation_target(command,entity)
    target = {"entity":subject,"predicate":predicate,"value":value,
              "attempt_id":receipt["id"],"attempt_revision":receipt["revision"]}
    assert confirmation_status(atlas,tools,target)["class"] == "missing_evidence"
    record = atlas.append_public(after,"observation",entity)
    assert confirmation_status(atlas,tools,target)["class"] == "extractor_gap"
    apply_observation(tools,atlas,working,record,budget,"goal")
    status = confirmation_status(atlas,tools,target)
    assert status["status"] == "confirmed" and status["evidence_ids"] == [record["id"]]
    another = atlas.append_public(command,"action_receipt",entity)
    assert confirmation_status(atlas,tools,{**target,"attempt_revision":another["revision"]})["status"] == "pending"
    atlas.close()


def test_tool_ablation_does_not_disable_the_other_tool():
    atlas = StateAtlas(Scope("dev","case","2"))
    a = atlas.append_public("You are facing the desk 1.","observation")
    atlas.append_public("You are facing the bed 1.","observation")
    working = {"state_graph":True,"shared_memory":True,"observations_since_summary":6,
               "interaction_feedback":{"evidence_ids":[a["id"]]}}
    tools = ToolExecutor(adapter=ALFWorldActions())
    for disabled in ("RETRIEVE","SUMMARIZE"):
        menu = candidate_actions(atlas,{**working,"disabled_memory_tools":[disabled]},"goal","context",Budget(),tools)
        assert {p["operation"] for p in menu.values()} == {"FINISH",({"RETRIEVE","SUMMARIZE"}-{disabled}).pop()}
    atlas.close()


def test_decision_lookup_returns_exact_span_and_does_not_repeat():
    atlas = StateAtlas(Scope("dev","case","3"))
    tools, budget = ToolExecutor(adapter=ALFWorldActions()), Budget()
    working = {"state_graph":True,"shared_memory":True,"decision_evidence":True,"current_entity":"task",
               "recovery_slots":[("alf:apple 1","held"),("alf:apple 2","held")],"public_state":{"held":["apple 1"]}}
    old = atlas.append_public("You pick up the apple 1 from the desk 1. Unrelated note.","observation")
    apply_observation(tools,atlas,working,old,budget,"goal")
    working["observation_id"] = atlas.append_public("You arrive at cabinet 1.","observation")["id"]
    assert [r["status"] for r in review(atlas,working,"goal")] == ["supported","pending"]
    menu = candidate_actions(atlas,working,"goal","state",budget,tools)
    proposal, = [p for p in menu.values() if p["operation"]=="RETRIEVE"]
    assert (proposal["query"],proposal["entity"],proposal["predicate"],proposal["as_of_revision"]) == (
        old["id"],"alf:apple 1","held",atlas.revision)
    result = tools.execute(atlas,working,proposal,"state",budget,"goal")
    assert result["retrieved"][0]["text"] == "You pick up the apple 1 from the desk 1"
    assert [item["id"] for item in context_view(atlas,working,"goal")["selected_evidence"]] == [old["id"]]
    assert not context_view(atlas,{**working,"recovery_slots":[]},"goal")["selected_evidence"]
    working.update(selected_ids=[],state_lookup_used=False)
    assert not any(p["operation"]=="RETRIEVE" for p in candidate_actions(atlas,working,"goal","state",budget,tools).values())
    atlas.close()
