from scope.atlas import StateAtlas
from scope.candidate_actions import candidate_actions
from scope.contracts import Budget, Scope
from scope.environments import ShopActions
from scope.memory_gaps import diagnose, skipped_candidate_reason
from scope.tools import ToolExecutor


def test_gap_diagnosis_distinguishes_budget_and_absence():
    common = {"target":"confirmed color", "menu_covers":False, "allowed":True}
    assert diagnose(**common,public_witness=False,extracted=False)=="missing_evidence"
    assert diagnose(**common,public_witness=True,extracted=False)=="extractor_gap"
    assert diagnose(**common,public_witness=True,extracted=True)=="menu_gap"
    assert diagnose(**common,public_witness=True,extracted=True,prefix_complete=False)=="undetermined"
    assert skipped_candidate_reason({"interaction_feedback":{},"state_lookup_used":True},Budget(),"state")["class"]=="undetermined"
    assert skipped_candidate_reason({"interaction_feedback":{"evidence_ids":["e"]},"state_lookup_used":True},
                                    Budget(),"state")["detail"]=="state_lookup_used_this_turn"


def test_first_product_summary_uses_matching_earlier_public_search_evidence():
    atlas=StateAtlas(Scope("dev","1","gap"))
    entity="product:12345678"
    old=atlas.append_public("query "*400+"12345678 listing","observation",clickables=["12345678"])
    page=atlas.append_public("Instruction: short goal [SEP] product page [SEP] price 42","observation",entity)
    working={"state_graph":True,"current_entity":entity,"candidate_since":page["revision"]}
    menu=candidate_actions(atlas,working,"goal","context",Budget(),ToolExecutor(adapter=ShopActions()))
    summary,=[p for p in menu.values() if p["operation"]=="SUMMARIZE"]
    assert [item["evidence_ref"]["id"] for item in summary["items"]]==[old["id"],page["id"]]
    assert "12345678" in summary["items"][0]["text"] and "product page" in summary["items"][1]["text"]
    atlas.close()
