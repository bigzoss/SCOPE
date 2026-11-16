from scope.alfworld import ALFWorldActions
from scope.alfworld_context import decision_card, historical_quote, treatment_candidates, treatment_stage
from scope.atlas import StateAtlas
from scope.contracts import Budget, Scope
from scope.state_graph import apply_observation
from scope.tools import ToolExecutor


def test_short_card_uses_latest_public_binding_and_does_not_repeat_visible_quote():
    adapter = ALFWorldActions()
    atlas = StateAtlas(Scope("dev", "mug", "e"))
    goal = "put a mug in a cabinet."
    working = {"state_graph": True, "public_state": {}}
    try:
        tools = ToolExecutor(adapter=adapter)
        budget = Budget()
        for text in ("On the desk 2, you see a mug 1.", "You pick up the mug 1 from the desk 2.",
                     "You arrive at cabinet 1. The cabinet 1 is closed."):
            record = atlas.append_public(text)
            working["public_state"] = adapter.observe(text, working["public_state"], None, [])
            working["observation_id"] = record["id"]
            apply_observation(tools, atlas, working, record, budget, goal)
        card = decision_card(atlas, working, goal)
        assert any("mug 1 · holding: True" in row["text"] for row in card)
        assert all("location: desk 2" not in row["text"] for row in card)
        quote = historical_quote(atlas, working, card, [text])
        assert quote and "pick up" in quote["text"] and quote["id"] != record["id"]
        assert historical_quote(atlas, working, card, [text, quote["text"]]) is None
    finally:
        atlas.close()


def test_alfworld_treatment_requires_native_confirmation_before_delivery():
    adapter = ALFWorldActions()
    goal = "put a cool tomato in countertop."
    seen = "You arrive at fridge 1. On the countertop 1, you see a tomato 1."
    state = adapter.observe(seen, {}, None, [])
    state = adapter.observe("You pick up the tomato 1 from the countertop 1.", state, None, [])
    attempted = adapter.observe("Nothing happens.", state, "cool tomato 1 with fridge 1", [])
    assert treatment_stage(attempted, goal)["command"] == "cool tomato 1 with fridge 1"
    assert not treatment_stage(attempted, goal)["treatment_confirmed"]
    assert all(row["command"] != "put tomato 1 in/on countertop 1" for row in treatment_candidates(
        attempted, goal, [{"command": "put tomato 1 in/on countertop 1"}]))
    state = adapter.observe("You cool the tomato 1 using the fridge 1.", attempted, None, [])
    assert treatment_stage(state, goal)["command"] == "go to countertop 1"
    state = adapter.observe("You arrive at countertop 1.", state, None, [])
    assert treatment_stage(state, goal)["command"] == "put tomato 1 in/on countertop 1"
