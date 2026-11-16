import pytest

from scope.interaction import ActionAdapter, action_text, execute_action
from scope.scienceworld_actions import ScienceWorldActions, focus_guard, science_command_guard, thermal_trial
from scope.atlas import StateAtlas
from scope.candidate_actions import candidate_actions
from scope.contracts import Budget, Scope
from scope.state_graph import apply_observation
from scope.tools import ToolExecutor


THERMOMETER = ("Measure unknown substance B. First, focus on the thermometer. "
               "Next, focus on the unknown substance B.")


@pytest.mark.parametrize("goal,command,blocked", [
    (THERMOMETER, "focus on unknown substance B", True),
    (THERMOMETER, "focus on thermometer", False),
    ("Turn on the blue light bulb. First, focus on the blue light bulb.", "focus on orange light bulb", True),
    ("Boil rubber. First, focus on the substance.", "focus on rubber", False),
    ("After the experiment, focus on the steepest inclined plane.", "focus on inclined plane G", True),
    ("If the trait is dominant, focus on the green box.", "focus on green box", True),
    (THERMOMETER, "open cupboard", False),
])
def test_focus_requires_public_goal_and_stage(goal, command, blocked):
    assert bool(focus_guard(command, goal, [])) is blocked


def test_focus_order_advances_only_on_public_confirmation():
    attempt = {"command": "focus on thermometer", "observation": "Nothing happens.", "is_error": False}
    assert focus_guard("focus on unknown substance B", THERMOMETER, [attempt])
    attempt["observation"] = "You focus on the thermometer."
    assert not focus_guard("focus on unknown substance B", THERMOMETER, [attempt])


def test_dispatch_guard_blocks_unready_focus_without_advancing_environment():
    class ScienceCommands(ActionAdapter):
        parse = staticmethod(action_text)

    class Environment:
        def public_command_guard(self, command):
            return focus_guard(command, THERMOMETER, [])

        def step_command(self, command):
            pytest.fail("Focus was sent to the environment")

    observation, done, public, trace = execute_action(
        Environment(), ScienceCommands(), "Action: focus on unknown substance B", "You see a cupboard.", {}, "normalize")
    assert observation == "You see a cupboard." and not done and public["is_error"]
    assert trace["action_status"] == "precondition_conflict" and trace["sent"] is False


def test_conditional_focus_opens_after_typed_public_measurement_and_keeps_its_source():
    goal = ("First, focus on the thermometer. Next, focus on the water. "
            "If the melting point of water is above 20.0 degrees celsius, focus on the red box. "
            "If the melting point of water is below 20.0 degrees celsius, focus on the green box.")
    history = [{"command": f"focus on {name}", "observation": f"You focus on the {name}."}
               for name in ("thermometer", "water")]
    adapter = ScienceWorldActions()
    attempted = adapter.observe("You use the thermometer on the water.", {}, None, [])
    assert science_command_guard("focus on green box", goal, attempted, history, "")
    temp = adapter.observe("The temperature of water is 18.5 degrees Celsius.", attempted, None, [])
    assert science_command_guard("focus on green box", goal, temp, history, "")
    observation = "The melting point of water is 18.5 degrees Celsius."
    state = adapter.observe(observation, temp, None, [])
    assert not science_command_guard("focus on green box", goal, state, history, observation)
    assert science_command_guard("focus on red box", goal, state, history, observation)
    assert science_command_guard("reset task", goal, state, history, observation)
    with_atlas = StateAtlas(Scope("dev", "water", "episode"))
    try:
        tools = ToolExecutor(adapter=adapter)
        budget = Budget()
        config = {"state_graph": True, "shared_memory": True, "decision_evidence": True, "public_state": state}
        historical = with_atlas.append_public(observation)
        apply_observation(tools, with_atlas, config, historical, budget, goal)
        current = with_atlas.append_public("This room is called the kitchen.")
        config["observation_id"] = current["id"]
        apply_observation(tools, with_atlas, config, current, budget, goal)
        menu = candidate_actions(with_atlas, config, goal, "state", budget, tools)
        assert any(p["operation"] == "RETRIEVE" and p["query"] == historical["id"] for p in menu.values())
    finally:
        with_atlas.close()


def test_science_projection_needs_environment_text_and_preserves_exact_spans():
    a = ScienceWorldActions()
    obs = ("This room is called the kitchen. On the table is: a red box. "
           "Water is now liquid.\nterminal 1 on battery is now connected to terminal 2 on buzzer\n"
           "You move the thermometer to the inventory.")
    facts = list(a.public_facts({"text": obs}))
    assert {f.predicate for f in facts} == {"room", "object_location", "material_state", "circuit_connection"}
    assert all(obs[f.start:f.end] for f in facts)
    assert not list(a.public_facts({"text": "Task description: The melting point of water is 100 degrees Celsius."}))


def test_native_thermometer_result_binds_to_its_public_action_receipt():
    adapter = ScienceWorldActions()
    reading = "the thermometer measures a temperature of 10 degrees celsius"
    atlas = StateAtlas(Scope("dev", "thermometer", "e"))
    try:
        assert not list(adapter.public_facts({"text": reading}))
        receipt = atlas.append_public("use thermometer on soap", "action_receipt")
        record = atlas.append_public(reading, action_receipt_id=receipt["id"])
        fact, = adapter.public_facts(record)
        assert fact.entity == "sci:soap" and fact.predicate == "temperature_c" and fact.value == "10"
        assert record["text"] == reading and record["action_receipt_id"] == receipt["id"]
        with pytest.raises(ValueError, match="immediate public action receipt"):
            atlas.append_public(reading, action_receipt_id=receipt["id"])
        assert adapter.observe("You move through the door to the kitchen.", {}, "go to kitchen", [])["facts"]["task"]["room"] == "sci:kitchen"
    finally:
        atlas.close()


def test_phase_interval_uses_bound_receipts_and_does_not_unlock_straddling_threshold():
    goal = ("First, focus on the thermometer. Next, focus on the solid unknown substance G. "
            "If the melting point of solid unknown substance G is above 50.0 degrees celsius, focus on the red box. "
            "If the melting point of solid unknown substance G is below 50.0 degrees celsius, focus on the blue box.")
    history = [{"command": f"focus on {name}", "observation": f"You focus on the {name}."}
               for name in ("thermometer", "solid unknown substance G")]
    def measure(name, temperature):
        return {"command": f"use thermometer on {name}",
                "observation": f"the thermometer measures a temperature of {temperature} degrees celsius"}
    solid = measure("solid unknown substance g", 40)
    unbound = measure("liquid unknown substance h", 45)
    assert thermal_trial(goal, history + [solid, unbound])["interval_c"] is None
    wide = history + [solid, measure("liquid unknown substance g", 75)]
    assert science_command_guard("focus on blue box", goal, {}, wide, "", thermal=True)
    narrow = history + [solid, measure("liquid unknown substance g", 45)]
    assert thermal_trial(goal, narrow)["interval_c"] == [39.5, 45.5]
    assert not science_command_guard("focus on blue box", goal, {}, narrow, "", thermal=True)
    assert science_command_guard("focus on red box", goal, {}, narrow, "", thermal=True)
