import json

from scope.alfworld import ALFWorldActions, ALFWorldEnvironment
from scope.contracts import Sample
from scope.evaluation import evaluate
from scope.interaction import ActionAdapter, execute_action, record_progress
from scope.io import digest, read_jsonl


def test_alfworld_dispatch_modes_use_public_names(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    env = ALFWorldEnvironment("http://unused", ["dev"], manifest)
    env.observation = "You see a cabinet 1, a cabinet 2, and a desklamp 1.\nYour task is to find book 9."
    sent = []

    def post(route, payload):
        assert route == "/step"
        sent.append(payload["command"])
        return {"text": "Nothing happens.", "done": False, "success": False}

    monkeypatch.setattr(env, "_post", post)
    adapter = ALFWorldActions()
    initial = env.observation
    state = adapter.observe(initial, {}, None, [])

    def run(command, mode):
        return execute_action(env, adapter, "Thought: plan\nAction: " + command, initial, state, mode)[3]

    assert run("look.", "legacy")["normalized_command"] == "look."
    assert run("look.", "normalize")["normalized_command"] == "look"
    assert run("go to receptacle", "grounded")["binding_status"] == "missing"
    lamp = run("go to desklamp.", "grounded")
    assert lamp["sent"] is False and lamp["binding_candidates"] == ["desklamp 1"]
    cabinet = run("open cabinet", "grounded")
    assert cabinet["binding_status"] == "ambiguous" and cabinet["binding_candidates"] == ["cabinet 1", "cabinet 2"]
    assert run("take book 9 from cabinet 1", "grounded")["binding_status"] == "unobserved"
    assert run("open cabinet 1", "grounded")["action_status"] == "environment_invalid"
    assert sent == ["look.", "look", "open cabinet 1"]


def test_alternating_actions_and_confirmed_state_change():
    adapter = ALFWorldActions()
    state = adapter.observe("You are facing the desk 1. You see a book 1.", {}, None, [])
    state = adapter.observe("You are not carrying anything.", state, "inventory", [])
    assert adapter.observe("You are facing the desk 1. You see a book 1.", state, "look", []) == state
    window = []
    same = digest(state)
    for command in ("look", "inventory", "look"):
        progress, loop = record_progress(window, same, same, command, "e1", True, False)
    assert progress == "legal_no_progress" and loop["actions"] == ["look", "inventory", "look"]
    held = adapter.observe("You pick up the book 1 from the desk 1.", state, "take book 1 from desk 1", [])
    assert held["held"] == ["book 1"]
    assert record_progress(window, same, digest(held), "take book 1 from desk 1", "e2", True, False) == ("state_changed", None)
    assert record_progress(window, digest(held), digest(held), "look", "e3", True, False) == ("legal_no_progress", None)


def test_new_benchmark_reuses_rollout_and_evaluation(tmp_path):
    class GridActions(ActionAdapter):
        def parse(self, response):
            return response.partition("Action: ")[2] or None

    class GridEnvironment:
        action_adapter = GridActions()
        actor_system = "Navigate the grid."
        task_goal = "Reach the next cell."

        def reset(self, task):
            self.metrics = {"success": 0., "r_strict": 0., "r_loose": 0., "purchased": 0.}
            return "cell 1"

        def step_command(self, command):
            if command == "right":
                self.metrics.update(success=1., r_strict=1., r_loose=1.)
                return "cell 2", True, {"is_error": False}
            return "cell 1", False, {"is_error": command == "left"}

        def close(self):
            pass

    class Policy:
        versions = {"actor": 0, "manager": 0}

        def __init__(self):
            self.actions = iter(["no command", "Action: left", "Action: scan", "Action: wait", "Action: scan", "Action: right"])
            self.prompts = []

        def fingerprint(self):
            return "fixed"

        def encode(self, messages):
            return [1]

        def sample(self, messages, role, phase, boundary, max_input, max_new, greedy=False):
            self.prompts.append(messages)
            return Sample(role, phase, [1], [2], [0.], next(self.actions), 0, boundary)

    class Registry:
        def evaluation_sequences(self, *args):
            return [["grid-dev"]]

    policy = Policy()
    config = {"environment": {"kind": "grid_fixture"},
              "runtime": {"action_mode": "grounded", "loop_policy": "state", "manager_mode": "rule"}}
    summary = evaluate(policy, GridEnvironment(), Registry(), config, tmp_path)
    assert summary["success_count"] == 1
    assert summary["public_interaction"] == {"format_errors": 1, "binding_failures": 0,
        "environment_invalid": 1, "invalid_no_progress": 2, "legal_no_progress": 3, "public_state_loops": 1,
        "episode_termination_reasons": {"environment_done": 1}}
    row = read_jsonl(tmp_path / "results.jsonl")[0]
    actions = [e for e in row["history"] if e["phase"] == "execute"]
    evidence = {e["id"]: e["text"] for e in row["public_evidence"]}
    assert actions[0]["observation_before_id"] == actions[0]["observation_after_id"] and actions[0]["sent"] is False
    assert evidence[actions[-1]["observation_before_id"]] == "cell 1"
    assert evidence[actions[-1]["observation_after_id"]] == "cell 2"
    assert actions[-1]["public_state_delta"] and actions[-1]["termination"] == "eos"
    assert "Public state loop" in policy.prompts[-1][-1]["content"]
    assert json.loads((tmp_path / "summary.json").read_text())["evaluation_contract"]["public_interaction"]["action_mode"] == "grounded"
