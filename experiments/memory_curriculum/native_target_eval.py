"""Frozen HotpotQA manager on native ScienceWorld and AgentBoard task instances."""
import argparse
import ast
import fcntl
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from experiments.memory_curriculum.shared_memory_eval import FixedTasks, read
from scope.checkpoint import load_checkpoint
from scope.cli import model_identity, source_hashes, verify_checkpoint_base
from scope.config import DEFAULT_RUNTIME
from scope.contracts import ContractError
from scope.evaluation import evaluate
from scope.interaction import ActionAdapter, action_text
from scope.io import atomic_json, digest, file_sha
from scope.policy import HFPolicy

ROOT = Path("/home/hyx/lm_workspace/SCOPE")
AGENTBOARD = ROOT / "baseline/AgentBoard"
BABY_CODE = AGENTBOARD / "agentboard/environment/babyai_env.py"
PDDL_CODE = AGENTBOARD / "agentboard/environment/pddl_env/pddl_env.py"
PDDL_PACKAGE = PDDL_CODE.parent / "pddlgym"
SEED = 233
STEP_CAP = 30
DOMAINS = ("pddl", "babyai", "scienceworld")


def literal_constant(path, name):
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise ContractError(f"{name} missing from {path}")


def labels(run, domain, expected):
    path = run / "eval_data" / f"{domain}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [row["id"] for row in rows] != list(range(expected)):
        raise ContractError(f"{domain} official row IDs differ")
    return rows, file_sha(path)


class TextCommands(ActionAdapter):
    parse = staticmethod(action_text)

    def normalize(self, command):
        return " ".join(command.lower().strip().rstrip(".").split())


class Target:
    action_adapter = TextCommands()

    def set_score(self, success=0, progress=0.):
        self.metrics = {"success": int(success), "r_strict": int(success),
                        "r_loose": float(progress), "purchased": 0}

    def close(self):
        pass

    def result(self, observation, done=False, valid=True):
        return observation, bool(done), {"is_error": not valid, "clickables": []}


class ScienceWorld(Target):
    actor_system = (
        "You are controlling a ScienceWorld experiment. Read the goal and public observations. "
        "Use text commands such as look around, inventory, go to <room>, pick up <object>, "
        "open <container>, focus on <object>, or check valid actions. "
        "Return one Action: line with an exact game command."
    )

    def __init__(self):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv(envStepLimit=STEP_CAP)
        self.tasks = {}
        for task in self.env.get_task_names():
            self.env.load(task, 0)
            variants = sorted(self.env.get_variations_test(), key=lambda i: digest([SEED, task, i]))
            if len(variants) < 3:
                raise ContractError(f"Insufficient native test variations for {task}")
            for index in variants[:3]:
                self.tasks[f"{task}:{index}"] = {"task": task, "variation": index}

    def reset(self, task_id):
        task = self.tasks[task_id]
        self.env.load(task["task"], task["variation"], simplificationStr="")
        observation, _ = self.env.reset()
        self.task_goal = self.env.get_task_description()
        self.set_score()
        return observation + "\n" + self.env.inventory()

    def step_command(self, command):
        if command == "check valid actions":
            actions = sorted(set(self.env.get_valid_action_object_combinations()))
            return self.result("Valid commands: " + "; ".join(actions))
        observation, _, done, info = self.env.step(command)
        score = max(0., min(1., info["score"] / 100.))
        self.set_score(done and score == 1., max(score, self.metrics["r_loose"]))
        return self.result(observation, done, valid="no known action matches" not in observation.lower())

    def shutdown(self):
        self.env.close()


class BabyAI(Target):
    action_names = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
    actor_system = (
        "Control the BabyAI agent. The visible local view is egocentric: ahead increases forward, "
        "and lateral offsets are negative left and positive right. Objects outside this view are unknown. "
        "Use left or right to turn 90 degrees, forward to move one tile, pickup or drop on the tile "
        "directly ahead, toggle to operate doors, and done only if necessary. Return one Action: line."
    )

    def __init__(self, run):
        import gymnasium
        import minigrid
        from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
        self.make = gymnasium.make
        self.colors, self.objects = IDX_TO_COLOR, IDX_TO_OBJECT
        self.env = None
        levels = literal_constant(BABY_CODE, "all_levels")
        selected = list(range(1, 22)) + [24, 25, 27, 28, 29, 31, 32]
        rows, self.label_sha = labels(run, "babyai", 112)
        instances = [(levels[level], seed) for level in selected for seed in range(1234, 1238)]
        self.tasks = {str(row["id"]): {"level": level, "seed": seed, "goal": row["goal"]}
                      for row, (level, seed) in zip(rows, instances, strict=True)}

    def describe(self, observation):
        view = observation["image"]
        center, back = view.shape[0] // 2, view.shape[1] - 1
        visible = []
        for x, y in np.ndindex(view.shape[:2]):
            kind, color, state = map(int, view[x, y])
            name = self.objects[kind]
            if name in {"unseen", "empty", "floor"} or (x, y) == (center, back):
                continue
            description = name if name == "wall" else f"{self.colors[color]} {name}"
            if name == "door":
                description += " " + ("open", "closed", "locked")[state]
            visible.append(f"{description} (left/right {x-center}, ahead {back-y})")
        kind, color, _ = map(int, view[center, back])
        held = self.objects[kind]
        inventory = "empty" if held in {"empty", "unseen", "floor"} else f"{self.colors[color]} {held}"
        return f"Heading: {observation['direction']} (0 east, 1 south, 2 west, 3 north). Holding: {inventory}. Visible: " + "; ".join(visible)

    def reset(self, task_id):
        self.close()
        task = self.tasks[task_id]
        self.env = self.make(task["level"])
        observation, _ = self.env.reset(seed=task["seed"])
        self.task_goal = observation["mission"]
        if self.task_goal.strip().casefold() != task["goal"].strip().casefold():
            raise ContractError(f"BabyAI native mission mismatch for {task_id}")
        self.set_score()
        return self.describe(observation)

    def step_command(self, command):
        if command not in self.action_names:
            return self.result("Unknown command. Use left, right, forward, pickup, drop, toggle, or done.", valid=False)
        obs, reward, terminated, truncated, _ = self.env.step(self.action_names.index(command))
        success = terminated and reward > 0
        self.set_score(success, success)
        return self.result(self.describe(obs), terminated or truncated)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class PDDL(Target):
    actor_system = (
        "Control a symbolic planning environment to satisfy every goal condition. "
        "Use check valid actions to list commands or look around for the current full state. "
        "Copy exact commands from the list. Return one Action: line."
    )

    def __init__(self, run):
        # Import the pinned AgentBoard engine without the package's optional image renderers.
        spec = importlib.util.spec_from_loader("pddlgym", loader=None, is_package=True)
        spec.submodule_search_locations = [str(PDDL_PACKAGE)]
        sys.modules["pddlgym"] = importlib.util.module_from_spec(spec)
        from pddlgym.core import PDDLEnv
        self.engine = PDDLEnv
        self.formats = literal_constant(PDDL_CODE, "predicate_map")
        rows, self.label_sha = labels(run, "pddl", 60)
        instances = [(name, index) for name, size in (
            ("gripper", 20), ("blockworld", 10), ("barman", 20), ("tyreworld", 10)
        ) for index in range(size)]
        self.tasks = {str(row["id"]): {"domain": name, "index": index, "goal": row["goal"]}
                      for row, (name, index) in zip(rows, instances, strict=True)}
        self.env = None

    def render(self, literal):
        args = [var.name for var in literal.variables]
        predicate = literal.predicate.name
        return self.formats[predicate].format(*args) if predicate in self.formats else " ".join([predicate, *args])

    def render_state(self):
        return " ".join(sorted(self.render(x) for x in self.state.literals))

    def reset(self, task_id):
        task = self.tasks[task_id]
        folder = PDDL_PACKAGE / "pddl"
        self.env = self.engine(str(folder / (task["domain"] + ".pddl")), str(folder / task["domain"]),
                               operators_as_actions=True, dynamic_action_space=True)
        self.env.fix_problem_index(task["index"])
        self.state, _ = self.env.reset()
        self.goal_literals = set(self.state.goal.literals)
        if not self.goal_literals:
            raise ContractError(f"Empty PDDL goal {task_id}")
        self.task_goal = "The goal is to satisfy the following conditions: " + ", ".join(
            sorted(self.render(x) for x in self.goal_literals))
        self.set_score()
        return self.render_state()

    def step_command(self, command):
        choices = {self.action_adapter.normalize(self.render(x)): x
                   for x in self.env.action_space.all_ground_literals(self.state)}
        if command == "check valid actions":
            return self.result("Valid commands: " + "; ".join(sorted(choices)))
        if command == "look around":
            return self.result(self.render_state())
        if command not in choices:
            return self.result("Invalid command in this state. Use check valid actions.", valid=False)
        previous = self.state
        self.state, _, done, _ = self.env.step(choices[command])
        progress = len(self.goal_literals & set(self.state.literals)) / len(self.goal_literals)
        self.set_score(done and progress == 1., max(progress, self.metrics["r_loose"]))
        return self.result(self.render_state(), done, valid=done or self.state != previous)


def create(domain, run):
    if domain == "scienceworld":
        return ScienceWorld()
    return {"pddl": PDDL, "babyai": BabyAI}[domain](run)


def execute(args, output):
    env = create(args.domain, args.run)
    try:
        manifest = {"benchmark": args.domain, "tasks": env.tasks,
                    "label_sha256": getattr(env, "label_sha", None),
                    "protocol": ("three fixed native held-out variations per task, native score"
                                 if args.domain == "scienceworld" else
                                 "AgentBoard instance selection, native transitions and terminal success")}
        if args.smoke:
            for task in env.tasks:
                observation = env.reset(task)
                if not observation or not env.task_goal:
                    raise ContractError(f"Empty task {task}")
                env.close()
            observation = env.reset(next(iter(env.tasks)))
            command = {"babyai": "left", "pddl": "check valid actions",
                       "scienceworld": "look around"}[args.domain]
            result, _, flags = env.step_command(command)
            if not result or flags["is_error"]:
                raise ContractError(f"Native step failed: {args.domain}")
            atomic_json(output / "smoke.json", {"passed": True, "reset_count": len(env.tasks), **manifest})
            return
        train = args.run / "train"
        receipt = read(train / "receipt.json")
        summary = read(train / "summary.json")
        checkpoint = train / "checkpoints" / receipt["selected_checkpoint"]
        if not summary["complete"] or file_sha(checkpoint / "state.pt") != summary["checkpoint_sha256"]:
            raise ContractError("Frozen source checkpoint did not verify")
        model = Path(receipt["model"]["path"])
        verify_checkpoint_base(checkpoint, model_identity(model, ROOT / "runs/.identity-cache"))
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / total))
        policy = HFPolicy(model, gradient_checkpointing=False)
        load_checkpoint(checkpoint, policy, restore_random=False)
        runtime = {**DEFAULT_RUNTIME, "state_graph": True, "memory_choice": True, "shared_memory": True,
                   "decision_evidence": True, "action_mode": "normalize", "loop_policy": "state",
                   "max_environment_steps": STEP_CAP, "max_turns": 2 * STEP_CAP, "total_tokens": 850000}
        config = {"environment": {"kind": args.domain}, "runtime": runtime,
                  "evaluation": {"seed": SEED, "sequence_length": 1}}
        pinned = {**manifest, "checkpoint_sha256": summary["checkpoint_sha256"],
                  "checkpoint": str(checkpoint), "parameter_fingerprint": policy.fingerprint(),
                  "source_training_receipt_sha256": file_sha(train / "receipt.json"),
                  "runtime": runtime, "driver_sha256": file_sha(Path(__file__)),
                  "source_sha256": source_hashes(), "target_updates": 0, "gpu": args.gpu}
        path = output / "receipt.json"
        if path.exists() and read(path) != pinned:
            raise ContractError("Frozen native evaluation receipt changed")
        atomic_json(path, pinned)
        evaluate(policy, env, FixedTasks(list(env.tasks)), config, output, "test")
    finally:
        env.close()
        if isinstance(env, ScienceWorld):
            env.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    output = args.run / args.domain / ("preflight" if args.smoke else "transferred")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, output)


if __name__ == "__main__":
    main()
