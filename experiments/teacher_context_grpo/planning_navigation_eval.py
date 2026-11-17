"""Frozen PDDL and BabyAI development comparisons using the native benchmark adapters."""
import argparse
import fcntl
from pathlib import Path
import re
from unittest.mock import patch

import target_action_adaptation as base
from scope.babyai_navigation import (PublicLocalMap, ahead, delivery_tiles, matches_item,
                                     matching_targets, mission_stages, ordered_mission)
from scope.io import atomic_json, read_jsonl
from scope.pddl_subgoals import avoid_repeated_inverse, rank_public_actions, relevant_state_facts
from scope.pddl_public_model import PublicSTRIPSModel, guard_repeated_transitions, canonical_fact, readable_fact


class MeteredPDDL(base.AdaptedPDDL):
    def reset(self, task_id):
        self.recent_states = []
        self.recent_edges = []
        obs = super().reset(task_id)
        self.measure()
        return obs

    def measure(self):
        self.metrics["final_coverage"] = len(self.goal_literals & set(self.state.literals)) / len(self.goal_literals)

    def step_command(self, command):
        previous = frozenset(self.state.literals)
        outcome = super().step_command(command)
        if frozenset(self.state.literals) != previous:
            self.recent_states.append(previous)
            self.recent_edges.append((previous, self.action_adapter.normalize(command)))
        self.measure()
        return outcome


class PlannedPDDL(MeteredPDDL):
    inverse = False
    strict_inverse = False

    def public_action_candidates(self, atlas, working, goal, observation, history):
        env = self.env
        available = {self.action_adapter.normalize(self.render(obj)): obj
                     for obj in env.action_space.all_ground_literals(self.state)}
        operators = {name.lower(): op for name, op in env.domain.operators.items()}
        state = set(self.state.literals)
        ranks, blockers = rank_public_actions(operators, available, state, self.goal_literals,
                                              self.recent_states, self.inverse, self.recent_edges,
                                              strict_inverse=self.strict_inverse)
        self.action_signals = {row["command"]: row for row in ranks}
        selected = (avoid_repeated_inverse(ranks) if self.inverse else ranks)[:8]
        self.blockers = blockers
        return [{"command": row["command"], "reason": "public_pddl_consequences",
                 "evidence_ids": [working["observation_id"]]} for row in selected]

    def render_state(self):
        checklist = self.goal_checklist()
        pending = checklist["pending"]
        if not pending:
            return "All public goals satisfied."
        return ("Pending goal conditions: " + "; ".join(pending)
                + "\nSatisfied conditions: " + "; ".join(checklist["satisfied"])
                + "\nPublic state signature: " + base.digest(sorted(map(str, self.state.literals)))[:12])

    def brief(self, menu):
        def ordered(items):
            return sorted(self.render(x) for x in items)
        rows = []
        for entry in menu:
            row = self.action_signals[entry["command"]]
            labels = [f"{key} {', '.join(ordered(row[key]))}" for key in ("advances", "unblocks", "risks") if row[key]]
            key = "repeated_inverse" if self.strict_inverse else "returns_to_recent_state"
            if self.inverse and row[key]:
                labels.append("returns to a recently visited state")
            if labels:
                rows.append(f"{entry['label']}: " + "; ".join(labels))
        return ("\nPublic goal prerequisites: " + "; ".join(ordered(self.blockers)[:8]) +
                "\nAction consequences: " + " | ".join(rows))


class LoopAwarePDDL(PlannedPDDL):
    inverse = True


class CompactStatePDDL(MeteredPDDL):
    def render_state(self):
        checklist = self.goal_checklist()
        facts = relevant_state_facts(set(self.state.literals), self.goal_literals)
        return ("Pending goal conditions: " + "; ".join(checklist["pending"])
                + "\nSatisfied conditions: " + "; ".join(checklist["satisfied"])
                + "\nCurrent public state: " + "; ".join(sorted({self.render(fact) for fact in facts})))

    def public_description(self):
        return MeteredPDDL.render_state(self)


class ReorderedPDDL(PlannedPDDL):
    render_state = MeteredPDDL.render_state


class CompactReorderedPDDL(PlannedPDDL):
    render_state = CompactStatePDDL.render_state


class ExactLoopPDDL(CompactReorderedPDDL):
    inverse = True
    strict_inverse = True


class DependencyStatePDDL(MeteredPDDL):
    """Preserve public action resources; old experimental arms remain unchanged."""
    dependency_ranking = False
    suppress_loops = False
    guard_repeated = False
    whole_goal = False
    resource_batch = False

    def reset(self, task_id):
        self.public_model = None
        super().reset(task_id)
        space = self.env.action_space
        space.all_ground_literals(self.state)
        operators = {name.lower(): op for name, op in self.env.domain.operators.items()}
        self.public_model = PublicSTRIPSModel(operators, space._all_ground_literals)
        return self.render_state()

    def render_state(self):
        # reset() renders before the grounding is available; preserve all facts then.
        facts = (self.public_model.state_view(set(self.state.literals), self.goal_literals)
                 if self.public_model else self.state.literals)
        checklist = self.goal_checklist()
        return ("Pending goal conditions: " + "; ".join(checklist["pending"])
                + "\nSatisfied conditions: " + "; ".join(checklist["satisfied"])
                + "\nCurrent public action resources: " + "; ".join(sorted(self.render(f) for f in facts)))

    def public_action_candidates(self, atlas, working, goal, observation, history):
        if not self.dependency_ranking:
            return super().public_action_candidates(atlas, working, goal, observation, history)
        available = {self.action_adapter.normalize(self.render(a)): a
                     for a in self.env.action_space.all_ground_literals(self.state)}
        ranked = self.public_model.rank(available, frozenset(self.state.literals), self.goal_literals,
                                       self.recent_states, self.recent_edges, self.suppress_loops,
                                       whole_goal=self.whole_goal, resource_batch=self.resource_batch)
        self.action_signals = {r['command']: r for r in ranked}
        if self.guard_repeated:
            # Never remove the only escape. Changed cargo/resource states do not
            # count as the same transition, so legitimate return trips remain.
            ranked = guard_repeated_transitions(ranked)
        return [{"command": row['command'], "reason": "public_grounded_dependencies",
                 "evidence_ids": [working['observation_id']]} for row in ranked[:8]]

    def brief(self, menu):
        if not self.dependency_ranking:
            return ""
        return "\nPublic planning estimates (heuristics, not rewards): " + " | ".join(
            f"{entry['label']}: next-goal distance {self.action_signals[entry['command']]['distance_after']}; "
            f"repeated state transition {self.action_signals[entry['command']]['repeated_inverse']}"
            for entry in menu)


class DependencyRankedPDDL(DependencyStatePDDL):
    dependency_ranking = True


class DependencyLoopPDDL(DependencyRankedPDDL):
    suppress_loops = True


class DependencyGuardedPDDL(DependencyLoopPDDL):
    guard_repeated = True


class CanonicalStatePDDL(DependencyGuardedPDDL):
    """Keep exact native commands, but preserve argument order/arity in facts."""
    def reset(self, task_id):
        observation = super().reset(task_id)
        self.task_goal = 'Satisfy all public goal facts: ' + '; '.join(
            sorted(canonical_fact(f) for f in self.goal_literals))
        return observation

    def goal_checklist(self):
        state = set(self.state.literals)
        return {'satisfied': sorted(canonical_fact(f) for f in self.goal_literals & state),
                'pending': sorted(canonical_fact(f) for f in self.goal_literals - state)}

    def render_state(self):
        facts = (self.public_model.state_view(set(self.state.literals), self.goal_literals)
                 if self.public_model else self.state.literals)
        checklist = self.goal_checklist()
        return ('Public goals: ' + base.json.dumps(checklist)
                + '\nPublic facts (predicate arguments retain domain order): '
                + '; '.join(sorted(canonical_fact(f) for f in facts)))


class WholeGoalPDDL(CanonicalStatePDDL):
    whole_goal = True

    def brief(self, menu):
        return '\nPublic relaxed all-goal estimates (not a plan or reward): ' + ' | '.join(
            f"{e['label']}: sum-cost {self.action_signals[e['command']]['aggregate_distance_after']}; "
            f"same-trip free-hand load {self.action_signals[e['command']]['batch_load_opportunity']}; "
            f"public transport phase {self.action_signals[e['command']]['transport_role']}; "
            f"achieves {sorted(canonical_fact(f) for f in self.action_signals[e['command']]['advances'])}; "
            f"undoes {sorted(canonical_fact(f) for f in self.action_signals[e['command']]['risks'])}"
            for e in menu)


class ResourceGoalPDDL(WholeGoalPDDL):
    resource_batch = True


class ReadableStatePDDL(DependencyGuardedPDDL):
    def reset(self, task_id):
        observation = super().reset(task_id)
        self.task_goal = 'The goal is to satisfy the following conditions: ' + ', '.join(
            sorted(readable_fact(f, self.formats) for f in self.goal_literals))
        return observation

    def goal_checklist(self):
        state = set(self.state.literals)
        return {'satisfied': sorted(readable_fact(f, self.formats) for f in self.goal_literals & state),
                'pending': sorted(readable_fact(f, self.formats) for f in self.goal_literals - state)}

    def render_state(self):
        facts = (self.public_model.state_view(set(self.state.literals), self.goal_literals)
                 if self.public_model else self.state.literals)
        checklist = self.goal_checklist()
        return ('Pending goal conditions: ' + '; '.join(checklist['pending'])
                + '\nSatisfied conditions: ' + '; '.join(checklist['satisfied'])
                + '\nCurrent public action resources: '
                + '; '.join(sorted(readable_fact(f, self.formats) for f in facts)))


class WholeGoalOnlyPDDL(DependencyGuardedPDDL):
    whole_goal = True

    def brief(self, menu):
        return '\nPublic all-goal relaxed cost estimates: ' + ' | '.join(
            f"{e['label']}: {self.action_signals[e['command']]['aggregate_distance_after']}; "
            f"undoes {sorted(canonical_fact(f) for f in self.action_signals[e['command']]['risks'])}"
            for e in menu)


class TransportOnlyPDDL(DependencyGuardedPDDL):
    resource_batch = True

    def brief(self, menu):
        note = super().brief(menu)
        roles = [f"{e['label']}: {self.action_signals[e['command']]['transport_role']}"
                 for e in menu if self.action_signals[e['command']]['transport_role']]
        return note + ('\nPublic transport action effects: ' + ' | '.join(roles) if roles else '')


class MeteredBabyAI(base.AdaptedBabyAI):
    def reset(self, task_id):
        self.navigation = PublicLocalMap()
        self.previous_command = None
        self.stage = 0
        self.rotations = 0
        self.longest_rotation_run = 0
        self.completed_unlocks = 0
        self.ever_faced_target = False
        return super().reset(task_id)

    def describe(self, observation):
        text = super().describe(observation)
        self.visible_tiles = self.navigation.observe(
            observation["image"], observation["direction"], self.previous_command, self.objects, self.colors)
        return text

    def step_command(self, command):
        self.previous_command = command
        position = ahead(self.navigation.pose, self.navigation.heading)
        before = self.visible_tiles.get(position)
        result = super().step_command(command)
        after = self.visible_tiles.get(ahead(self.navigation.pose, self.navigation.heading))
        self.rotations = self.rotations + 1 if command in {"left", "right"} else 0
        self.longest_rotation_run = max(self.longest_rotation_run, self.rotations)
        if command == "toggle" and before and after and before[0] == after[0] == "door" and before[2] == 2 and after[2] == 0:
            self.completed_unlocks += 1
        self.ever_faced_target |= ahead(self.navigation.pose, self.navigation.heading) in matching_targets(
            self.visible_tiles, self.task_goal)
        self.metrics.update(unlocks=self.completed_unlocks, longest_rotation_run=self.longest_rotation_run,
                            reached_public_target=self.ever_faced_target)
        return result


class MappedBabyAI(MeteredBabyAI):
    staged = False

    def reset(self, task_id):
        self.pending_door = None
        return super().reset(task_id)

    def step_command(self, command):
        stages = mission_stages(self.task_goal)
        if self.staged and self.stage < len(stages) - 1:
            directions = matching_targets(self.visible_tiles, stages[self.stage])
            was_facing = ahead(self.navigation.pose, self.navigation.heading) in directions
        else:
            was_facing = False
        result = super().step_command(command)
        if was_facing:
            stage = stages[self.stage]
            front = self.visible_tiles.get(ahead(self.navigation.pose, self.navigation.heading))
            held = self.holding()
            color_kind = re.search(r"(?:red|green|blue|purple|yellow|grey)?\s*(?:ball|key|box)", stage)
            expected = color_kind[0].strip() if color_kind else ''
            if ((stage.startswith("open ") and command == "toggle" and front and front[0] == "door" and front[2] == 0)
                    or (stage.startswith("pick up ") and command == "pickup" and held and held.endswith(expected))
                    or stage.startswith("go to ")):
                self.stage += 1
        return result

    def holding(self):
        image = self.visible_observation["image"]
        kind, color, _ = map(int, image[image.shape[0] // 2, -1])
        name = self.objects[kind]
        return None if name in {"empty", "floor", "unseen"} else f"{self.colors[color]} {name}"

    def public_action_candidates(self, atlas, working, goal, observation, history):
        entries = super().public_action_candidates(atlas, working, goal, observation, history)
        nav = self.navigation
        stages = mission_stages(goal)
        instruction = stages[min(self.stage, len(stages) - 1)] if self.staged and stages else goal
        targets = matching_targets(nav.tiles, instruction)
        path, reached = nav.next_steps(targets)
        front_at = ahead(nav.pose, nav.heading)
        front = nav.tiles.get(front_at)
        held = self.holding()
        if self.staged and self.pending_door:
            frame, door_position = self.pending_door
            door = nav.tiles.get(door_position)
            if frame != nav.frame or not door or door[0] != "door" or door[2] != 2:
                self.pending_door = None
        if self.staged and not self.pending_door and front and front[0] == "door" and front[2] == 2 and (reached or not targets):
            self.pending_door = (nav.frame, front_at)
        if self.staged and self.pending_door:
            door_position = self.pending_door[1]
            color = nav.tiles[door_position][1]
            if held != f"{color} key":
                instruction = f"find the {color} key to unlock the door"
                path, reached = nav.next_steps(matching_targets(nav.tiles, instruction))
            else:
                instruction = f"open the {color} door with the held key"
                path, reached = nav.next_steps({door_position})
        hint = path[0] if path else None
        if self.staged and reached and not path and front:
            if front[0] == "door" and ("open" in instruction or front[2] == 2):
                hint = "toggle"
            elif front[0] in {"key", "box", "ball"} and held is None and ("pick up" in instruction or "unlock" in instruction):
                hint = "pickup"
        self.route_hint = hint if any(x["command"] == hint for x in entries) else None
        self.route_note = f"Public navigation: {instruction}. Front: {front or 'unknown'}. Holding: {held or 'empty'}."
        if self.route_hint:
            for entry in entries:
                if entry["command"] == self.route_hint:
                    entry["reason"] = "public_route_next_step"
        return entries

    def brief(self, menu):
        label = next((row["label"] for row in menu if row["command"] == self.route_hint), None)
        return "\n" + self.route_note + (f" Suggested next move: {label}." if label else "")


class StagedBabyAI(MappedBabyAI):
    staged = True


class GoalDirectedBabyAI(MappedBabyAI):
    """Route using the egocentric map, and commit stages from observed outcomes."""

    def reset(self, task_id):
        observation = super().reset(task_id)
        self.mission = ordered_mission(self.task_goal)
        self.confirmed_deliveries = self.matching_keys_picked = 0
        self.metrics.update(mission_stages_completed=0, confirmed_deliveries=0, matching_keys_picked=0)
        return observation

    def current_stage(self):
        return self.mission[self.stage] if self.stage < len(self.mission) else None

    def step_command(self, command):
        stage = self.current_stage()
        nav = self.navigation
        frame, front_at = nav.frame, ahead(nav.pose, nav.heading)
        front = self.visible_tiles.get(front_at)
        held = self.holding()
        door = nav.tiles.get(self.pending_door[1]) if self.pending_door else None
        result = super().step_command(command)
        after = self.visible_tiles.get(front_at) if frame == nav.frame else None
        current_held = self.holding()
        if (command == "pickup" and door and not held and current_held == f"{door[1]} key"
                and front and front[:2] == ("key", door[1])):
            self.matching_keys_picked += 1
        if stage:
            action, obj, target = stage
            faced = ahead(nav.pose, nav.heading) in matching_targets(self.visible_tiles, obj)
            was_facing = front and front_at in matching_targets({front_at: front}, obj)
            delivered = (action == "deliver" and command == "drop" and not current_held and matches_item(held, obj)
                         and after and after[:2] == tuple(reversed(held.split(" ", 1)))
                         and any(ahead(front_at, i) in matching_targets(self.visible_tiles, target) for i in range(4)))
            if ((action == "face" and faced)
                or (action == "pickup" and command == "pickup" and was_facing and not held and matches_item(current_held, obj))
                or (action == "toggle" and command == "toggle" and was_facing and front[0] == "door"
                    and after and after[0] == "door" and after[2] == 0)
                or delivered):
                self.stage += 1
                self.confirmed_deliveries += int(bool(delivered))
        self.metrics.update(mission_stages_completed=self.stage, confirmed_deliveries=self.confirmed_deliveries,
                            matching_keys_picked=self.matching_keys_picked)
        return result

    def public_action_candidates(self, atlas, working, goal, observation, history):
        stage = self.current_stage()
        if not stage:
            return super().public_action_candidates(atlas, working, goal, observation, history)
        entries = base.AdaptedBabyAI.public_action_candidates(self, atlas, working, goal, observation, history)
        nav, held = self.navigation, self.holding()
        action, obj, target = stage
        front_at = ahead(nav.pose, nav.heading)
        front = self.visible_tiles.get(front_at)
        targets = matching_targets(nav.tiles, obj)
        role = action
        instruction = f"{action} {obj}"
        if action == "deliver" and matches_item(held, obj):
            role = "drop"
            targets = delivery_tiles(nav.tiles, matching_targets(nav.tiles, target))
            instruction = f"place {held} next to {target} on an observed empty tile"
        elif action == "deliver":
            role, instruction = "pickup", f"pick up {obj} for delivery next to {target}"
        path, reached = nav.next_steps(targets)
        if self.pending_door:
            frame, pos = self.pending_door
            door = nav.tiles.get(pos)
            if frame != nav.frame or not door or door[0] != "door" or door[2] != 2:
                self.pending_door = None
        closed = None
        if not reached:
            closed = nav.nearest_reachable({pos for pos, tile in nav.tiles.items()
                                            if tile[0] == "door" and tile[2] == 1})
        if not self.pending_door and (not reached or role == "toggle"):
            doors = {pos for pos, tile in nav.tiles.items() if tile[0] == "door" and tile[2] == 2}
            if role == "toggle" and targets:
                doors &= targets
            elif closed is not None:
                doors.clear()
            else:
                doors = {pos for pos in doors if pos == front_at or matches_item(held, f"{nav.tiles[pos][1]} key")
                         or nav.next_steps(matching_targets(nav.tiles, f"{nav.tiles[pos][1]} key"))[1]}
            locked = nav.nearest_reachable(doors)
            if locked is not None:
                self.pending_door = (nav.frame, locked)
        if self.pending_door:
            pos = self.pending_door[1]
            key = f"{nav.tiles[pos][1]} key"
            if held == key:
                instruction, role, targets = f"return to the locked door with the {key} and open it", "toggle", {pos}
            else:
                instruction, role, targets = f"find and pick up the {key} for the locked door", "pickup", matching_targets(nav.tiles, key)
            path, reached = nav.next_steps(targets)
        if not reached and closed is not None:
            path, reached = nav.next_steps({closed})
            instruction += f"; open the known unlocked {nav.tiles[closed][1]} door to explore further"
            role = "toggle"
        needs_free_hand = role == "pickup" and held
        if needs_free_hand:
            free = {pos for pos, tile in nav.tiles.items() if tile[0] in {"empty", "floor"}}
            path, reached = nav.next_steps(free)
            instruction += f"; place the held {held} on a confirmed empty tile first"
            role = "drop"
        hint = path[0] if path else None
        if reached and not path and front and role not in {"face", "deliver"}:
            if role != "drop" or front[0] in {"empty", "floor"}:
                hint = role
        self.route_hint = hint if any(row["command"] == hint for row in entries) else None
        if self.route_hint:
            next(row for row in entries if row["command"] == self.route_hint)["reason"] = "public_mission_prerequisite"
        self.route_note = f"Public mission stage: {instruction}. Front: {front or 'unknown'}. Holding: {held or 'empty'}."
        return entries


class Runner(base.AdaptationRunner):
    def _actor_messages(self, atlas, working, goal, observation, history):
        messages, view_hash, view_id = super()._actor_messages(atlas, working, goal, observation, history)
        if hasattr(self.env, "brief") and self.action_menu:
            note = self.env.brief(self.action_menu)
            previous = messages[-1]["content"]
            messages[-1]["content"] += note
            self.turn_audits[-1]['planning_card_in_view'] = True
            if len(self.policy.encode(messages)) > self.cfg.get("actor_max_input", 6144):
                messages[-1]["content"] = previous
                self.turn_audits[-1]['planning_card_in_view'] = False
            if isinstance(self.env, DependencyRankedPDDL):
                self.turn_audits[-1]['planning_signals'] = {
                    c: {k: (sorted(map(str, v)) if isinstance(v, (set, frozenset)) else v)
                        for k, v in row.items()} for c, row in self.env.action_signals.items()}
        return messages, base.digest(messages), view_id


def create_env(root, domain, arm, specs, config):
    if domain == "pddl":
        cls = {"adapted_interface": MeteredPDDL, "goal_dependencies": PlannedPDDL,
               "goal_dependencies_inverse": LoopAwarePDDL, "compact_state_only": CompactStatePDDL,
               "reordered_only": ReorderedPDDL, "compact_reordered": CompactReorderedPDDL,
               "compact_reordered_exact_loop": ExactLoopPDDL, "dependency_state": DependencyStatePDDL,
               "grounded_dependencies": DependencyRankedPDDL, "grounded_dependencies_loop": DependencyLoopPDDL,
               "grounded_dependencies_guard": DependencyGuardedPDDL,
               "canonical_state_guard": CanonicalStatePDDL, "whole_goal_guard": WholeGoalPDDL,
               "resource_goal_guard": ResourceGoalPDDL, "readable_state_only": ReadableStatePDDL,
               "whole_goal_only": WholeGoalOnlyPDDL, "transport_only": TransportOnlyPDDL}[arm]
        return cls(root, specs), None
    cls = {"adapted_interface": MeteredBabyAI, "local_map": MappedBabyAI, "local_map_stages": StagedBabyAI,
           "public_mission_stages": GoalDirectedBabyAI}[arm]
    return cls(root, specs), None


def run(root, domain):
    with patch.object(base, "create_env", create_env), patch.object(base, "AdaptationRunner", Runner):
        base.run(root, [domain])
    report = base.shared.read(root / "summary.json")
    cells = report["domains"][domain]["cells"]
    for arm, cell in cells.items():
        rows = read_jsonl(root / domain / arm / "results.jsonl")
        if domain == "pddl":
            cell["mean_final_coverage"] = sum(r["metrics"]["final_coverage"] for r in rows) / len(rows)
        else:
            cell.update(public_target_reached=sum(r["metrics"]["reached_public_target"] for r in rows),
                        confirmed_unlocks=sum(r["metrics"]["unlocks"] for r in rows),
                        mean_longest_rotation_run=sum(r["metrics"]["longest_rotation_run"] for r in rows) / len(rows),
                        mean_environment_steps=sum(sum(e["phase"] == "execute" and e.get("sent") is True
                                                       for e in r["history"]) for r in rows) / len(rows))
            if arm == "public_mission_stages":
                cell.update(mission_stages_completed=sum(r["metrics"]["mission_stages_completed"] for r in rows),
                            matching_keys_picked=sum(r["metrics"]["matching_keys_picked"] for r in rows),
                            confirmed_deliveries=sum(r["metrics"]["confirmed_deliveries"] for r in rows))
    atomic_json(root / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--domain", choices=("pddl", "babyai"), required=True)
    args = parser.parse_args()
    with (args.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve(), args.domain)
        except Exception as exc:
            atomic_json(args.root / "error.json", {"error": str(exc), "traceback": base.traceback.format_exc()})
            atomic_json(args.root / "status.json", {"phase": "needs_attention"})
            raise
