"""Public STRIPS dependency model; heuristic guidance, never an environment reward.

Ground actions must come from the environment's public domain/problem grounding.
Negative preconditions are checked for real successor predictions and relaxed only
in the explicitly labelled reachability heuristic. No solution files are read.
"""
from collections import Counter
from dataclasses import dataclass
from math import inf
from string import Formatter

from .pddl_subgoals import action_effects, ground, literals


def canonical_fact(literal):
    """Lossless public state serialization; never use this to rewrite commands."""
    args = [getattr(value, 'name', str(value)) for value in literal.variables]
    text = f"{literal.predicate.name}({', '.join(args)})"
    return ('not ' if getattr(literal, 'is_negative', False) else '') + text


def readable_fact(literal, formats):
    """Repair resource facts while retaining the established public vocabulary."""
    args = [getattr(value, 'name', str(value)) for value in literal.variables]
    name = literal.predicate.name
    if name == 'carry' and len(args) == 2:
        return f'Gripper {args[1]} is carrying {args[0]}.'
    if name == 'holding' and len(args) == 2:
        return f'Hand {args[0]} is holding {args[1]}.'
    template = formats.get(name)
    if template:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
        if fields == [''] * len(args):
            return template.format(*args)
    return canonical_fact(literal)


def transport_batch_opportunity(op, state, goals):
    """Gripper public-schema hint: load another undelivered ball for the same trip.

    Activate only with explicit gripper/free/carry/at predicates. This is a
    domain adapter heuristic, not a learned planner or a reward signal.
    """
    def slots(facts, name, arity):
        return {tuple(getattr(v, 'name', str(v)) for v in f.variables)
                for f in facts if f.predicate.name == name and len(f.variables) == arity}
    hands = slots(state, 'gripper', 1)
    free = slots(state, 'free', 1)
    carried = slots(state, 'carry', 2)
    locations, desired = slots(state, 'at', 2), slots(goals, 'at', 2)
    for obj, hand in slots(op.adds, 'carry', 2):
        if (hand,) not in hands & free:
            continue
        destinations = {place for item, place in desired - locations if item == obj}
        if any((other, place) in desired and other != obj and (other_hand,) in hands
               for other, other_hand in carried for place in destinations):
            return True
    return False


def transport_role(op, state, goals):
    """Public gripper schema only; expose complete load/travel/deliver guidance."""
    def tuples(facts, name):
        return {tuple(getattr(v, 'name', str(v)) for v in f.variables)
                for f in facts if f.predicate.name == name}
    if not tuples(state, 'gripper'):
        return None
    carried = {obj for obj, hand in tuples(state, 'carry')}
    desired = tuples(goals, 'at')
    adds_at, deletes_at = tuples(op.adds, 'at'), tuples(op.deletes, 'at')
    if any(obj in carried and (obj, place) in desired for obj, place in adds_at):
        return 'deliver_to_goal'
    if deletes_at & desired & tuples(state, 'at'):
        return 'undo_delivery'
    if transport_batch_opportunity(op, state, goals):
        return 'batch_load'
    destinations = {place for obj, place in desired if obj in carried}
    if any(place in destinations for place, in tuples(op.adds, 'at-robby')):
        return 'travel_loaded'
    if any(obj in carried for obj, place in adds_at):
        return 'unload_off_goal'
    return None


def guard_repeated_transitions(rows):
    """Keep a legal escape if every action repeats, otherwise require a new edge."""
    return [row for row in rows if not row['repeated_inverse']] or rows


@dataclass(frozen=True)
class GroundOperator:
    positive: frozenset
    negative: frozenset
    adds: frozenset
    deletes: frozenset

    def applicable(self, state):
        return self.positive <= state and not self.negative & state

    def successor(self, state):
        if not self.applicable(state):
            raise ValueError("Public action preconditions are not satisfied")
        return frozenset((state - self.deletes) | self.adds)


class PublicSTRIPSModel:
    def __init__(self, operators, actions):
        self.actions = {}
        for action in sorted(actions, key=str):
            operator = operators[action.predicate.name.lower()]
            binding = dict(zip(operator.params, action.variables))
            positive, negative = set(), set()
            for lifted in literals(operator.preconds):
                if not hasattr(lifted, "predicate"):
                    raise ValueError("Only conjunctive STRIPS preconditions are supported")
                item = ground(lifted, binding)
                (negative if item.is_negative else positive).add(item.positive if item.is_negative else item)
            adds, deletes = action_effects(operators, action)
            self.actions[action] = GroundOperator(*(frozenset(x) for x in (positive, negative, adds, deletes)))
        self.dynamic_predicates = {x.predicate for op in self.actions.values() for x in op.adds | op.deletes}

    def state_view(self, state, goals):
        # Preserve every dynamic fact. Static facts can be omitted from the text
        # only when no action precondition or goal refers to them.
        relevant = {g.predicate for g in goals} | self.dynamic_predicates
        relevant |= {p.predicate for op in self.actions.values() for p in op.positive | op.negative}
        return {fact for fact in state if fact.predicate in relevant}

    def relaxed_costs(self, state):
        costs = {fact: 0 for fact in state}
        changed = True
        while changed:
            changed = False
            for op in self.actions.values():
                if not all(p in costs for p in op.positive):
                    continue
                cost = 1 + sum(costs[p] for p in op.positive)
                for fact in op.adds:
                    if cost < costs.get(fact, inf):
                        costs[fact] = cost
                        changed = True
        return costs

    def rank(self, available, state, goals, recent_states, recent_edges, suppress_loops=True,
             whole_goal=False, resource_batch=False):
        state, goals = frozenset(state), frozenset(goals)
        pending = goals - state
        costs = self.relaxed_costs(state)
        before = min((costs.get(g, inf) for g in pending), default=0)
        # Include satisfied goals at cost zero: undoing one then has a visible
        # restoration cost. Sum is a heuristic, NOT a joint-plan length/reward.
        total_before = sum(costs.get(g, inf) for g in goals)
        visited, used = Counter(recent_states[-16:]), Counter(recent_edges[-24:])
        rows = []
        for command, action in available.items():
            op = self.actions[action]
            following = op.successor(state)
            after_costs = self.relaxed_costs(following)
            after = min((after_costs.get(g, inf) for g in pending), default=0)
            delta = before - after if before < inf and after < inf else 0
            advances = pending & following
            risks = goals & state - following
            total_after = sum(after_costs.get(g, inf) for g in goals)
            total_delta = total_before - total_after if total_before < inf and total_after < inf else 0
            repeated = following != state and visited[following] > 0 and used[state, command] >= 1
            # A static dependency label cannot exempt a return to an identical
            # state. New cargo or changed resources produce a different state.
            # Re-achieving a goal after undoing it is not novel progress either.
            # Penalize rather than remove: necessary temporary reversals stay legal.
            penalty = 30 if suppress_loops and repeated else 0
            transport = transport_role(op, state, goals) if resource_batch else None
            batching = transport == 'batch_load'
            score = 10 * len(advances) - 4 * len(risks) + 4 * (total_delta if whole_goal else delta)
            if transport:
                # The relaxed all-goal sum misprices shared hand restoration:
                # even dropping loaded cargo at its source can appear cheaper.
                # Use the complete public transport phase, not a pickup bonus.
                score = {'deliver_to_goal': 50, 'batch_load': 40, 'travel_loaded': 30,
                         'undo_delivery': -20, 'unload_off_goal': -20}[transport]
            rows.append({"command": command, "advances": advances, "risks": risks,
                         "distance_before": before if before < inf else None,
                         "distance_after": after if after < inf else None,
                         "distance_delta": delta, "repeated_inverse": repeated,
                         "aggregate_distance_before": total_before if total_before < inf else None,
                         "aggregate_distance_after": total_after if total_after < inf else None,
                         "aggregate_distance_delta": total_delta,
                         "batch_load_opportunity": batching, "transport_role": transport,
                         "score": score - penalty})
        return sorted(rows, key=lambda row: (-row["score"], row["command"]))
