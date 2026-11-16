"""Rank actions by public STRIPS effects and the prerequisites of pending goals."""
from collections import Counter


def literals(formula):
    return getattr(formula, "literals", (formula,))


def ground(formula, assignment):
    return formula.predicate(*(assignment.get(arg, arg) for arg in formula.variables))


def public_blockers(operators, goals, state):
    """Find absent fully bound positive prerequisites of a public goal achiever."""
    blockers = set()
    for operator in operators.values():
        for effect in literals(operator.effects):
            if not hasattr(effect, "predicate") or effect.is_anti:
                continue
            for goal in goals:
                if goal.predicate != effect.predicate:
                    continue
                bound = dict(zip(effect.variables, goal.variables))
                if ground(effect, bound) != goal:
                    continue
                for pre in literals(operator.preconds):
                    if (hasattr(pre, "predicate") and not pre.is_negative and not pre.is_anti
                            and all(arg in bound for arg in pre.variables)):
                        item = ground(pre, bound)
                        if item not in state:
                            blockers.add(item)
    return blockers


def partially_bound_prerequisites(operators, goals):
    patterns = set()
    for operator in operators.values():
        for effect in literals(operator.effects):
            if not hasattr(effect, "predicate") or effect.is_anti:
                continue
            for goal in goals:
                if goal.predicate != effect.predicate:
                    continue
                binding = dict(zip(effect.variables, goal.variables))
                for pre in literals(operator.preconds):
                    if not hasattr(pre, "predicate") or pre.is_anti or pre.is_negative:
                        continue
                    fixed = tuple((index, binding[arg]) for index, arg in enumerate(pre.variables) if arg in binding)
                    if 0 < len(fixed) < len(pre.variables):
                        patterns.add((pre.predicate, fixed))
    return patterns


def action_effects(operators, action):
    operator = operators[action.predicate.name.lower()]
    if len(operator.params) != len(action.variables):
        raise ValueError("Ground action arity differs from its public operator")
    assignment = dict(zip(operator.params, action.variables))
    adds, deletes = set(), set()
    for effect in literals(operator.effects):
        if not hasattr(effect, "predicate"):
            continue
        value = ground(effect, assignment)
        if value.is_anti:
            deletes.add(value.inverted_anti)
        else:
            adds.add(value)
    return adds, deletes


def rank_public_actions(operators, available, state, goals, history, suppress_inverse=False, recent_edges=(),
                        *, strict_inverse=False):
    pending = goals - state
    blockers = public_blockers(operators, pending, state)
    patterns = partially_bound_prerequisites(operators, pending) if suppress_inverse and not strict_inverse else ()
    recent = history[-8:]
    previous_states = Counter(recent)
    ranked = []
    current = frozenset(state)
    used = Counter(recent_edges[-12:])
    for command, action in available.items():
        adds, deletes = action_effects(operators, action)
        next_state = frozenset((state - deletes) | adds)
        advance = (adds - state) & pending
        unblocks = (adds - state) & blockers
        unblocks |= {added for added in adds - state if any(
            added.predicate == predicate and all(added.variables[index] == value for index, value in fixed)
            for predicate, fixed in patterns)}
        risks = goals & state & deletes - adds
        inverse = next_state in previous_states and next_state != state
        repeated_inverse = inverse and used[current, command] >= (2 if strict_inverse else 1)
        suppress = repeated_inverse and not advance and not unblocks if strict_inverse else inverse
        repeat_penalty = (12 + 3 * previous_states[next_state]) if suppress_inverse and suppress else 0
        score = 10 * len(advance) + 6 * len(unblocks) - 4 * len(risks) - repeat_penalty
        ranked.append({"command": command, "advances": advance, "unblocks": unblocks,
                       "risks": risks, "returns_to_recent_state": inverse,
                       "repeated_inverse": repeated_inverse, "score": score})
    ranked.sort(key=lambda row: (-row["score"], row["command"]))
    return ranked, blockers


def relevant_state_facts(state, goals):
    """Keep goal-linked facts and the full public block/hand support chain."""
    block_predicates = {"on", "ontable", "on-table", "clear", "holding", "handempty", "hand-empty"}
    entities = {arg for item in goals - state for arg in item.variables}
    related = set()
    for item in state:
        name = item.predicate.name.lower()
        if name in block_predicates or not item.variables or entities.intersection(item.variables):
            related.add(item)
    return related


def avoid_repeated_inverse(rows):
    """Retain legal fallback moves, including a reversal that advances a public prerequisite."""
    filtered = [row for row in rows if not row["repeated_inverse"] or row["advances"] or row["unblocks"]]
    return filtered or rows
