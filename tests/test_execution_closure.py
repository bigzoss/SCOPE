from scope.pddl_public_model import canonical_fact, PublicSTRIPSModel, GroundOperator, transport_batch_opportunity, transport_role
from scope.scienceworld_execution import public_aliases, wait_rejection, step_diagnostic
from scope.scienceworld_actions import science_command_guard, public_focus_stage
from test_public_dependency_stages import fact


def test_fact_render_preserves_order_arity_and_negation():
    assert canonical_fact(fact('holding', 'left', 'shot1')) == 'holding(left, shot1)'
    assert canonical_fact(fact('carry', 'ball6', 'right')) == 'carry(ball6, right)'
    assert canonical_fact(fact('handempty')) == 'handempty()'


def test_multi_cargo_and_goal_restoration_are_visible_in_ranking():
    m = PublicSTRIPSModel({}, [])
    a, b, free, cargo1, cargo2, ball2, goal1, goal2 = [fact(x) for x in
        ('a', 'b', 'free', 'cargo1', 'cargo2', 'ball2', 'goal1', 'goal2')]
    def op(pre, add, delete):
        return GroundOperator(frozenset(pre), frozenset(), frozenset(add), frozenset(delete))
    m.actions = {'load2': op({a, free, ball2}, {cargo2}, {free, ball2}),
                 'out': op({a}, {b}, {a}), 'back': op({b}, {a}, {b}),
                 'drop1': op({b, cargo1}, {goal1}, {cargo1}),
                 'drop2': op({b, cargo2}, {goal2, free}, {cargo2}),
                 'undo1': op({b, goal1}, {cargo1}, {goal1})}
    state = frozenset({a, free, cargo1, ball2})
    goals = {goal1, goal2}
    rows = m.rank({'load2': 'load2', 'out': 'out'}, state, goals, [], [], whole_goal=True)
    assert rows[0]['command'] == 'load2'
    finished1 = frozenset({b, goal1, cargo2})
    rows = m.rank({'drop2': 'drop2', 'undo1': 'undo1'}, finished1, goals, [], [], whole_goal=True)
    assert rows[0]['command'] == 'drop2'
    assert rows[1]['risks'] == {goal1} and rows[1]['aggregate_distance_delta'] < 0
    # Undo remains available when it is the only legal way out.
    assert m.rank({'undo1': 'undo1'}, finished1, goals, [], [], whole_goal=True)


def test_alias_requires_current_scoped_public_observation_and_citation():
    history = [{'command': 'look at unknown substance', 'observation': 'unknown substance B'}]
    assert public_aliases(history) == {'unknown substance': {'target': 'unknown substance b', 'turn': 0}}
    goal = 'First, focus on the unknown substance B. Next, focus on the thermometer.'
    assert science_command_guard('focus on unknown substance', goal, {}, [], '', resolve_aliases=True)
    assert science_command_guard('focus on unknown substance', goal, {}, history, '', resolve_aliases=True) is None
    assert science_command_guard('focus on unknown substance', goal, {}, history, '') is not None  # archived protocol
    history.append({'command': 'focus on unknown substance', 'observation': 'You focus on the unknown substance B.'})
    assert public_focus_stage(goal, history, {}, resolve_aliases=True)[0] == 'thermometer'
    history.append({'command': 'go to kitchen', 'observation': 'This room is called the kitchen.'})
    assert public_aliases(history) == {}


def test_alias_rejects_ambiguity_failed_inspection_and_stale_reinspection():
    for response in ('unknown substance B or C', 'unknown substance B\nunknown substance C',
                     'unknown substance B is red', 'Nothing happens.'):
        assert not public_aliases([{'command': 'look at unknown substance', 'observation': response}])
    history = [{'command': 'look at unknown substance', 'observation': 'unknown substance B'},
               {'command': 'look at unknown substance', 'observation': 'Nothing happens.', 'is_error': True}]
    assert not public_aliases(history)


def test_wait_budget_and_native_completion_are_distinct():
    assert wait_rejection('wait', 21, 30)
    assert wait_rejection('wait1', 29, 30)
    assert wait_rejection('wait1', 28, 30) is None
    assert wait_rejection('look around', 29, 30) is None
    truncated = step_diagnostic('wait', 21, 32, True, False, 30)
    assert truncated['native_budget_truncated'] and truncated['native_move_delta'] == 11
    assert truncated['termination_reason'] == 'native_environment_step_budget'
    completed = step_diagnostic('focus on green box', 29, 30, True, True, 30)
    assert not completed['native_budget_truncated'] and completed['termination_reason'] == 'environment_done'


def test_batch_hint_needs_free_hand_and_same_destination_not_completed_ball():
    state = {fact('gripper', 'left'), fact('gripper', 'right'), fact('free', 'right'),
             fact('carry', 'ball1', 'left'), fact('at', 'ball2', 'a')}
    goals = {fact('at', 'ball1', 'b'), fact('at', 'ball2', 'b')}
    op = GroundOperator(frozenset(), frozenset(), frozenset({fact('carry', 'ball2', 'right')}), frozenset())
    assert transport_batch_opportunity(op, state, goals)
    assert not transport_batch_opportunity(op, state - {fact('free', 'right')}, goals)
    assert not transport_batch_opportunity(op, state, {fact('at', 'ball1', 'c'), fact('at', 'ball2', 'b')})
    assert not transport_batch_opportunity(op, state | {fact('at', 'ball2', 'b')}, goals)


def test_loaded_trip_does_not_prefer_unloading_at_source():
    state = {fact('gripper', 'left'), fact('gripper', 'right'), fact('at-robby', 'a'),
             fact('carry', 'ball1', 'left'), fact('carry', 'ball2', 'right')}
    goals = {fact('at', 'ball1', 'b'), fact('at', 'ball2', 'b')}
    def op(add):
        return GroundOperator(frozenset(), frozenset(), frozenset(add), frozenset())
    assert transport_role(op({fact('at-robby', 'b')}), state, goals) == 'travel_loaded'
    assert transport_role(op({fact('at', 'ball1', 'a')}), state, goals) == 'unload_off_goal'
    assert transport_role(op({fact('at', 'ball1', 'b')}), state, goals) == 'deliver_to_goal'
    assert transport_role(op({fact('at-robby', 'b')}), {fact('holding', 'left', 'shot1')}, goals) is None


def test_alias_invalidated_when_object_scope_changes():
    history = [{'command': 'look at unknown substance', 'observation': 'unknown substance B'},
               {'command': 'move unknown substance to green box', 'observation': 'You move the unknown substance B to the green box.'}]
    assert not public_aliases(history)
