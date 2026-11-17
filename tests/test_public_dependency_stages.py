from dataclasses import dataclass

from scope.pddl_public_model import PublicSTRIPSModel, guard_repeated_transitions
from scope.scienceworld_stages import experiment_stage, select_experiment_menu


@dataclass(frozen=True)
class Predicate:
    name: str

    def __call__(self, *args):
        return Literal(self, args)


@dataclass(frozen=True)
class Literal:
    predicate: Predicate
    variables: tuple
    is_negative: bool = False
    is_anti: bool = False


def fact(name, *args):
    return Predicate(name)(*args)


def model():
    # Already grounded independent test domain; actions below model load/travel/deliver.
    from scope.pddl_public_model import GroundOperator
    m = PublicSTRIPSModel({}, [])
    at_a, at_b, free, cargo, ball_a, goal = [fact(x) for x in ('at_a', 'at_b', 'free', 'cargo', 'ball_a', 'goal')]
    def op(pre, add, delete):
        return GroundOperator(frozenset(pre), frozenset(), frozenset(add), frozenset(delete))
    m.actions = {'load': op({at_a, free, ball_a}, {cargo}, {free, ball_a}),
                 'out': op({at_a}, {at_b}, {at_a}), 'back': op({at_b}, {at_a}, {at_b}),
                 'deliver': op({at_b, cargo}, {goal, free}, {cargo})}
    m.dynamic_predicates = {f.predicate for o in m.actions.values() for f in o.adds | o.deletes}
    return m, frozenset({at_a, free, ball_a}), frozenset({goal})


def test_joint_dependencies_prefer_loading_to_empty_travel():
    m, initial, goals = model()
    rows = m.rank({'load': 'load', 'out': 'out'}, initial, goals, [], [])
    assert rows[0]['command'] == 'load'
    carried = m.actions['load'].successor(initial)
    assert m.rank({'out': 'out'}, carried, goals, [], [])[0]['distance_delta'] > 0


def test_state_projection_preserves_resources_and_repeated_transport_is_not_a_loop():
    m, initial, goals = model()
    assert m.state_view(initial, goals) == initial
    away = m.actions['out'].successor(initial)
    returning = m.rank({'back': 'back'}, away, goals, [initial], [(away, 'back')])[0]
    assert returning['repeated_inverse'] and returning['score'] < 0
    carried = m.actions['load'].successor(initial)
    row = m.rank({'out': 'out'}, carried, goals, [away], [(initial, 'out')])[0]
    assert not row['repeated_inverse']


def test_navigation_and_door_opening_have_separate_menu_slots():
    commands = ['go to hallway', 'open door to hallway', 'open drawer', 'look at blue box']
    commands += [f'move blue box to object {i}' for i in range(20)]
    selected, audit = select_experiment_menu(commands, 'First, focus on the thermometer. Then use the blue box.',
                                             [], None, 'thermometer', [])
    assert 'go to hallway' in selected and 'open door to hallway' in selected
    assert audit['eligible_movement'] == 1


def test_sample_placement_does_not_confirm_heater_power():
    trial = {'subject': 'sci:unknown substance y', 'samples': [{'phase': 'solid', 'turn': 0}], 'interval_c': None}
    history = [{'command': 'move solid unknown substance y to stove',
                'observation': 'You move the solid unknown substance Y to the stove.'},
               {'command': 'activate stove', 'observation': 'Nothing happens.'}]
    stage = experiment_stage('', history, trial, None, [])
    assert stage['phase'] == 'activate_device' and stage['powered'] is None
    selected, audit = select_experiment_menu(['activate stove', 'look at stove', 'go to hallway'],
                                             'Measure the melting point.', history, trial, None, [])
    assert 'activate stove' in audit['offered_preparation']
    history[-1]['observation'] = 'The stove is now activated.'
    assert experiment_stage('', history, trial, None, [])['phase'] == 'observe_transition'
    history.append({'command': 'move solid unknown substance y to blue box',
                    'observation': 'You move the solid unknown substance Y to the blue box.'})
    assert experiment_stage('', history, trial, None, [])['phase'] == 'place_sample'


def test_thermal_goal_does_not_replace_the_missing_instrument_target():
    trial = {'subject': 'sci:unknown substance y', 'samples': [], 'interval_c': None}
    stage = experiment_stage('First focus on the thermometer.', [], trial, 'thermometer', [])
    assert stage['target'] == 'thermometer' and stage['phase'] == 'find_required_object'


def test_reachieving_a_goal_in_the_same_repeated_state_is_penalized():
    m, initial, goals = model()
    carried = m.actions['load'].successor(initial)
    away = m.actions['out'].successor(carried)
    delivered = m.actions['deliver'].successor(away)
    fresh = m.rank({'deliver': 'deliver'}, away, goals, [], [])[0]
    repeated = m.rank({'deliver': 'deliver'}, away, goals, [delivered], [(away, 'deliver')])[0]
    assert repeated['score'] < fresh['score'] and repeated['repeated_inverse']
    assert guard_repeated_transitions([repeated, fresh]) == [fresh]
    assert guard_repeated_transitions([repeated]) == [repeated]
