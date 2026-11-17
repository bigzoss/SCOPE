from test_public_dependency_stages import fact
from scope.pddl_public_model import readable_fact
from scope.scienceworld_stages import select_continuous_experiment_menu
from scope.scienceworld_stages import select_progress_experiment_menu


def test_readable_facts_preserve_resources_and_existing_vocabulary():
    formats = {'holding': 'You hold {}.', 'carry': '{} carries {}.', 'at': '{} is at {}.'}
    assert readable_fact(fact('holding', 'left', 'shot1'), formats) == 'Hand left is holding shot1.'
    assert readable_fact(fact('carry', 'ball6', 'right'), formats) == 'Gripper right is carrying ball6.'
    assert readable_fact(fact('at', 'ball6', 'rooma'), formats) == 'ball6 is at rooma.'
    assert readable_fact(fact('custom', 'a', 'b'), {'custom': '{}'}) == 'custom(a, b)'


def thermal_fixture():
    trial = {'subject': 'sci:unknown substance y', 'samples': [{'phase': 'solid', 'turn': 0}], 'interval_c': None}
    history = [{'command': 'move solid unknown substance y to stove',
                'observation': 'You move the solid unknown substance Y to the stove.'},
               {'command': 'activate stove', 'observation': 'The stove is now activated.'}]
    return trial, history


def test_heating_keeps_wait_and_measurement_after_repetition():
    trial, history = thermal_fixture()
    measure = 'use thermometer on solid unknown substance y'
    history += [{'command': measure, 'observation': 'the thermometer measures a temperature of 35 degrees celsius'}] * 8
    commands = [measure, 'wait1', 'wait', 'look at solid unknown substance y', 'look around',
                'move solid unknown substance y to kitchen', 'disconnect solid unknown substance y', 'go to bathroom']
    selected, audit = select_continuous_experiment_menu(commands, 'Measure melting point.', history, trial, None, [])
    assert measure in selected and 'wait1' in selected and 'wait' not in selected
    assert 'move solid unknown substance y to kitchen' not in selected
    assert audit['experiment_stage']['powered'] is True
    assert 'look around' in selected


def test_failed_activation_and_missing_device_do_not_imply_heating():
    trial, history = thermal_fixture()
    history[-1]['observation'] = 'Nothing happens.'
    selected, audit = select_continuous_experiment_menu(['activate stove', 'wait1', 'look around'],
        '', history, trial, None, [])
    assert 'activate stove' in selected and 'wait1' not in selected
    selected, _ = select_continuous_experiment_menu(['go to kitchen', 'open door to kitchen', 'look around'],
        '', [], trial, None, [])
    assert 'go to kitchen' in selected


def test_confirmed_submission_still_available():
    trial, history = thermal_fixture()
    selected, _ = select_continuous_experiment_menu(['focus on green box', 'wait1', 'look around'],
        '', history, trial, None, ['focus on green box'])
    assert 'focus on green box' in selected


def test_identical_inspection_deferred_only_until_preparation_attempt():
    trial, _ = thermal_fixture()
    measure = 'use thermometer on solid unknown substance y'
    move = 'move solid unknown substance y to stove'
    row = {'command': measure, 'observation': 'Temperature is 20 degrees celsius.'}
    commands = [measure, move, 'look around']
    selected, audit = select_progress_experiment_menu(commands, '', [row, row], trial, None, [])
    assert measure not in selected and move in selected
    assert audit['no_information_deferred'][0]['count'] == 2
    selected, _ = select_progress_experiment_menu(commands, '', [row, row,
        {'command': move, 'observation': 'Nothing happens.'}], trial, None, [])
    assert measure in selected


def test_changed_reading_missing_preparation_and_heating_remain_observable():
    trial, powered = thermal_fixture()
    measure = 'use thermometer on solid unknown substance y'
    a = {'command': measure, 'observation': 'Temperature is 20 degrees celsius.'}
    b = {'command': measure, 'observation': 'Temperature is 22 degrees celsius.'}
    commands = [measure, 'move solid unknown substance y to stove', 'look around', 'wait1']
    selected, _ = select_progress_experiment_menu(commands, '', [a, b], trial, None, [])
    assert measure in selected
    selected, _ = select_progress_experiment_menu([measure, 'look around'], '', [a, a], trial, None, [])
    assert measure in selected
    selected, _ = select_progress_experiment_menu(commands, '', powered + [a, a], trial, None, [])
    assert measure in selected and 'wait1' in selected
