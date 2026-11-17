"""Native development integration checks. No LLM, teacher, or training calls.

The heuristic-only PDDL rollouts validate public transition guidance, not model
success rates. ScienceWorld replays a dev public prefix to test object binding.
"""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path('/home/hyx/lm_workspace/SCOPE')
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/teacher_context_grpo'),
               str(ROOT / 'runs/hotpot_four_bench_20261110.3j9gw8h4/native_deps')]
os.environ['PATH'] = '/home/hyx/lm_workspace/longagent/.java/bin:' + os.environ['PATH']

import planning_navigation_eval as planning
import target_action_adaptation as base
from scope.io import atomic_json, file_sha
from scope.scienceworld_execution import public_aliases, step_diagnostic


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    generated = output / 'problems/pddl/gripper'
    generated.mkdir(parents=True)
    domain = ROOT / 'baseline/AgentBoard/agentboard/environment/pddl_env/pddlgym/pddl/gripper.pddl'
    (generated.parent / 'gripper.pddl').write_bytes(domain.read_bytes())
    specs = []
    for index, n in enumerate((3, 5, 8)):
        balls = [f'ball{i+1}' for i in range(n)]
        initial = ['(room rooma)', '(room roomb)', '(at-robby rooma)', '(free left)', '(free right)',
                   '(gripper left)', '(gripper right)'] + [f'(ball {b}) (at {b} rooma)' for b in balls]
        text = (f'(define (problem development-{n}) (:domain gripper-strips) '
                f'(:objects rooma roomb left right {" ".join(balls)}) '
                f'(:init {" ".join(initial)}) (:goal (and {" ".join(f"(at {b} roomb)" for b in balls)})))')
        (generated / f'dev{index}.pddl').write_text(text)
        specs.append({'id': f'generated-gripper-{n}', 'domain': 'gripper', 'index': index, 'goal': 'all balls at roomb'})
    fixture = ROOT / 'runs/public_prerequisite_dev_20261117'
    records = []
    for cls in (planning.DependencyGuardedPDDL, planning.CanonicalStatePDDL, planning.WholeGoalPDDL,
                planning.ResourceGoalPDDL):
        env = cls(fixture, specs)
        try:
            for spec in specs:
                with patch.object(base.native, 'PDDL_PACKAGE', output / 'problems'):
                    observation = env.reset(spec['id'])
                steps = []
                for _ in range(30):
                    menu = env.public_action_candidates(None, {'observation_id': 'native-dev'}, env.task_goal, observation, [])
                    if not menu:
                        raise AssertionError('No native legal command')
                    command = menu[0]['command']
                    observation, done, flags = env.step_command(command)
                    assert not flags['is_error']
                    steps.append({'command': command, 'observation': observation,
                                  'final_coverage': env.metrics['final_coverage']})
                    if done:
                        break
                records.append({'arm': cls.__name__, 'task_id': spec['id'], 'steps': steps,
                                'metrics': dict(env.metrics), 'driver': 'ranked_menu_first_no_llm'})
        finally:
            env.close()
    atomic_json(output / 'pddl_native_rollouts.json', records)

    specs = [{'id': 'use-thermometer:270', 'task': 'use-thermometer', 'variation': 270}]
    env = base.BudgetedScienceWorld(specs)
    replay = fixture / 'scienceworld_target_fix/scienceworld/public_stage_prerequisites/results.jsonl'
    source = next(json.loads(s) for s in replay.read_text().splitlines()
                  if json.loads(s)['task_id'] == specs[0]['id'])
    actions = [e['command'] for e in source['history'] if e['phase'] == 'execute' and e.get('sent')][:15]
    history = []
    try:
        env.reset(specs[0]['id'])
        assert 270 in env.env.get_variations_dev() and 270 not in env.env.get_variations_test()
        env.current_goal = env.task_goal
        for command in actions:
            env.public_history = history
            assert not env.public_command_guard(command), command
            observation, done, flags = env.step_command(command)
            assert not done
            history.append({'command': command, 'observation': observation, 'is_error': flags['is_error']})
        env.public_history = history
        command = 'focus on unknown substance'
        bindings = public_aliases(history)
        assert bindings[command[9:]]['target'] == 'unknown substance b'
        assert not env.public_command_guard(command)
        observation, done, flags = env.step_command(command)
        assert 'unknown substance B' in observation
        history.append({'command': command, 'observation': observation, 'is_error': flags['is_error']})
        alias_result = {'history': history, 'bindings': bindings, 'source_sha256': file_sha(replay),
                        'native_steps': env.native_step_audits}
        waits = []
        for command in ('wait1', 'wait'):
            env.reset(specs[0]['id'])
            before = env.env.get_num_moves()
            observation, _, done, info = env.env.step(command)  # calibration, bypass candidate guard
            row = step_diagnostic(command, before, env.env.get_num_moves(), done,
                                  env.env.server.getCompleted(), env.env.envStepLimit)
            row['observation'] = observation
            assert row['native_move_delta'] == {'wait1': 2, 'wait': 11}[command]
            waits.append(row)
        env.reset(specs[0]['id'])
        for _ in range(3):
            before = env.env.get_num_moves()
            _, _, done, info = env.env.step('wait')
        truncation = step_diagnostic('wait', before, env.env.get_num_moves(), done,
                                     env.env.server.getCompleted(), env.env.envStepLimit)
        assert truncation['native_budget_truncated']
        atomic_json(output / 'scienceworld_native_validation.json',
                    {'alias_replay': alias_result, 'wait_calibration': waits, 'truncation': truncation})
    finally:
        env.shutdown()
    atomic_json(output / 'summary.json', {'passed': True, 'teacher_calls': 0, 'model_calls': 0,
        'training_updates': 0, 'pddl_driver': 'heuristic_only_not_model_evaluation',
        'pddl': [{'arm': r['arm'], 'task_id': r['task_id'], 'steps': len(r['steps']), **r['metrics']} for r in records],
        'scienceworld': {'public_alias_replay_passed': True, 'native_wait_costs': {'wait': 11, 'wait1': 2},
                        'native_budget_truncation_verified': True},
        'source_hashes': {str(p.relative_to(ROOT)): file_sha(p) for directory in ('scope', 'experiments')
                          for p in (ROOT / directory).rglob('*.py')}})


if __name__ == '__main__':
    run(Path(sys.argv[1]))
