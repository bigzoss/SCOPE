"""Small frozen development comparison, with source/data receipts and no teacher calls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch


def prepare(project, output, profile='prerequisites'):
    from scope.io import atomic_json, file_sha
    if output.exists():
        raise FileExistsError(f"Preserve existing experiment: {output}")
    output.mkdir(parents=True)
    source = output / 'source'
    for directory in ('scope', 'experiments'):
        for file in (project / directory).rglob('*.py'):
            destination = source / file.relative_to(project)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, destination)
    previous = project / 'runs/scope_formal_retest_cdbddc7_20261116/pddl'
    recipe = json.loads((previous / 'receipt.json').read_text())
    (output / 'eval_data').symlink_to((previous / 'eval_data').resolve())
    problem_root = output / 'development_pddl'
    folder = problem_root / 'pddl/gripper'
    folder.mkdir(parents=True)
    domain = project / 'baseline/AgentBoard/agentboard/environment/pddl_env/pddlgym/pddl/gripper.pddl'
    shutil.copyfile(domain, folder.parent / 'gripper.pddl')
    tasks = []
    for index, n in enumerate((3, 5, 8) if profile == 'closure' else (3, 5)):
        balls = [f'ball{i + 1}' for i in range(n)]
        init = ['(room rooma)', '(room roomb)', '(at-robby rooma)', '(free left)', '(free right)',
                '(gripper left)', '(gripper right)']
        init += [f'(ball {b}) (at {b} rooma)' for b in balls]
        goal = ' '.join(f'(at {b} roomb)' for b in balls)
        problem = (f'(define (problem scope-dev-{n}) (:domain gripper-strips) '
                   f'(:objects rooma roomb left right {" ".join(balls)}) '
                   f'(:init {" ".join(init)}) (:goal (and {goal})))')
        (folder / f'dev{index}.pddl').write_text(problem, encoding='utf-8')
        tasks.append({'id': f'generated-gripper-{n}', 'domain': 'gripper', 'index': index,
                      'goal': '; '.join(f'{b} is at roomb' for b in balls)})
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv(envStepLimit=30)
    science = []
    try:
        for name in ('use-thermometer', 'measure-melting-point-unknown-substance'):
            env.load(name, 0)
            variation = sorted(env.get_variations_dev())[0]
            assert variation not in env.get_variations_test()
            science.append({'id': f'{name}:{variation}', 'task': name, 'variation': variation})
    finally:
        env.close()
    recipe.update(split='dev', memory_gib=21, tasks={'pddl': tasks, 'scienceworld': science},
                  arms_by_domain={'pddl': ['compact_reordered_exact_loop', 'dependency_state', 'grounded_dependencies_loop',
                                          'grounded_dependencies_guard'],
                                  'scienceworld': ['public_thermal_evidence', 'public_stage_prerequisites']},
                  source_hashes={str(p.relative_to(source)): file_sha(p) for p in source.rglob('*.py')},
                  development_problem_root=str(problem_root),
                  scope_note='Two generated gripper instances and two official ScienceWorld dev variants. '
                             'Engineering pilot only; not a full benchmark or a memory-policy training comparison.')
    recipe['assets'].update({str(p): file_sha(p) for p in problem_root.rglob('*.pddl')})
    if profile == 'closure':
        recipe.update(profile=profile,
                      arms_by_domain={'pddl': ['grounded_dependencies_guard', 'canonical_state_guard',
                                              'whole_goal_guard', 'resource_goal_guard'],
                                      'scienceworld': ['public_stage_prerequisites', 'public_budget_aliases']},
                      scope_note='Three generated transport problems and two official ScienceWorld dev variants. '
                                 'Frozen model ablations; no target training. Native step limit remains 30.')
    atomic_json(output / 'receipt.json', recipe)
    atomic_json(output / 'status.json', {'phase': 'prepared', 'rollouts': sum(
        len(recipe['tasks'][domain]) * len(arms) for domain, arms in recipe['arms_by_domain'].items())})


def run(output):
    from scope.io import atomic_json
    import planning_navigation_eval as planning
    import target_action_adaptation as base
    original = base.create_env
    from public_memory_ablation import MemoryAblationRunner
    component_profile = base.shared.read(output / 'receipt.json').get('profile') == 'components'

    def runner(*args, **kwargs):
        cls = MemoryAblationRunner if component_profile and kwargs.get('domain') == 'scienceworld' else planning.Runner
        return cls(*args, **kwargs)

    def create(root, domain, arm, specs, config):
        if domain != 'pddl':
            return original(root, domain, arm, specs, config)
        env, service = planning.create_env(root, domain, arm, specs, config)
        reset = env.reset
        problem_root = Path(base.shared.read(root / 'receipt.json')['development_problem_root'])
        def reset_development(task_id):
            with patch.object(base.native, 'PDDL_PACKAGE', problem_root):
                return reset(task_id)
        env.reset = reset_development
        return env, service

    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch.object(base, 'create_env', create), patch.object(base, 'AdaptationRunner', runner):
                base.run(output, list(base.shared.read(output / 'receipt.json')['tasks']))
        except Exception as exc:
            atomic_json(output / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(output / 'status.json', {'phase': 'needs_attention'})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--project', type=Path, default=Path('/home/hyx/lm_workspace/SCOPE'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('prerequisites', 'closure'), default='prerequisites')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    source = args.project if args.mode == 'prepare' else args.output / 'source'
    sys.path[:0] = [str(source), str(source / 'experiments/teacher_context_grpo'),
                   str(args.project / 'runs/hotpot_four_bench_20261110.3j9gw8h4/native_deps')]
    os.environ['PATH'] = '/home/hyx/lm_workspace/longagent/.java/bin:' + os.environ['PATH']
    prepare(args.project, args.output, args.profile) if args.mode == 'prepare' else run(args.output)
