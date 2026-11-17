"""Frozen full PDDL/ScienceWorld retest with archived paired controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ARMS = {'pddl': 'grounded_dependencies_guard', 'scienceworld': 'public_stage_prerequisites'}
CLOSURE_ARMS = {'pddl': 'resource_goal_guard', 'scienceworld': 'public_budget_aliases'}
COUNTS = {'pddl': 60, 'scienceworld': 90}


def read(path):
    return json.loads(path.read_text())


def prepare(project, root, profile='prerequisites'):
    from scope.io import atomic_json, file_sha
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    old = project / 'runs/scope_formal_retest_cdbddc7_20261116'
    context, planning = read(old / 'receipt.json'), read(old / 'pddl/receipt.json')
    assert context['model'] == planning['model'] and context['checkpoint'] == planning['checkpoint']
    for directory in ('scope', 'experiments'):
        for file in (project / directory).rglob('*.py'):
            dest = root / 'source' / file.relative_to(project)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
    hashes = {str(p.relative_to(root / 'source')): file_sha(p) for p in (root / 'source').rglob('*.py')}
    arms = {'pddl': 'transport_only'} if profile == 'transport' else CLOSURE_ARMS if profile == 'closure' else ARMS
    manifest = {'domains': {}, 'counts': COUNTS, 'arms': arms, 'profile': profile,
                'teacher_calls': 0, 'target_updates': 0,
                'scope': 'Fixed formal regression sets previously used for error analysis; not a new untouched test.'}
    for domain, arm in arms.items():
        target = root / domain
        target.mkdir()
        (target / 'source').symlink_to(root / 'source')
        (target / 'eval_data').symlink_to((old / 'pddl/eval_data').resolve())
        tasks = planning['tasks']['pddl'] if domain == 'pddl' else context['benchmarks'][domain]['tasks']
        assert len(tasks) == len({row['id'] for row in tasks}) == COUNTS[domain]
        reference = old / ('pddl/pddl/compact_reordered_exact_loop' if domain == 'pddl' else 'scienceworld/optimized')
        if profile in {'closure', 'transport'}:
            prior = project / 'runs/public_prerequisite_formal_20261117'
            assert read(prior / 'status.json')['phase'] == 'complete'
            previous = read(prior / domain / 'receipt.json')
            assert previous['model'] == planning['model'] and previous['checkpoint'] == planning['checkpoint']
            assert previous['tasks'][domain] == tasks
            reference = prior / domain / domain / ARMS[domain]
        baseline = read(reference / 'summary.json')
        assert baseline['complete'] and baseline['n'] == COUNTS[domain]
        assert file_sha(reference / 'results.jsonl') == baseline['results_sha256']
        runtime = (read(reference / 'receipt.json')['runtime'] if domain == 'pddl' or profile == 'closure'
                   else context['benchmarks'][domain]['runtime'])
        recipe = {**planning, 'split': 'test', 'memory_gib': 21, 'tasks': {domain: tasks},
                  'arms_by_domain': {domain: [arm]}, 'source_hashes': hashes,
                  'assets': {**planning['assets'], **context['assets_sha256']}}
        atomic_json(target / 'receipt.json', recipe)
        manifest['domains'][domain] = {'task_ids': [row['id'] for row in tasks], 'tasks': tasks,
            'runtime': runtime, 'reference': str(reference), 'reference_sha256': baseline['results_sha256'],
            'reference_parameter_hash': read(reference / 'sequences/sequence_00000.json')['parameter_hash']}
    atomic_json(root / 'manifest.json', manifest)
    atomic_json(root / 'status.json', {'phase': 'prepared', 'expected': sum(COUNTS[d] for d in arms)})


def worker(root, domain):
    import planning_navigation_eval as planning
    import target_action_adaptation as base
    from public_memory_ablation import MemoryAblationRunner
    from scope.config import DEFAULT_RUNTIME
    from scope.io import atomic_json
    target = root / domain
    original = base.create_env
    def create(path, name, arm, specs, config):
        if name == 'pddl':
            return planning.create_env(path, name, arm, specs, config)
        return original(path, name, arm, specs, config)
    runtime = {**DEFAULT_RUNTIME, 'state_graph': True, 'memory_choice': True, 'shared_memory': True,
               'decision_evidence': True, 'action_mode': 'normalize', 'loop_policy': 'state',
               'max_environment_steps': 30, 'max_turns': 60, 'total_tokens': 850000, 'decision_recovery': False}
    assert runtime == read(root / 'manifest.json')['domains'][domain]['runtime'], 'Historical runtime differs'
    with (target / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            runner = planning.Runner if domain == 'pddl' else MemoryAblationRunner
            with patch.object(base, 'create_env', create), patch.object(base, 'AdaptationRunner', runner):
                base.run(target, [domain])
        except Exception as exc:
            atomic_json(target / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            raise


def summarize(root, domain):
    from collections import Counter
    from action_memory_dev import bootstrap
    from scope.io import file_sha
    from full_public_memory_eval import score
    from optimized_formal_eval import compare, verified_rows
    manifest = read(root / 'manifest.json')
    spec = manifest['domains'][domain]
    output = root / domain / domain / manifest['arms'][domain]
    summary = read(output / 'summary.json')
    assert all(summary[k] for k in ('complete', 'memory_unchanged', 'parameters_unchanged_during_eval'))
    rows = verified_rows(output / 'results.jsonl', summary['results_sha256'], spec['task_ids'])
    baseline = verified_rows(Path(spec['reference']) / 'results.jsonl', spec['reference_sha256'], spec['task_ids'])
    for path in (output, Path(spec['reference'])):
        assert read(path / 'sequences/sequence_00000.json')['parameter_hash'] == spec['reference_parameter_hash']
    assert len(rows) == summary['n'] == COUNTS[domain]
    assert all(x['environment_steps'] <= 30 and x['actor_turns'] <= 60 for x in rows)
    def aggregate(data):
        result = score(data)
        if domain == 'pddl':
            result['mean_final_coverage'] = sum(x['metrics']['final_coverage'] for x in data) / len(data)
        elif manifest.get('profile') == 'closure':
            diagnostics = [e['native_step_diagnostic'] for row in data for e in row['history']
                           if 'native_step_diagnostic' in e]
            if diagnostics:
                assert len(diagnostics) == result['actions_sent']
                result['mean_native_moves'] = sum(e['native_move_delta'] for e in diagnostics) / len(data)
                result['native_wait_commands'] = dict(Counter(
                    e['command'] for e in diagnostics if e['command'] in {'wait', 'wait1'}))
        return result
    group_field = 'domain' if domain == 'pddl' else 'task'
    groups = {}
    for group in sorted({x[group_field] for x in spec['tasks']}):
        ids = {x['id'] for x in spec['tasks'] if x[group_field] == group}
        groups[group] = {'baseline': aggregate([x for x in baseline if x['task_id'] in ids]),
                         'optimized': aggregate([x for x in rows if x['task_id'] in ids])}
    paired = compare(rows, baseline)
    if domain == 'pddl':
        paired['final_coverage'] = bootstrap([x['metrics']['final_coverage'] for x in baseline],
                                             [x['metrics']['final_coverage'] for x in rows])
    return {'complete': True, 'baseline': aggregate(baseline), 'optimized': aggregate(rows),
            'paired': paired, 'groups': groups, 'results_sha256': file_sha(output / 'results.jsonl')}


def supervise(root):
    from scope.io import atomic_json
    arms = read(root / 'manifest.json')['arms']
    results = {}
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for domain, arm in arms.items():
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(root), '--domain', domain]
            with (root / domain / 'worker.log').open('ab') as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            atomic_json(root / domain / 'process.json', {'pid': process.pid, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'command': command})
            while process.poll() is None:
                p = root / domain / domain / arm / 'status.json'
                done = read(p).get('completed', 0) if p.exists() else 0
                atomic_json(root / 'status.json', {'phase': 'evaluating', 'domain': domain,
                    'completed_in_domain': done, 'expected_in_domain': COUNTS[domain],
                    'completed': sum(COUNTS[d] for d in results) + done, 'expected': sum(COUNTS[d] for d in arms),
                    'completed_domains': list(results), 'worker_pid': process.pid})
                time.sleep(10)
            if process.returncode:
                error = root / domain / 'error.json'
                raise RuntimeError(read(error)['error'] if error.exists() else f'{domain} exited {process.returncode}')
            results[domain] = summarize(root, domain)
            atomic_json(root / domain / 'paired_summary.json', results[domain])
        atomic_json(root / 'summary.json', {'complete': True, 'domains': results, 'teacher_calls': 0,
                                           'target_updates': 0, 'own_workers_ended': True})
        lines = ['# PDDL / ScienceWorld full regression retest', '',
                 '| Environment | Previous success | New success | Previous progress | New progress |',
                 '|---|---:|---:|---:|---:|']
        for domain, record in results.items():
            a, b = record['baseline'], record['optimized']
            lines.append(f"| {domain} | {a['success_count']}/{a['n']} | {b['success_count']}/{b['n']} | "
                         f"{a['mean_native_progress']:.4f} | {b['mean_native_progress']:.4f} |")
        lines += ['', 'Frozen weights and budgets; zero teacher calls or target updates. These fixed sets have informed earlier error analysis.',
                  'See summary.json for paired confidence intervals, final PDDL coverage, per-domain metrics and token costs.']
        (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        atomic_json(root / 'status.json', {'phase': 'complete', 'completed': sum(COUNTS[d] for d in arms), 'expected': sum(COUNTS[d] for d in arms)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'worker', 'supervise'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--domain', choices=tuple(ARMS))
    parser.add_argument('--profile', choices=('prerequisites', 'closure', 'transport'), default='prerequisites')
    args = parser.parse_args()
    project = Path('/home/hyx/lm_workspace/SCOPE')
    source = project if args.mode == 'prepare' else args.root / 'source'
    sys.path[:0] = [str(source), str(source / 'experiments/teacher_context_grpo'),
                   str(project / 'runs/hotpot_four_bench_20261110.3j9gw8h4/native_deps')]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['PATH'] = '/home/hyx/lm_workspace/longagent/.java/bin:' + os.environ['PATH']
    if args.mode == 'prepare':
        prepare(project, args.root, args.profile)
    elif args.mode == 'worker':
        worker(args.root, args.domain)
    else:
        try:
            supervise(args.root)
        except Exception as exc:
            from scope.io import atomic_json
            atomic_json(args.root / 'supervisor_error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
