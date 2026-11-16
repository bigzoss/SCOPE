"""Evaluate the frozen public-memory comparison on fixed ALFWorld and ScienceWorld tests."""
import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
from pathlib import Path
import traceback
from unittest.mock import patch

import torch

from action_memory_dev import DevPolicy, analyze, bootstrap
from experiments.memory_curriculum import shared_memory_eval as shared
from public_memory_ablation import ARMS, MemoryAblationRunner
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.evaluation import evaluate
from scope.io import atomic_json, atomic_jsonl, file_sha
from scope.policy import HFPolicy
from target_action_adaptation import GUARDED_SCIENCE_ARM, create_env


def score(rows):
    steps = [step for row in rows for step in row['history'] if step['phase'] == 'execute']
    return {'n': len(rows), 'success_count': sum(row['metrics']['success'] for row in rows),
            'strict_success_rate': sum(row['metrics']['success'] for row in rows) / len(rows),
            'mean_native_progress': sum(row['metrics']['r_loose'] for row in rows) / len(rows),
            'mean_total_tokens': sum(row['costs']['charged_tokens'] for row in rows) / len(rows),
            'illegal_action_rate_per_turn': sum(row['illegal_actions'] for row in rows) / max(1, len(steps)),
            'native_score_improvements': sum(step['native_score_improved'] for step in steps),
            'native_accepted_actions': sum(step['action_status'] == 'accepted' for step in steps),
            'actions_sent': sum(step['sent'] is True for step in steps),
            'confirmed_public_action_effects': sum(step['new_public_action_confirmation'] for step in steps),
            'termination_reasons': dict(Counter(row['termination_reason'] for row in rows)),
            'terminal_focus_failures': sum(row['termination_reason'] == 'environment_done'
                and not row['metrics']['success'] and any(
                step['progress'] == 'terminal' and (step.get('normalized_command') or '').startswith('focus on ')
                for step in row['history'] if step['phase'] == 'execute') for row in rows)}


def evaluate_shard(root, domain, specs, arm, manifest, policy, fixed_actor, output):
    source = 'source_initial_manager' if arm == 'public_history' else arm
    load_checkpoint(Path(manifest['checkpoints'][source]['path']), policy.policy,
                    restore_random=False, roles={'manager'})
    if policy.fingerprint('actor') != fixed_actor:
        raise ContractError('Frozen executor changed')
    runtime = {**manifest['benchmarks'][domain]['runtime']}
    history_only = arm == 'public_history'
    if history_only:
        runtime['manager_mode'] = 'rule'
    ids = [row['id'] for row in specs]
    config = {'environment': {'kind': 'alfworld_http' if domain == 'alfworld' else domain},
              'runtime': runtime, 'evaluation': {'seed': manifest['seed'], 'sequence_length': 1}}
    receipt = {'manifest_sha256': file_sha(root / 'receipt.json'), 'arm': arm, 'task_ids': ids,
               'domain': domain, 'runtime': runtime, 'actor_sha256': fixed_actor,
               'manager_sha256': None if history_only else policy.fingerprint('manager'),
               'weights_sha256': policy.fingerprint()}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'receipt.json'
    if path.exists() and shared.read(path) != receipt:
        raise ContractError('Resumed shard has a different receipt')
    atomic_json(path, receipt)
    arm_name = 'adapted_interface' if domain == 'alfworld' else GUARDED_SCIENCE_ARM
    env, service = create_env(root, domain, arm_name, specs, config)
    try:
        with (service(config, output) if service else nullcontext()):
            try:
                if service:
                    env.health()
                with patch('scope.evaluation.EpisodeRunner', lambda *a, **kw: MemoryAblationRunner(
                        *a, domain=domain, arm='public_candidates', public_history_only=history_only, **kw)):
                    summary = evaluate(policy, env, shared.FixedTasks(ids), config, output, 'test')
            finally:
                env.close()
    finally:
        if domain == 'scienceworld':
            env.shutdown()
    if (not summary['complete'] or not summary['memory_unchanged'] or not summary['parameters_unchanged_during_eval']
            or policy.fingerprint() != receipt['weights_sha256']
            or (service and not shared.read(output / 'alf_cleanup.json')['own_processes_ended'])):
        raise ContractError('Shard failed frozen evaluation checks')
    rows, audit = analyze(output)
    if ([row['task_id'] for row in rows] != ids or file_sha(output / 'results.jsonl') != summary['results_sha256']
            or (history_only and any(row['costs']['manager_input'] or row['costs']['manager_output'] for row in rows))):
        raise ContractError('Task, data hash or public-history control mismatch')
    return rows, audit['memory']


def run(root, domain, memory_gib):
    manifest = shared.read(root / 'receipt.json')
    for group, prefix in (('source_sha256', root / 'source'), ('assets_sha256', Path('/'))):
        for name, sha in manifest[group].items():
            if file_sha(prefix / name) != sha:
                raise ContractError(f'Frozen asset changed: {name}')
    for record in manifest['checkpoints'].values():
        if file_sha(Path(record['path']) / 'state.pt') != record['sha256']:
            raise ContractError('Source checkpoint changed')
    shared.BRIDGE = root / 'serve_alfworld.py'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(
        min(1., memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    inner = HFPolicy(manifest['model'], gradient_checkpointing=False)
    load_checkpoint(manifest['checkpoints']['source_trained_manager']['path'], inner, restore_random=False)
    fixed_actor = inner.fingerprint('actor')
    if fixed_actor != manifest['actor_sha256']:
        raise ContractError('Executor fingerprint differs from the development comparison')
    policy = DevPolicy(inner)
    benchmark = manifest['benchmarks'][domain]
    specs = benchmark['tasks']
    by_arm, audits = {arm: [] for arm in ARMS}, {arm: Counter() for arm in ARMS}
    output = root / domain
    chunk_size = manifest['chunk_size']
    for start in range(0, len(specs), chunk_size):
        for arm in ARMS:
            shard = output / arm / f'shard_{start // chunk_size:03d}'
            atomic_json(output / 'status.json', {'phase': 'evaluating', 'arm': arm,
                        'shard': shard.name, 'tasks_per_arm': len(specs),
                        'completed': {key: len(rows) for key, rows in by_arm.items()}})
            rows, audit = evaluate_shard(root, domain, specs[start:start + chunk_size], arm,
                                         manifest, policy, fixed_actor, shard)
            by_arm[arm].extend(rows)
            audits[arm].update(audit)
    cells, contrasts = {}, {}
    ids = [row['id'] for row in specs]
    for arm, rows in by_arm.items():
        if [row['task_id'] for row in rows] != ids or len(set(ids)) != benchmark['expected']:
            raise ContractError('Result IDs do not match the entire frozen test manifest')
        result = output / arm / 'results.jsonl'
        atomic_jsonl(result, rows)
        cells[arm] = {**score(rows), 'memory_usage': dict(audits[arm]), 'results_sha256': file_sha(result)}
    for left, right in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
        contrasts[f'{right}_vs_{left}'] = {
            metric: bootstrap([row['metrics'][metric] for row in by_arm[left]],
                              [row['metrics'][metric] for row in by_arm[right]])
            for metric in ('success', 'r_loose')}
    atomic_json(output / 'summary.json', {'complete': True, 'benchmark': domain, 'split': benchmark['split'],
                'cells': cells, 'paired': contrasts, 'actor_sha256': fixed_actor,
                'manifest_sha256': file_sha(root / 'receipt.json'), 'target_updates': 0, 'teacher_calls': 0})
    atomic_json(output / 'status.json', {'phase': 'complete', 'completed': {arm: len(specs) for arm in ARMS}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--domain', choices=('alfworld', 'scienceworld'), required=True)
    parser.add_argument('--memory-gib', type=float, required=True)
    args = parser.parse_args()
    output = args.root / args.domain
    output.mkdir(exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve(), args.domain, args.memory_gib)
        except Exception as error:
            atomic_json(output / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
            atomic_json(output / 'status.json', {'phase': 'needs_attention'})
            raise
