"""Run the optimized policies on the frozen formal ALFWorld and ScienceWorld manifests."""
import argparse
from contextlib import nullcontext
import fcntl
from pathlib import Path
import traceback
from unittest.mock import patch

import torch

from action_memory_dev import DevPolicy, analyze, bootstrap
from compact_alfworld import CompactALFRunner
from experiments.memory_curriculum import shared_memory_eval as shared
from full_public_memory_eval import score
from public_memory_ablation import MemoryAblationRunner
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.evaluation import evaluate
from scope.io import atomic_json, file_sha, read_jsonl
from scope.policy import HFPolicy
from target_action_adaptation import GROUNDED_SCIENCE_ARM, create_env


def verified_rows(path, expected_sha, ids):
    if file_sha(path) != expected_sha:
        raise ContractError(f'Result file changed: {path}')
    rows = read_jsonl(path)
    if [row['task_id'] for row in rows] != ids:
        raise ContractError(f'Result task IDs differ from the frozen order: {path}')
    return rows


def compare(rows, baseline):
    deltas = {}
    for label, key in [('strict_success', 'success'), ('native_progress', 'r_loose')]:
        deltas[label] = bootstrap([row['metrics'][key] for row in baseline],
                                  [row['metrics'][key] for row in rows])
    deltas['total_tokens'] = bootstrap([row['costs']['charged_tokens'] for row in baseline],
                                      [row['costs']['charged_tokens'] for row in rows])
    deltas['rescued'] = sum(a['metrics']['success'] < b['metrics']['success'] for a, b in zip(baseline, rows))
    deltas['regressed'] = sum(a['metrics']['success'] > b['metrics']['success'] for a, b in zip(baseline, rows))
    return deltas


def run(root, domain, gpu_memory):
    manifest = shared.read(root / 'receipt.json')
    for group, prefix in [('source_sha256', root / 'source'), ('assets_sha256', Path('/'))]:
        for name, sha in manifest[group].items():
            if file_sha(prefix / name) != sha:
                raise ContractError(f'Frozen input changed: {name}')
    benchmark = manifest['benchmarks'][domain]
    ids = [row['id'] for row in benchmark['tasks']]
    if len(set(ids)) != len(ids) or len(ids) != benchmark['expected']:
        raise ContractError('Formal manifest count differs')
    reference = manifest['reference'][domain]
    prior_summary = shared.read(Path(reference['summary']))
    if not prior_summary['complete'] or prior_summary['actor_sha256'] != manifest['actor_sha256']:
        raise ContractError('Historical reference uses a different actor or an incomplete test')
    references = {
        arm: verified_rows(Path(path), prior_summary['cells'][arm]['results_sha256'], ids)
        for arm, path in reference['cells'].items()
    }
    checkpoint = manifest['checkpoint']
    if file_sha(Path(checkpoint['path']) / 'state.pt') != checkpoint['sha256']:
        raise ContractError('Frozen policy checkpoint changed')
    shared.BRIDGE = root / 'serve_alfworld.py'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(
        min(1., gpu_memory * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    model = HFPolicy(manifest['model'], gradient_checkpointing=False)
    load_checkpoint(checkpoint['path'], model, restore_random=False)
    if model.fingerprint('actor') != manifest['actor_sha256']:
        raise ContractError('Executor fingerprint differs from the previous formal test')
    frozen = model.fingerprint()
    policy = DevPolicy(model)
    runtime = dict(benchmark['runtime'])
    config = {'environment': {'kind': 'alfworld_http' if domain == 'alfworld' else domain},
              'runtime': runtime, 'evaluation': {'seed': manifest['seed'], 'sequence_length': 1}}
    output = root / domain
    arm = 'adapted_interface' if domain == 'alfworld' else GROUNDED_SCIENCE_ARM
    env, service = create_env(root, domain, arm, benchmark['tasks'], config)
    if domain == 'alfworld':
        def runner(*args, **kwargs):
            return CompactALFRunner(*args, **kwargs, domain=domain, arm='public_candidates', gated_manager=True)
    else:
        def runner(*args, **kwargs):
            return MemoryAblationRunner(*args, **kwargs, domain=domain, arm='public_candidates')
    try:
        atomic_json(output / 'status.json', {'phase': 'evaluating', 'expected': len(ids)})
        with (service(config, output) if service else nullcontext()):
            try:
                if service:
                    env.health()
                with patch('scope.evaluation.EpisodeRunner', runner):
                    outcome = evaluate(policy, env, shared.FixedTasks(ids), config, output / 'optimized', 'test')
            finally:
                env.close()
    finally:
        if domain == 'scienceworld':
            env.shutdown()
    if (not outcome['complete'] or not outcome['memory_unchanged']
            or not outcome['parameters_unchanged_during_eval'] or model.fingerprint() != frozen):
        raise ContractError('Formal evaluation did not complete with frozen weights')
    if service and not shared.read(output / 'alf_cleanup.json')['own_processes_ended']:
        raise ContractError('Owned ALFWorld bridge did not exit')
    result = output / 'optimized' / 'results.jsonl'
    rows = verified_rows(result, outcome['results_sha256'], ids)
    _, diagnostic = analyze(result.parent)
    atomic_json(output / 'summary.json', {
        'complete': True, 'benchmark': domain, 'split': benchmark['split'], 'n': len(rows),
        'optimized': {**score(rows), 'memory_usage': diagnostic['memory'], 'results_sha256': file_sha(result)},
        'reference': {arm: prior_summary['cells'][arm] for arm in references},
        'paired': {arm: compare(rows, baseline) for arm, baseline in references.items()},
        'actor_sha256': manifest['actor_sha256'], 'parameter_sha256': frozen,
        'manifest_sha256': file_sha(root / 'receipt.json'), 'target_updates': 0, 'teacher_calls': 0})
    atomic_json(output / 'status.json', {'phase': 'complete', 'completed': len(rows), 'expected': len(ids)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--domain', choices=('alfworld', 'scienceworld'), required=True)
    parser.add_argument('--memory-gib', type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.domain
    output.mkdir(exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root, args.domain, args.memory_gib)
        except Exception as exc:
            atomic_json(output / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(output / 'status.json', {'phase': 'needs_attention'})
            raise
