"""Paired frozen evaluation of HotpotQA actor warmup on four target environments."""
import argparse
from contextlib import nullcontext
import fcntl
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from experiments.memory_curriculum import native_target_eval as native
from experiments.memory_curriculum import shared_memory_eval as shared
from experiments.memory_curriculum.hotpot_transfer_eval import prepare as prepare_alfworld
from scope.alfworld import ALFWorldEnvironment
from scope.checkpoint import load_checkpoint
from scope.config import DEFAULT_RUNTIME
from scope.contracts import ContractError
from scope.evaluation import evaluate, paired_comparison
from scope.io import atomic_json, file_sha
from scope.policy import HFPolicy


EXPECTED = {'alfworld': 134, 'scienceworld': 90, 'pddl': 60, 'babyai': 112}
CONDITIONS = ('initial', 'warmup_only')
SEED = 233


def read(path):
    return shared.read(path)


def target(domain, root, model, gpu):
    if domain == 'alfworld':
        config, registry = prepare_alfworld(domain, root, gpu, model)
        config['runtime'].update(max_environment_steps=50, max_turns=100, total_tokens=850000)
        ids = registry.ids('test')
        manifest = config['environment']['manifest']
        return config, registry, ids, file_sha(manifest)
    env = native.create(domain, root)
    try:
        ids = list(env.tasks)
        runtime = {**DEFAULT_RUNTIME, 'state_graph': True, 'memory_choice': True, 'shared_memory': True,
                   'decision_evidence': True, 'action_mode': 'normalize', 'loop_policy': 'state',
                   'max_environment_steps': native.STEP_CAP, 'max_turns': 2 * native.STEP_CAP,
                   'total_tokens': 850000}
        config = {'environment': {'kind': domain}, 'runtime': runtime,
                  'evaluation': {'seed': SEED, 'sequence_length': 1}}
        return config, shared.FixedTasks(ids), ids, getattr(env, 'label_sha', None)
    finally:
        env.close()
        if isinstance(env, native.ScienceWorld):
            env.shutdown()


def run(root):
    manifest = read(root / 'receipt.json')
    source = root / 'source'
    for name, sha in manifest['source_sha256'].items():
        if file_sha(source / name) != sha:
            raise ContractError('Transfer evaluation source changed')
    for path, sha in manifest['assets_sha256'].items():
        if file_sha(path) != sha:
            raise ContractError('Target asset changed')
    queue = Path(manifest['after_ablation'])
    while read(queue)['phase'] != 'worker_exited':
        phase = read(queue)['phase']
        if phase in {'waiting_for_diagnostic_repair', 'diagnostic_process_exited_without_summary'}:
            raise ContractError(f'Previous GPU2 job stopped at {phase}')
        atomic_json(root / 'status.json', {'phase': 'queued', 'after': str(queue), 'gpu': manifest['gpu']})
        time.sleep(20)
    previous = read(queue)
    if previous['exit_code'] != 0:
        raise ContractError('Previous GPU2 ablation exited with an error')
    shared.BRIDGE = root / 'serve_alfworld.py'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    policy = HFPolicy(Path(manifest['model']), gradient_checkpointing=False)
    manager = None
    results = {}
    for domain, expected in EXPECTED.items():
        (root / domain).mkdir(exist_ok=True)
        config, registry, ids, asset_sha = target(domain, root, manifest['model'], manifest['gpu'])
        if len(ids) != expected or len(set(ids)) != expected:
            raise ContractError(f'{domain} has an unexpected task manifest')
        cells = {}
        for name in CONDITIONS:
            output = root / domain / name
            output.mkdir(exist_ok=True)
            atomic_json(root / 'status.json', {'phase': 'evaluating', 'domain': domain, 'condition': name,
                                              'expected': expected, 'gpu': manifest['gpu']})
            record = manifest['checkpoints'][name]
            checkpoint = Path(record['path'])
            if file_sha(checkpoint / 'state.pt') != record['sha256']:
                raise ContractError('Source checkpoint hash changed')
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            load_checkpoint(checkpoint, policy, restore_random=False)
            if manager is None:
                manager = policy.fingerprint('manager')
            if policy.fingerprint('manager') != manager:
                raise ContractError('Actor comparison has different manager weights')
            fingerprint = policy.fingerprint()
            receipt = {'domain': domain, 'task_ids': ids, 'asset_sha256': asset_sha, 'checkpoint': record,
                       'parameter_fingerprint': fingerprint, 'manager_sha256': manager,
                       'runtime': config['runtime'], 'seed': SEED, 'target_updates': 0, 'teacher_calls': 0}
            path = output / 'receipt.json'
            if path.exists() and read(path) != receipt:
                raise ContractError('Target evaluation receipt changed')
            atomic_json(path, receipt)
            if domain == 'alfworld':
                endpoint = config['environment']['base_url'] = f'http://127.0.0.1:{shared.port()}'
                env = ALFWorldEnvironment(endpoint, ids, config['environment']['manifest'])
                service = shared.alf_service(config, output)
            else:
                env = native.create(domain, root)
                service = nullcontext()
            try:
                with service:
                    if domain == 'alfworld':
                        env.health()
                    evaluate(policy, env, registry, config, output, 'test')
            finally:
                if domain != 'alfworld':
                    env.close()
                    if isinstance(env, native.ScienceWorld):
                        env.shutdown()
            summary = read(output / 'summary.json')
            if policy.fingerprint() != fingerprint or not summary['complete'] or summary['n'] != expected:
                raise ContractError('Target evaluation did not finish with frozen weights')
            if domain == 'alfworld' and not read(output / 'alf_cleanup.json')['own_processes_ended']:
                raise ContractError('Owned ALFWorld bridge remained active')
            cells[name] = {k: summary[k] for k in ('n', 'success_count', 'strict_success_rate',
                                                  'mean_loose_reward', 'mean_total_tokens')}
        results[domain] = {'cells': cells, 'paired': paired_comparison(root / domain / 'initial', root / domain / 'warmup_only')}
        atomic_json(root / 'summary.json', {'complete': len(results) == len(EXPECTED), 'domains': results})
    atomic_json(root / 'status.json', {'phase': 'complete', 'target_updates': 0, 'teacher_calls': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as error:
            atomic_json(root / 'error.json', {'type': type(error).__name__, 'error': str(error),
                                             'traceback': traceback.format_exc()})
            atomic_json(root / 'status.json', {'phase': 'needs_attention'})
            raise


if __name__ == '__main__':
    main()
