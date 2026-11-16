"""Compare frozen source managers with public history alone on fresh ALFWorld dev tasks."""
import argparse
import fcntl
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch

from action_memory_dev import DevPolicy, analyze, bootstrap
from experiments.memory_curriculum import shared_memory_eval as shared
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.evaluation import evaluate
from scope.io import atomic_json, digest, file_sha
from scope.policy import HFPolicy
from scope.rollout import ACTOR_SYSTEM
from target_action_adaptation import AdaptationRunner, create_env


ARMS = ('public_history', 'source_initial_manager', 'source_trained_manager')


class MemoryAblationRunner(AdaptationRunner):
    def __init__(self, *args, public_history_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.public_history_only = public_history_only

    def _menu_order_key(self, atlas, working, goal, observation, history, command):
        # Atlas revisions can differ after a write or retrieval at the same public state.
        return digest([atlas.scope.task, len(history), observation, command])

    def _base_actor_messages(self, atlas, working, goal, observation, history):
        if not self.public_history_only:
            return super()._base_actor_messages(atlas, working, goal, observation, history)
        history = list(history)
        current = observation
        limit = self.cfg.get('actor_max_input', 6144)
        while True:
            messages = [{'role': 'system', 'content': getattr(self.env, 'actor_system', ACTOR_SYSTEM)},
                        {'role': 'user', 'content': goal}]
            for record in history:
                messages += [{'role': 'assistant', 'content': record['response']},
                             {'role': 'user', 'content': record['observation']}]
            content = 'Current observation:\n' + current
            if working.get('warning'):
                content += '\n' + working['warning']
            if history:
                messages[-1]['content'] = content
            else:
                messages[-1]['content'] += '\n' + content
            if len(self.policy.encode(messages)) <= limit:
                saved = atlas.save_view([], None)
                return messages, digest(messages), saved['id']
            if history:
                history.pop(0)
            elif len(current) > 600:
                end = max(150, len(current) // 4)
                current = current[:end] + '\n[page middle omitted]\n' + current[-end:]
            else:
                raise ContractError('Cannot preserve goal and current public observation')


def wait_for_menu_validation(root, validation, independent_menu_stage=None):
    while True:
        status = shared.read(validation / 'status.json')
        if status['phase'] == 'complete':
            result = validation / 'summary.json'
            break
        if independent_menu_stage and (independent_menu_stage / 'publication.json').exists():
            result = independent_menu_stage / 'summary.json'
            break
        if status['phase'] == 'needs_attention':
            raise ContractError('ALFWorld interface validation reported an error')
        process = shared.read(validation / 'process.json')
        pid = Path('/proc') / str(process['pid'])
        if not pid.exists():
            if shared.read(validation / 'status.json')['phase'] == 'complete':
                break
            raise ContractError('ALFWorld interface validation exited before completion')
        atomic_json(root / 'status.json', {'phase': 'waiting_for_interface_validation'})
        time.sleep(20)
    before_after = shared.read(result)
    actions = before_after['domains']['alfworld']['cells']['adapted_interface']['actions']
    if not before_after['complete'] or not actions['turns_using_public_action_menu']:
        raise ContractError('The ALFWorld menu was never executed in interface validation')
    return file_sha(result)


def run(root):
    receipt_path = root / 'receipt.json'
    manifest = shared.read(receipt_path)
    for group, prefix in (('source_sha256', root / 'source'), ('assets_sha256', Path('/'))):
        for name, sha in manifest[group].items():
            if file_sha(prefix / name) != sha:
                raise ContractError(f'Frozen evaluation input changed: {name}')
    checkpoints = {key: Path(record['path']) for key, record in manifest['checkpoints'].items()}
    for key, path in checkpoints.items():
        if file_sha(path / 'state.pt') != manifest['checkpoints'][key]['sha256']:
            raise ContractError(f'Source checkpoint changed: {key}')
    validation = Path(manifest['menu_validation'])
    independent_menu_stage = manifest.get('independent_menu_stage')
    validation_sha = wait_for_menu_validation(root, validation,
                                              Path(independent_menu_stage) if independent_menu_stage else None)
    validated_arm = shared.read(validation / 'alfworld' / 'adapted_interface' / 'receipt.json')
    ids = [record['id'] for record in manifest['tasks']]
    if len(ids) != len(set(ids)) or not set(ids) <= set(validated_arm['task_ids']):
        raise ContractError('Memory probe targets differ from the preregistered ALFWorld tasks')
    runtime = validated_arm['runtime']
    shared.BRIDGE = root / 'serve_alfworld.py'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    inner = HFPolicy(manifest['model'], gradient_checkpointing=False)
    load_checkpoint(checkpoints['source_trained_manager'], inner, restore_random=False)
    fixed_actor = inner.fingerprint('actor')
    policy = DevPolicy(inner)
    cells, by_arm = {}, {}
    for arm in ARMS:
        source = 'source_initial_manager' if arm == 'public_history' else arm
        load_checkpoint(checkpoints[source], inner, restore_random=False, roles={'manager'})
        if inner.fingerprint('actor') != fixed_actor:
            raise ContractError('Fixed actor parameters changed between arms')
        public_history_only = arm == 'public_history'
        arm_runtime = {**runtime, **({'manager_mode': 'rule'} if public_history_only else {})}
        config = {'environment': {'kind': 'alfworld_http'}, 'runtime': arm_runtime,
                  'evaluation': {'seed': validated_arm['seed'], 'sequence_length': 1}}
        output = root / arm
        output.mkdir(parents=True, exist_ok=True)
        full_weights = policy.fingerprint()
        arm_receipt = {'arm': arm, 'receipt_sha256': file_sha(receipt_path), 'actor_sha256': fixed_actor,
                       'manager_sha256': None if public_history_only else inner.fingerprint('manager'),
                       'parameters_sha256': full_weights, 'task_ids': ids, 'runtime': arm_runtime}
        path = output / 'receipt.json'
        if path.exists() and shared.read(path) != arm_receipt:
            raise ContractError('Memory comparison resumption identity differs')
        atomic_json(path, arm_receipt)
        atomic_json(root / 'status.json', {'phase': 'evaluating', 'arm': arm, 'expected': len(ids)})
        env, service = create_env(root, 'alfworld', 'adapted_interface', manifest['tasks'], config)
        with service(config, output):
            try:
                env.health()
                with patch('scope.evaluation.EpisodeRunner', lambda *a, **kw: MemoryAblationRunner(
                        *a, domain='alfworld', arm='public_candidates',
                        public_history_only=public_history_only, **kw)):
                    summary = evaluate(policy, env, shared.FixedTasks(ids), config, output, 'dev')
            finally:
                env.close()
        if policy.fingerprint() != full_weights or not summary['complete'] or summary['n'] != len(ids):
            raise ContractError('Memory condition did not complete with frozen weights')
        if not shared.read(output / 'alf_cleanup.json')['own_processes_ended']:
            raise ContractError('Owned ALFWorld service did not exit')
        records, diagnostic = analyze(output)
        if len(records) != len(ids) or {row['task_id'] for row in records} != set(ids):
            raise ContractError('Evaluated ALFWorld IDs do not match the manifest')
        if public_history_only and any(row['costs'].get('manager_input') or row['costs'].get('manager_output')
                                      for row in records):
            raise ContractError('Public history control invoked a memory policy')
        by_arm[arm] = {row['task_id']: row for row in records}
        cells[arm] = {**diagnostic, **{k: summary[k] for k in ('n', 'success_count', 'strict_success_rate',
                       'mean_loose_reward', 'mean_total_tokens', 'illegal_action_rate_per_turn', 'results_sha256')},
                      'manager_sha256': arm_receipt['manager_sha256']}
    if cells[ARMS[1]]['manager_sha256'] == cells[ARMS[2]]['manager_sha256']:
        raise ContractError('Source training checkpoints contain identical managers')
    contrasts = {}
    for left, right in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
        contrasts[f'{right}_vs_{left}'] = {
            key: bootstrap([by_arm[left][task]['metrics'][key] for task in ids],
                           [by_arm[right][task]['metrics'][key] for task in ids])
            for key in ('success', 'r_loose')}
    atomic_json(root / 'summary.json', {'complete': True, 'cells': cells, 'paired': contrasts,
                                      'actor_sha256': fixed_actor, 'target_updates': 0, 'teacher_calls': 0,
                                      'interface_validation_sha256': validation_sha})
    atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve())
        except Exception as error:
            atomic_json(args.root / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
