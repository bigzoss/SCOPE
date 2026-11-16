"""Compare rule, initial and trained memory policies behind the same ALFWorld gate."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
from time import perf_counter
import traceback
from unittest.mock import patch

import torch

from action_memory_dev import DevPolicy, bootstrap
from compact_alfworld import CompactALFRunner, TreatmentALFActions
from experiments.memory_curriculum import shared_memory_eval as shared
from full_public_memory_eval import score
from optimized_formal_eval import verified_rows
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.evaluation import evaluate
from scope.io import atomic_json, file_sha
from scope.policy import HFPolicy
from target_action_adaptation import create_env


ARMS = ('rules', 'source_initial_manager', 'source_trained_manager')
TREATMENT_ARM = 'source_trained_manager_treatment_stages'


class RuleMemoryRunner(CompactALFRunner):
    """Execute the offered verified retrieval or compression when the common gate opens."""

    def _manager(self, atlas, observation, working, goal, budget, phase, greedy, boundary, costs, forced=None):
        start = perf_counter()
        diagnostics = []
        _, _, skipped, menu = self._manager_request(atlas, observation, working, goal, budget, phase, diagnostics)
        active = next((op for op in menu.values() if op['operation'] != 'FINISH'), None)
        if skipped or active is None:
            result = {'accepted': True, 'finish': True, 'reason': skipped or 'no_public_decision_gap'}
        else:
            result = self.tools.execute(atlas, working, active, phase, budget, goal)
            if active['operation'] == 'SUMMARIZE':
                working['compressed'] = True
        offered = any(op['operation'] == 'RETRIEVE' for op in menu.values())
        result.update(rule_policy=True, decision_audit={
            **self.manager_audit, 'selected': result.get('operation'), 'retrieval_offered': offered,
            'selected_finish_with_retrieval': False})
        self.last_tool_seconds = perf_counter() - start
        return None, result


def summarize(rows):
    ops = [event['result'] for row in rows for event in row['manager_operations']]
    steps = [e for row in rows for e in row['history'] if e['phase'] == 'execute']
    return {**score(rows),
            'mean_manager_tokens': sum(r['costs']['manager_input'] + r['costs']['manager_output'] for r in rows) / len(rows),
            'mean_actor_tokens': sum(r['costs']['actor_input'] + r['costs']['actor_output'] for r in rows) / len(rows),
            'operations': dict(Counter(op.get('operation') for op in ops if op.get('operation'))),
            'retrieval_offered': sum(bool(op.get('decision_audit', {}).get('retrieval_offered')) for op in ops),
            'decisions_with_retrieved_evidence': sum(bool(e['retrieved_evidence_in_view']) for e in steps)}


def run(root, memory_gib):
    receipt = shared.read(root / 'receipt.json')
    for group, directory in (('source_sha256', root / 'source'), ('assets_sha256', Path('/'))):
        for name, expected in receipt[group].items():
            if file_sha(directory / name) != expected:
                raise ContractError('ALFWorld evaluation input changed')
    models = {key: Path(value['path']) for key, value in receipt['checkpoints'].items()}
    for name, path in models.items():
        if file_sha(path / 'state.pt') != receipt['checkpoints'][name]['sha256']:
            raise ContractError('A manager checkpoint changed')
    ids = [task['id'] for task in receipt['tasks']]
    if not ids or len(ids) != len(set(ids)):
        raise ContractError('ALFWorld development IDs differ')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(
        min(1., memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    inner = HFPolicy(receipt['model'], gradient_checkpointing=False)
    load_checkpoint(models['source_trained_manager'], inner, restore_random=False)
    actor = inner.fingerprint('actor')
    policy = DevPolicy(inner)
    shared.BRIDGE = root / 'serve_alfworld.py'
    rows, metrics, hashes = {}, {}, {}
    arms = receipt.get('arms', ARMS)
    if not arms or len(set(arms)) != len(arms) or set(arms) - {*ARMS, TREATMENT_ARM}:
        raise ContractError('Unknown or duplicate ALFWorld conditions')
    for arm in arms:
        rule = arm == 'rules'
        treatment = arm == TREATMENT_ARM
        source = 'source_initial_manager' if rule else 'source_trained_manager' if treatment else arm
        load_checkpoint(models[source], inner, roles={'manager'}, restore_random=False)
        frozen = inner.fingerprint()
        before = inner.fingerprint('manager')
        runtime = {**receipt['runtime'], **({'alf_treatment_stages': True} if treatment else {})}
        config = {'runtime': runtime, 'environment': {'kind': 'alfworld_http'},
                  'evaluation': {'seed': 233, 'sequence_length': 1}}
        output = root / arm
        output.mkdir(exist_ok=True, parents=True)
        path = output / 'receipt.json'
        resolved = {'campaign_sha256': file_sha(root / 'receipt.json'), 'condition': arm,
                    'actor_sha256': actor, 'manager_sha256': before, 'ids': ids}
        if path.exists() and shared.read(path) != resolved:
            raise ContractError('ALFWorld resumption receipt changed')
        atomic_json(path, resolved)
        atomic_json(root / 'status.json', {'phase': 'evaluating', 'arm': arm, 'expected': len(ids)})
        env, service = create_env(root, 'alfworld', 'adapted_interface', receipt['tasks'], config)
        if treatment:
            env.action_adapter = TreatmentALFActions()
        runner = RuleMemoryRunner if rule else CompactALFRunner
        try:
            with service(config, output):
                env.health()
                with patch('scope.evaluation.EpisodeRunner', lambda *args, **kwargs:
                           runner(*args, **kwargs, domain='alfworld', arm='public_candidates', gated_manager=True)):
                    result = evaluate(policy, env, shared.FixedTasks(ids), config, output, 'dev')
        finally:
            env.close()
        if (not result['complete'] or not result['parameters_unchanged_during_eval']
                or inner.fingerprint('actor') != actor or inner.fingerprint() != frozen
                or not shared.read(output / 'alf_cleanup.json')['own_processes_ended']):
            raise ContractError('ALFWorld evaluation did not complete with the frozen actor and manager')
        rows[arm] = verified_rows(output / 'results.jsonl', result['results_sha256'], ids)
        metrics[arm] = {**summarize(rows[arm]), 'results_sha256': result['results_sha256']}
        if rule and metrics[arm]['mean_manager_tokens'] != 0:
            raise ContractError('Rules condition invoked the manager model')
        hashes[arm] = before
    if ARMS[1] in hashes and ARMS[2] in hashes and hashes[ARMS[1]] == hashes[ARMS[2]]:
        raise ContractError('The two manager checkpoints are identical')
    if TREATMENT_ARM in hashes and ARMS[2] in hashes and hashes[TREATMENT_ARM] != hashes[ARMS[2]]:
        raise ContractError('Treatment stages changed the manager weights')
    paired = {}
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            paired[f'{right}_vs_{left}'] = {
                'strict_success': bootstrap([row['metrics']['success'] for row in rows[left]],
                                            [row['metrics']['success'] for row in rows[right]]),
                'total_tokens': bootstrap([row['costs']['charged_tokens'] for row in rows[left]],
                                          [row['costs']['charged_tokens'] for row in rows[right]])}
    atomic_json(root / 'summary.json', {'complete': True, 'split': 'dev', 'cells': metrics, 'paired': paired,
                'actor_sha256': actor, 'manager_sha256': hashes, 'teacher_calls': 0, 'target_updates': 0})
    atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, default=21)
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve(), args.memory_gib)
        except Exception as exc:
            atomic_json(args.root / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
