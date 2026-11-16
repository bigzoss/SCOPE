"""Replay one accepted management operation against KEEP on the same public prefix."""
import argparse
import fcntl
from pathlib import Path
import traceback

import torch

from action_memory_dev import DevPolicy
from compact_alfworld import CompactALFRunner
from experiments.memory_curriculum import shared_memory_eval as shared
from public_prefix_repair import prefix_identity
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.io import atomic_json, atomic_jsonl, digest, file_sha, read_jsonl
from scope.policy import HFPolicy
from target_action_adaptation import create_env


class CounterfactualManagerRunner(CompactALFRunner):
    def run_probe(self, task_id, episode_id, operation, keep=None):
        self.desired_operation = operation
        self.intervention_plan = keep
        self.anchor = None
        self.manager_index = 0
        result = self.run(task_id, episode_id, greedy=True)
        steps = [event for event in result.events if event['phase'] == 'execute']
        if keep and self.anchor is None:
            raise ContractError('KEEP replay failed to reach the selected manager decision')
        return result, steps, self.anchor

    def _manager(self, atlas, observation, working, goal, budget, phase, greedy, boundary, costs, forced=None):
        index = self.manager_index
        self.manager_index += 1
        intervention = self.intervention_plan
        stamp = {'index': index, 'boundary': boundary, 'phase': phase, 'actor_turn': len(self.turn_audits)}
        if intervention and intervention['index'] == index:
            if stamp != intervention:
                raise ContractError('KEEP replay diverged before the manager action')
            sample, result = super()._manager(atlas, observation, working, goal, budget, phase, greedy,
                                               boundary, costs, forced='FINISH')
            if sample is not None or result.get('operation') != 'FINISH':
                raise ContractError('KEEP intervention failed to replace the selected operation')
            self.anchor = stamp
            return sample, result
        sample, result = super()._manager(atlas, observation, working, goal, budget, phase, greedy,
                                           boundary, costs, forced=forced)
        if not intervention and self.anchor is None and sample is not None and result['accepted'] and (
                result.get('operation') == self.desired_operation):
            self.anchor = stamp
        return sample, result


def groups(results):
    selected = []
    for target in ('RETRIEVE', 'SUMMARIZE'):
        eligible = [row for row in results if any(event.get('sampled') and event['result']['accepted']
                    and event['result'].get('operation') == target for event in row['manager_operations'])]
        ordered = sorted(eligible, key=lambda row: digest([233, target, row['task_id']]))
        selected.extend((row, target) for row in ordered[:2])
    return selected


def run(root):
    receipt = shared.read(root / 'receipt.json')
    for path, expected in receipt['assets_sha256'].items():
        if file_sha(path) != expected:
            raise ContractError('Replay input changed')
    for name, expected in receipt['source_sha256'].items():
        if file_sha(root / 'source' / name) != expected:
            raise ContractError('Replay driver source changed')
    main = Path(receipt['ablation'])
    result_path = main / 'source_trained_manager/results.jsonl'
    expected = shared.read(main / 'summary.json')['cells']['source_trained_manager']['results_sha256']
    if file_sha(result_path) != expected:
        raise ContractError('Trained manager ablation results changed')
    candidates = groups(read_jsonl(result_path))
    if not candidates:
        atomic_jsonl(root / 'pairs.jsonl', [])
        atomic_json(root / 'summary.json', {'complete': True, 'selected': 0, 'reason': 'no_accepted_manager_actions'})
        atomic_json(root / 'status.json', {'phase': 'complete'})
        return
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(
        min(1., 21 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    checkpoint = receipt['checkpoints']['source_trained_manager']
    if file_sha(Path(checkpoint['path']) / 'state.pt') != checkpoint['sha256']:
        raise ContractError('Replay checkpoint differs')
    model = HFPolicy(receipt['model'], gradient_checkpointing=False)
    load_checkpoint(checkpoint['path'], model, restore_random=False)
    frozen = model.fingerprint()
    shared.BRIDGE = root / 'serve_alfworld.py'
    config = {'environment': {'kind': 'alfworld_http'}, 'runtime': receipt['runtime']}
    env, service = create_env(root, 'alfworld', 'adapted_interface', receipt['tasks'], config)
    rows = []
    with service(config, root):
        try:
            env.health()
            runner = CounterfactualManagerRunner(DevPolicy(model), env, config['runtime'], 'dev',
                                                 domain='alfworld', arm='public_candidates', gated_manager=True)
            for reference, operation in candidates:
                task = reference['task_id']
                episode = f"eval:{reference['sequence_id']}:{task}"
                factual, decisions, anchor = runner.run_probe(task, episode, operation)
                reference_steps = [event for event in reference['history'] if event['phase'] == 'execute']
                if (len(decisions) != len(reference_steps) or factual.metrics != reference['metrics']
                        or prefix_identity(decisions, len(decisions)) != prefix_identity(reference_steps, len(reference_steps))):
                    raise ContractError('The factual replay differs from the evaluated trajectory')
                if anchor is None:
                    rows.append({'task_id': task, 'operation': operation, 'status': 'selected_operation_not_reproduced'})
                    continue
                kept, control, _ = runner.run_probe(task, episode, operation, keep=anchor)
                turn = anchor['actor_turn']
                if (len(decisions) <= turn or len(control) <= turn or
                        prefix_identity(decisions, turn) != prefix_identity(control, turn)):
                    raise ContractError('Environment public prefix differs before the manager choice')
                actual, keep = decisions[turn], control[turn]
                rows.append({'task_id': task, 'operation': operation, 'status': 'paired', 'boundary': anchor,
                             'factual_command': actual['normalized_command'], 'keep_command': keep['normalized_command'],
                             'strict_delta': factual.metrics['success'] - kept.metrics['success'],
                             'token_delta': factual.costs['charged_tokens'] - kept.costs['charged_tokens'],
                             'keep_evidence_preserved': set(keep['required_evidence_ids']) <= set(actual['context_evidence_ids'])})
                atomic_jsonl(root / 'pairs.jsonl', rows)
                atomic_json(root / 'status.json', {'phase': 'evaluating', 'completed': len(rows), 'expected': len(candidates)})
        finally:
            env.close()
    if model.fingerprint() != frozen or not shared.read(root / 'alf_cleanup.json')['own_processes_ended']:
        raise ContractError('Replay model changed or native service remained active')
    atomic_jsonl(root / 'pairs.jsonl', rows)
    pairs = [row for row in rows if row['status'] == 'paired']
    atomic_json(root / 'summary.json', {'complete': True, 'selected': len(candidates), 'paired': len(pairs),
                'changed_actions': sum(r['factual_command'] != r['keep_command'] for r in pairs),
                'strict_positive': sum(r['strict_delta'] > 0 for r in pairs),
                'strict_negative': sum(r['strict_delta'] < 0 for r in pairs),
                'compression_savings_with_evidence': sum(r['operation'] == 'SUMMARIZE' and
                    r['strict_delta'] == 0 and r['token_delta'] < 0 and r['keep_evidence_preserved'] for r in pairs),
                'target_updates': 0, 'pairs_sha256': file_sha(root / 'pairs.jsonl')})
    atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve())
        except Exception as exc:
            atomic_json(args.root / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
