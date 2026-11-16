"""Fixed-budget local sampling and credit ablation for the warmed HotpotQA actor."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.memory_curriculum.hotpot_grpo import step_data
from frozen_memory_diagnostics import compare, journal, metrics, question_key, read
from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.contracts import ContractError
from scope.hotpot import rollout, split_row
from scope.hotpot_intervention import continuation_tokens
from scope.io import append_jsonl, atomic_json, digest, file_sha


ARMS = {
    'independent_paired': ('independent', 'paired_delta'),
    'distinct_paired': ('distinct_effects', 'paired_delta'),
    'distinct_group': ('distinct_effects', 'group_relative'),
}


def prepare(root, diagnostic):
    result = read(diagnostic / 'summary.json')
    if not result['complete'] or result['prefix_n'] != 128 or not result['parameters_unchanged']:
        raise ContractError('Frozen diagnostics must complete before the training ablation')
    findings = result['prefix_contrasts_vs_keep']
    # Open the pilot only for actions already executable in the verified menu.
    candidates = {name: findings[name]['em'] for name in ('retrieve:public_menu_retrieval',) if name in findings}
    probes = journal(diagnostic / 'prefix_diagnostics.jsonl')
    compression = [r['cells'] for r in probes if {'standard_compression', 'protected_compression'} <= set(r['cells'])]
    retrieval = [r['cells'] for r in probes if 'public_menu_retrieval' in r['cells']]
    if retrieval:
        candidates['public_menu_vs_current'] = compare([x['current'] for x in retrieval],
                                                       [x['public_menu_retrieval'] for x in retrieval], 'em')
    if compression:
        candidates['protected_vs_standard'] = compare([x['standard_compression'] for x in compression],
                                                       [x['protected_compression'] for x in compression], 'em')
    triggers = {k: v for k, v in candidates.items() if v['wins'] > v['losses']}
    if not triggers:
        atomic_json(root / 'status.json', {'phase': 'candidate_gate_not_met', 'candidate_evidence': candidates})
        return False
    old = read(diagnostic / 'receipt.json')
    pair = Path(old['checkpoints']['warmup_only']['path']).parents[3]
    source_data = pq.read_table(pair / 'data/train.parquet').to_pylist()
    previous = read(pair / 'receipt.json')
    old_pair = read(Path(previous['source_run']) / 'receipt.json')
    used = set(previous['train_ids']) | set(previous['warmup_ids']) | set(old_pair['train']['ids'][:3000])
    dev = pq.read_table(diagnostic / 'reserved_dev.parquet').to_pylist()
    heldout = {question_key(r['question']) for r in dev}
    selected = [r for r in source_data if r['id'] not in used and question_key(r['question']) not in heldout][:256]
    if len(selected) != 256 or len(dev) != 512:
        raise ContractError('Ablation splits are incomplete')
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), root / 'train.parquet')
    shutil.copy2(diagnostic / 'reserved_dev.parquet', root / 'dev.parquet')
    manifest = {'model': old['model'], 'warmstart': old['checkpoints']['warmup_only'], 'rollout_options': old['rollout_options'],
                'seed': 233, 'train_ids': [r['id'] for r in selected], 'dev_ids': [r['id'] for r in dev],
                'arms': ARMS, 'group_size': 4, 'teacher_calls': 0, 'candidate_gate': triggers,
                'diagnostic_summary_sha256': file_sha(diagnostic / 'summary.json'),
                'train_sha256': file_sha(root / 'train.parquet'), 'dev_sha256': file_sha(root / 'dev.parquet'),
                'source_hashes': {str(p.relative_to(Path(__file__).parent)): file_sha(p)
                                  for p in Path(__file__).parent.rglob('*.py')}}
    atomic_json(root / 'receipt.json', manifest)
    return True


def evaluate(policy, tasks, path, manifest):
    before = policy.fingerprint()
    rows = journal(path)
    if [r['task_id'] for r in rows] != manifest['dev_ids'][:len(rows)] or any(r['parameters'] != before for r in rows):
        raise ContractError('Ablation evaluation journal identity changed')
    for public, gold in tasks[len(rows):]:
        with policy.task_cache():
            trace, _ = rollout(policy, public, 'validation', manifest['seed'], greedy=True, **manifest['rollout_options'])
        row = {'task_id': public['id'], 'parameters': before, **metrics(trace, public, gold),
               'total_tokens': sum(continuation_tokens(trace).values())}
        append_jsonl(path, row)
        rows.append(row)
    if policy.fingerprint() != before:
        raise ContractError('Ablation evaluation changed parameters')
    return rows


def run(root, diagnostic):
    import torch
    from train_speed_policy import TrainingSpeedPolicy

    if not (root / 'receipt.json').exists() and not prepare(root, diagnostic):
        return
    manifest = read(root / 'receipt.json')
    for name, sha in manifest['source_hashes'].items():
        if file_sha(Path(__file__).parent / name) != sha:
            raise ContractError('Ablation source changed')
    if file_sha(root / 'train.parquet') != manifest['train_sha256'] or file_sha(root / 'dev.parquet') != manifest['dev_sha256']:
        raise ContractError('Ablation data changed')
    checkpoint = Path(manifest['warmstart']['path'])
    if file_sha(checkpoint / 'state.pt') != manifest['warmstart']['sha256']:
        raise ContractError('Ablation warmstart changed')
    seed = manifest['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    policy = TrainingSpeedPolicy(manifest['model'])
    policy.fast_update = True
    train = [split_row(r) for r in pq.read_table(root / 'train.parquet').to_pylist()]
    dev = [split_row(r) for r in pq.read_table(root / 'dev.parquet').to_pylist()]
    load_checkpoint(checkpoint, policy, restore_random=False)
    policy.reset_reference('manager')
    actor = policy.fingerprint('actor')
    base = policy.frozen_base_signature()
    initial = evaluate(policy, dev, root / 'initial.jsonl', manifest)
    results, all_rows = {}, {'initial': initial}
    for arm, (sampling, objective) in ARMS.items():
        output = root / arm
        output.mkdir(exist_ok=True)
        identity = digest([manifest, arm])
        load_checkpoint(checkpoint, policy, restore_random=False)
        policy.reset_reference('manager')
        optimizer = {'manager': torch.optim.AdamW(policy.parameters('manager'), lr=1e-5, weight_decay=0.)}
        latest = output / 'checkpoints/latest.json'
        metrics_path = output / 'training.jsonl'
        if latest.exists():
            saved = load_checkpoint(output / 'checkpoints' / read(latest)['path'], policy, optimizer, identity)
            start = saved['step']
        else:
            start = 0
        records = journal(metrics_path)
        if len(records) < start or [r['task_id'] for r in records[:start]] != manifest['train_ids'][:start]:
            raise ContractError('Ablation checkpoint has an incomplete training journal')
        if len(records) > start:
            from scope.io import atomic_jsonl
            atomic_jsonl(output / f'uncommitted_after_{start:05d}_{time.time_ns()}.jsonl', records[start:])
            records = records[:start]
            atomic_jsonl(metrics_path, records)
        for i in range(start, len(train)):
            with policy.task_cache():
                weighted, record = step_data(policy, train[i], i + 1, 'alternating', 0, manifest['rollout_options'], 0.,
                    training_seed=seed, local_candidates=manifest['group_size'], local_continuation='greedy',
                    trajectory_objective='strict', local_objective=objective, local_candidate_sampling=sampling,
                    local_gap_stage_selection=True)
            update = policy.grpo_update(weighted, 'manager', optimizer['manager'], beta=.01)
            if policy.fingerprint('actor') != actor or policy.frozen_base_signature() != base:
                raise ContractError('Ablation changed the executor or backbone')
            local = record.get('local_group')
            m = local['metrics'] if local else {}
            row = {'step': i + 1, 'task_id': record['task_id'], 'update_kind': record['update_kind'],
                   'optimizer_step': update['optimizer_step'], 'sampling_tokens': record['sampling_tokens'],
                   'local_metrics': m, 'stage': record.get('local_stage')}
            if local:
                row['candidate_sampling_audit'] = [
                    {k: c['sample'][k] for k in ('text', 'choice_token_ids', 'completion_ids', 'old_logprobs', 'boundary', 'version')}
                    for c in local['candidates']]
            append_jsonl(metrics_path, row)
            records.append(row)
            if (i + 1) % 64 == 0 or i + 1 == len(train):
                save_checkpoint(output / 'checkpoints', i + 1, policy, optimizer, identity)
            atomic_json(root / 'status.json', {'phase': 'training', 'arm': arm, 'completed': i + 1, 'expected': len(train)})
        atomic_json(root / 'status.json', {'phase': 'evaluate', 'arm': arm})
        all_rows[arm] = evaluate(policy, dev, output / 'dev.jsonl', manifest)
        groups = [r['local_metrics'] for r in records if r['local_metrics']]
        tokens = Counter()
        for record in records:
            tokens.update(record['sampling_tokens'])
        results[arm] = {'trained': len(records), 'updates': sum(r['optimizer_step'] for r in records),
                        'local_groups': len(groups), 'nonzero_groups': sum(not m['task_zero_signal'] for m in groups),
                        'trainable_nonzero_groups': sum(m['trainable_nonzero_credit'] for m in groups),
                        'mean_unique_effects': float(np.mean([m['unique_effects'] for m in groups])) if groups else 0.,
                        'mean_unique_final_views': float(np.mean([m['unique_final_views'] for m in groups])) if groups else 0.,
                        'sampling_tokens': dict(tokens), 'actor_unchanged': True}
    pairs = {'sampling': ('independent_paired', 'distinct_paired'), 'credit': ('distinct_group', 'distinct_paired')}
    summary = {'complete': True, 'teacher_calls': 0, 'training': results, 'cells': {}, 'contrasts': {},
               'independent_dev_ids_verified': True, 'candidate_gate': manifest['candidate_gate']}
    for name, rows in all_rows.items():
        summary['cells'][name] = {'n': len(rows), 'em_count': sum(r['em'] for r in rows),
                                   'f1': float(np.mean([r['f1'] for r in rows])),
                                   'vs_initial': compare(initial, rows, 'em')}
    for name, (a, b) in pairs.items():
        summary['contrasts'][name] = {k: compare(all_rows[a], all_rows[b], k) for k in ('em', 'f1', 'support_visible_recall', 'total_tokens')}
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'status.json', {'phase': 'complete'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root, args.diagnostic.resolve())
        except Exception as error:
            atomic_json(root / 'error.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            atomic_json(root / 'status.json', {'phase': 'needs_attention'})
            raise


if __name__ == '__main__':
    main()
