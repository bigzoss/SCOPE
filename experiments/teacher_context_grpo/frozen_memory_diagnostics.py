"""Independent HotpotQA checkpoint evaluation and frozen same-prefix memory probes."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scope.contracts import ContractError
from scope.hotpot import (NOOP, _model_messages, answer_from_view, answer_messages, answer_view,
                          rollout, score, split_row)
from scope.hotpot_intervention import _archive, continuation_tokens, coverage, public_support_inventory
from scope.hotpot_memory import contains, quote, sentences
from scope.hotpot_teacher_repair import format_compatible
from scope.io import atomic_json, append_jsonl, digest, file_sha
from scope.retrieval import terms


SEED = 233
EVAL_N = 512
PAIR = 'hotpot_teacher_v2_pair_20261113_v2'
FULL = 'hotpot_full_train_20261110.QhXsPtKo'
CHECKPOINTS = {
    'initial': 'new_teacher/checkpoints/step_00000',
    'manager_only': 'new_teacher/checkpoints/step_01000',
    'warmup_only': 'actor_warmup/sft/checkpoints/step_00064',
    'warmup_and_manager': 'new_teacher_actor_warmup/checkpoints/step_01000',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def journal(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def question_key(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def prepare(project, root):
    pair = project / 'runs' / PAIR
    full = project / 'runs' / FULL
    if root.exists():
        raise ContractError('Choose a fresh diagnostic directory')
    source_dev = full / 'dataset/validation-00000-of-00001.parquet'
    dev = pq.read_table(source_dev).to_pylist()
    dev_ids = {row['id'] for row in dev}
    excluded_ids, excluded_questions, exclusion_files = set(), set(), {}
    for run in (project / 'runs').glob('hotpot*'):
        for path in run.rglob('*.json'):
            name = path.name.lower()
            if not any(key in name for key in ('receipt', 'manifest', 'selection', 'split')):
                continue
            if any(part in {'source', 'teacher_cache', 'updates'} for part in path.relative_to(run).parts):
                continue
            record = read(path)
            matched = dev_ids.intersection(strings(record))
            if matched:
                excluded_ids.update(matched)
                exclusion_files[str(path)] = file_sha(path)
        for path in run.rglob('*.parquet'):
            if path.parent == full / 'dataset' and path == source_dev:
                continue
            if not any(k in path.name.lower() for k in ('dev', 'valid', 'train')):
                continue
            cols = set(pq.ParquetFile(path).schema_arrow.names)
            if not {'id', 'question'} <= cols:
                continue
            rows = pq.read_table(path, columns=['id', 'question']).to_pylist()
            excluded_ids.update(dev_ids.intersection(row['id'] for row in rows))
            excluded_questions.update(question_key(row['question']) for row in rows)
            exclusion_files[str(path)] = file_sha(path)
    # Match duplicate normalized questions for previously recorded evaluation IDs.
    excluded_questions.update(question_key(row['question']) for row in dev if row['id'] in excluded_ids)
    eligible, used = [], set()
    for row in sorted(dev, key=lambda r: digest([SEED, 'independent-dev', r['id']])):
        key = question_key(row['question'])
        if row['id'] in excluded_ids or key in excluded_questions or key in used:
            continue
        eligible.append(row)
        used.add(key)
    if len(eligible) < EVAL_N:
        raise ContractError('Independent development pool is too small')
    selected = eligible[:EVAL_N]

    # Diagnostic strata fixed using existing source outcomes only.
    pools = {k: [] for k in ('retrieve_missing', 'retrieve_full', 'compress_correct', 'compress_failed')}
    source_arm = pair / 'new_teacher_actor_warmup'
    source_ids = read(pair / 'receipt.json')['train_ids']
    for path in sorted((source_arm / 'updates').glob('step_*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            record = json.load(f)
        primary = record['primary'][0]
        trace, scored = primary['trace'], primary['score']
        if source_ids[record['step'] - 1] != trace['task_id']:
            raise ContractError('Source trajectory identity differs from the training manifest')
        if not scored['em'] and any(o['operation'] == 'RETRIEVE' for a in trace['actions'][3:] for o in a['menu'].values()):
            pools['retrieve_full' if scored['support_visible_recall'] == 1 else 'retrieve_missing'].append((trace, path))
        if any(o.get('strategy') == 'preserve_public_relations' for o in trace['actions'][2]['menu'].values()):
            pools['compress_correct' if scored['em'] else 'compress_failed'].append((trace, path))
    prefixes, used = [], set()
    for stratum, n in [('retrieve_missing', 48), ('retrieve_full', 16), ('compress_correct', 32), ('compress_failed', 32)]:
        ranked = sorted(pools[stratum], key=lambda pair: digest([SEED, stratum, pair[0]['task_id']]))
        available = [(trace, path) for trace, path in ranked if trace['task_id'] not in used][:n]
        if len(available) != n:
            raise ContractError(f'Insufficient source prefixes for {stratum}')
        for trace, path in available:
            task = trace['task_id']
            used.add(task)
            prefixes.append({'task_id': task, 'stratum': stratum, 'step_path': str(path),
                             'step_sha256': file_sha(path), 'seed': trace['seed']})
    training = pq.read_table(pair / 'data/train.parquet').to_pylist()
    source_rows = {r['id']: r for r in training if r['id'] in used}
    if len(source_rows) != 128 or used & {r['id'] for r in selected}:
        raise ContractError('Diagnostic IDs missing or shared with evaluation')

    root.mkdir(parents=True)
    snapshot = root / 'source'
    shutil.copytree(pair / 'source', snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__).resolve(), snapshot / 'frozen_memory_diagnostics.py')
    pq.write_table(pa.Table.from_pylist(selected), root / 'holdout.parquet')
    pq.write_table(pa.Table.from_pylist([source_rows[x['task_id']] for x in prefixes]), root / 'prefix_tasks.parquet')
    # Reserve a disjoint development split before looking at diagnostic results.
    pq.write_table(pa.Table.from_pylist(eligible[EVAL_N:2 * EVAL_N]), root / 'reserved_dev.parquet')
    models = {k: {'path': str(pair / v), 'sha256': file_sha(pair / v / 'state.pt')} for k, v in CHECKPOINTS.items()}
    pair_receipt = read(pair / 'receipt.json')
    manifest = {'seed': SEED, 'model': pair_receipt['model'], 'checkpoints': models,
                'rollout_options': read(source_arm / 'receipt.json')['rollout_options'],
                'official_dev_sha256': file_sha(source_dev), 'holdout_ids': [r['id'] for r in selected],
                'excluded_recorded_dev_ids': len(excluded_ids), 'eligible_dev_n': len(eligible),
                'exclusion_files': exclusion_files, 'prefixes': prefixes, 'teacher_calls': 0,
                'source_hashes': {str(p.relative_to(snapshot)): file_sha(p) for p in snapshot.rglob('*.py')},
                'data_hashes': {p.name: file_sha(p) for p in root.glob('*.parquet')}}
    atomic_json(root / 'receipt.json', manifest)
    atomic_json(root / 'status.json', {'phase': 'prepared'})


def metrics(trace, public, gold):
    result = score(trace, gold)
    result['diagnosis'] = ('correct' if result['em'] else
                           'format_compatible_review' if format_compatible(trace['answer'], gold['answer']) else
                           'support_complete_wrong' if result['support_visible_recall'] == 1 else
                           'source_missing' if public_support_inventory(public, gold)['outside_public'] else
                           'support_not_delivered')
    result.update(answer=trace['answer'], view_sha256=digest(trace['answer_view']),
                  input_tokens=len(trace['answer_sample']['prompt_ids']),
                  output_tokens=len(trace['answer_sample']['completion_ids']))
    return result


def compare(before, after, key):
    if len(before) != len(after) or not before:
        raise ContractError('Invalid paired comparison')
    delta = np.asarray([b[key] - a[key] for a, b in zip(before, after, strict=True)])
    bootstrap = np.random.default_rng(SEED).choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
    return {'n': len(delta), 'delta': float(delta.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum())}


def diagnostic(policy, public, gold, row, opts):
    # Rebuild the exact historical prefix under the current frozen adapter versions.
    path = Path(row['step_path'])
    if file_sha(path) != row['step_sha256']:
        raise ContractError('Historical source prefix changed')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        historical = json.load(f)['primary'][0]['trace']
    def prefix_chooser(messages, menu, public_options, working_sources):
        stage = json.loads(messages[-1]['content'])['stage']
        action = next(a for a in historical['actions'] if a['stage'] == stage)
        if menu != action['menu']:
            raise ContractError('Frozen diagnostic reconstructed a different historical menu')
        return {'choice': action['choice'], 'origin': 'archived_student_prefix'}

    stage = 2 if row['stratum'].startswith('compress') else 3
    anchor, _ = rollout(policy, public, 'rl_train', row['seed'], chooser=prefix_chooser, **opts)
    action = anchor['actions'][stage]
    archive = _archive(anchor, public)
    anchor_view = action['working_before']
    # Pick the current frozen manager's action at the same historical boundary.
    messages = _model_messages(action['stage'], action['menu'], public, archive, anchor_view,
                               opts['sentence_candidates'], opts['shared_interface'], action['state_claims_before'])
    sample = policy.sample_choice(messages, action['stage'], action['boundary'], list(action['menu']), 4096, True)
    control = next(k for k, v in action['menu'].items() if v['operation'] == NOOP[action['stage']])
    choices = {'keep': control, 'current': sample.text}
    if stage == 2:
        for label, option in action['menu'].items():
            if option.get('strategy') == 'preserve_public_relations':
                choices['protected_compression'] = label
            elif option['operation'] == 'COMPRESS':
                choices.setdefault('standard_compression', label)
    else:
        offered = [(k, v['evidence']) for k, v in action['menu'].items() if v['operation'] == 'RETRIEVE']
        query = set(terms(public['question']))
        if offered:
            choices['public_menu_retrieval'] = max(offered, key=lambda pair: (
                len(query & set(terms(quote(archive, pair[1])))),
                len(query & set(terms(archive[pair[1]['id']]['title']))), -ord(pair[0])))[0]
    cells, traces, by_label = {}, {}, {}
    for name, label in choices.items():
        if label not in by_label:
            by_label[label] = rollout(policy, public, 'rl_train', row['seed'], factual=anchor,
                                      intervention=stage, intervention_label=label, greedy=True, **opts)[0]
        trace = by_label[label]
        traces[name] = trace
        cells[name] = {**metrics(trace, public, gold), 'action': action['menu'][label]['operation'], 'label': label,
                       'continuation_tokens': sum(continuation_tokens(trace, stage).values()),
                       'boundary_decision_tokens': len(sample.prompt_ids) + len(sample.completion_ids)}
    if stage == 3:
        base = traces['keep']
        shown = [{k: r[k] for k in ('id', 'start', 'end')} for r in base['answer_view']]
        query = set(terms(public['question']))
        offered_refs = [v['evidence'] for v in action['menu'].values() if v.get('evidence')]
        visible = [(r['id'], r['start'], r['end']) for r in shown]
        public_refs, oracle_refs = [], []
        supports = set(map(tuple, gold['supporting_facts']))
        for record in archive.values():
            for ref in sentences(record):
                if any(i == ref['id'] and a <= ref['start'] and b >= ref['end'] for i, a, b in visible):
                    continue
                i = record['sentence_offsets'].index((ref['start'], ref['end']))
                if (record['title'], i) in supports:
                    oracle_refs.append(ref)
                if any(x['id'] == ref['id'] and x['start'] <= ref['start'] and x['end'] >= ref['end'] for x in offered_refs):
                    continue
                text = quote(archive, ref)
                bridge = any(contains(quote(archive, prior), record['title']) for prior in anchor_view if prior['id'] != ref['id'])
                relevance = len(query & set(terms(text))) + len(query & set(terms(record['title'])))
                if relevance or bridge:
                    public_refs.append((2 * int(bridge) + relevance, digest(ref), ref))
        public_refs.sort(key=lambda item: (-item[0], item[1]))
        for name, refs in [('public_span_recovery', [x[2] for x in public_refs[:1]]), ('gold_span_upper_bound', oracle_refs)]:
            if not refs:
                continue
            view = answer_view(archive, shown + refs)
            if len(policy.encode(answer_messages(public['question'], view))) > 4096:
                continue
            answer, reader = answer_from_view(policy, public['question'], view, digest(view))
            result = {**base, 'answer': answer, 'answer_view': view, 'answer_sample': vars(reader)}
            cells[name] = {**metrics(result, public, gold), 'action': 'terminal_view_recovery',
                           'refs_added': refs, 'diagnostic_only': True}
    return {'task_id': public['id'], 'stratum': row['stratum'], 'stage': action['stage'],
            'boundary': action['boundary'], 'historical_coverage': coverage(anchor, public, gold), 'cells': cells}


def evaluate(root):
    import torch
    from scope.checkpoint import load_checkpoint
    from train_speed_policy import TrainingSpeedPolicy

    manifest = read(root / 'receipt.json')
    for name, sha in manifest['source_hashes'].items():
        if file_sha(root / 'source' / name) != sha:
            raise ContractError('Frozen experiment source hash changed')
    for name, sha in manifest['data_hashes'].items():
        if file_sha(root / name) != sha:
            raise ContractError('Frozen experiment data hash changed')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    policy = TrainingSpeedPolicy(manifest['model'], gradient_checkpointing=False)
    options = manifest['rollout_options']
    holdout = [split_row(r) for r in pq.read_table(root / 'holdout.parquet').to_pylist()]
    data, fingerprints = {}, {}
    for name, record in manifest['checkpoints'].items():
        checkpoint = Path(record['path'])
        if file_sha(checkpoint / 'state.pt') != record['sha256']:
            raise ContractError('Checkpoint hash changed')
        load_checkpoint(checkpoint, policy, restore_random=False)
        before = policy.fingerprint()
        fingerprints[name] = {'manager': policy.fingerprint('manager'), 'actor': policy.fingerprint('actor')}
        output = root / f'{name}.jsonl'
        rows = journal(output)
        if [r['task_id'] for r in rows] != manifest['holdout_ids'][:len(rows)] or any(r['checkpoint'] != record['sha256'] for r in rows):
            raise ContractError('Holdout journal does not match its frozen checkpoint or ordered manifest')
        for public, gold in holdout[len(rows):]:
            with policy.task_cache():
                trace, _ = rollout(policy, public, 'validation', SEED, greedy=True, **options)
            result = {'task_id': public['id'], 'checkpoint': record['sha256'], **metrics(trace, public, gold),
                      'coverage': coverage(trace, public, gold), 'total_tokens': sum(continuation_tokens(trace).values())}
            append_jsonl(output, result)
            rows.append(result)
            atomic_json(root / 'status.json', {'phase': 'holdout', 'condition': name, 'completed': len(rows), 'expected': EVAL_N})
        if policy.fingerprint() != before:
            raise ContractError('Checkpoint changed during frozen evaluation')
        data[name] = rows
    if fingerprints['initial']['manager'] != fingerprints['warmup_only']['manager']:
        raise ContractError('Actor-only comparison has unequal initial managers')
    if [r['view_sha256'] for r in data['initial']] != [r['view_sha256'] for r in data['warmup_only']]:
        raise ContractError('Actor-only comparison changed visible evidence')
    pairs = {'warmup_only': ('initial', 'warmup_only'), 'manager_only': ('initial', 'manager_only'),
             'manager_after_warmup': ('warmup_only', 'warmup_and_manager'), 'final_difference': ('manager_only', 'warmup_and_manager')}
    holdout_report = {'n': EVAL_N, 'parameters_unchanged': True, 'actor_only_views_identical': True,
                      'fingerprints': fingerprints, 'cells': {}, 'contrasts': {}}
    for name, rows in data.items():
        holdout_report['cells'][name] = {'em_count': sum(r['em'] for r in rows),
            **{k: float(np.mean([r[k] for r in rows])) for k in ['em', 'f1', 'support_visible_recall', 'total_tokens']},
            'diagnoses': dict(Counter(r['diagnosis'] for r in rows)), 'sha256': file_sha(root / f'{name}.jsonl')}
    for name, (a, b) in pairs.items():
        holdout_report['contrasts'][name] = {k: compare(data[a], data[b], k) for k in ('em', 'f1', 'support_visible_recall')}
    atomic_json(root / 'holdout_summary.json', holdout_report)

    load_checkpoint(Path(manifest['checkpoints']['warmup_only']['path']), policy, restore_random=False)
    before = policy.fingerprint()
    output = root / 'prefix_diagnostics.jsonl'
    probes = journal(output)
    if [r['task_id'] for r in probes] != [x['task_id'] for x in manifest['prefixes'][:len(probes)]]:
        raise ContractError('Prefix journal order changed')
    tasks = pq.read_table(root / 'prefix_tasks.parquet').to_pylist()
    for index in range(len(probes), len(tasks)):
        public, gold = split_row(tasks[index])
        with policy.task_cache():
            result = diagnostic(policy, public, gold, manifest['prefixes'][index], options)
        append_jsonl(output, result)
        probes.append(result)
        atomic_json(root / 'status.json', {'phase': 'prefix_diagnostics', 'completed': len(probes), 'expected': 128})
    if policy.fingerprint() != before or len(probes) != 128:
        raise ContractError('Frozen diagnostics changed parameters or lack prefixes')
    diagnoses = {}
    for name in ['current', 'public_menu_retrieval', 'public_span_recovery', 'gold_span_upper_bound',
                 'standard_compression', 'protected_compression']:
        for family in ('retrieve', 'compress'):
            matched = [r['cells'] for r in probes if r['stratum'].startswith(family) and name in r['cells']]
            if not matched:
                continue
            a, b = [r['keep'] for r in matched], [r[name] for r in matched]
            diagnoses[f'{family}:{name}'] = {k: compare(a, b, k) for k in ('em', 'f1', 'support_visible_recall')}
            if 'continuation_tokens' in b[0]:
                diagnoses[f'{family}:{name}']['continuation_tokens'] = compare(a, b, 'continuation_tokens')
            diagnoses[f'{family}:{name}']['reader_input_tokens'] = compare(a, b, 'input_tokens')
    atomic_json(root / 'summary.json', {'complete': True, 'holdout': holdout_report, 'prefix_n': len(probes),
        'prefix_strata': dict(Counter(x['stratum'] for x in probes)), 'prefix_contrasts_vs_keep': diagnoses,
        'prefix_sha256': file_sha(output), 'parameters_unchanged': True, 'teacher_calls': 0})
    atomic_json(root / 'status.json', {'phase': 'complete'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args.project.resolve(), root)
        return
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            evaluate(root)
        except Exception as error:
            atomic_json(root / 'error.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            atomic_json(root / 'status.json', {'phase': 'needs_attention'})
            raise


if __name__ == '__main__':
    main()
