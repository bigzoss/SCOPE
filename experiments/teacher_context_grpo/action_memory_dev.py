"""Frozen public-action and recovery development comparisons on native environments."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
import fcntl
import json
import re
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch

from experiments.memory_curriculum import native_target_eval as native
from experiments.memory_curriculum import shared_memory_eval as shared
from scope.alfworld import ALFWorldEnvironment
from scope.checkpoint import load_checkpoint
from scope.config import DEFAULT_RUNTIME
from scope.contracts import ContractError, Sample
from scope.decision_evidence import review
from scope.evaluation import evaluate
from scope.io import atomic_json, digest, file_sha
from scope.recovery import prepare_recovery
from scope.rollout import EpisodeRunner


ARMS = ('current_interface', 'public_candidates', 'public_candidates_recovery')
SEED = 233
RETRIEVAL = {'RETRIEVE', 'FETCH_EVIDENCE'}


class DevBabyAI(native.BabyAI):
    def __init__(self, run, specs):
        super().__init__(run)
        from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
        self.colors, self.objects = IDX_TO_COLOR, IDX_TO_OBJECT
        self.tasks = {x['id']: x for x in specs}


class DevPolicy:
    """Keep the standard model API and score explicit labels on the actor adapter."""
    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def actor_choice(self, messages, boundary, labels, limit):
        ids = self.encode(messages)
        if len(ids) > limit:
            raise ContractError('Public action menu exceeds the actor input limit')
        choices = self.choice_ids(labels)
        sample = Sample('actor', 'execute', ids, [choices[0]], [], '', self.versions['actor'],
                        boundary, 'choice', choice_token_ids=choices)
        with self.torch.no_grad():
            probabilities = self.choice_logprobs(sample, 'actor')
            index = int(probabilities.argmax())
        sample.text = labels[index]
        sample.completion_ids = [choices[index]]
        sample.old_logprobs = [float(probabilities[index])]
        return sample


def history_slots(runner, atlas, working, goal):
    """Inventory of evidence available prior to the decision, scored without labels."""
    extractor = getattr(runner.tools.validator, 'public_facts', None)
    rows = []
    for slot in review(atlas, working, goal):
        key = slot['entity'], slot['predicate']
        supported = [ref['id'] for ref in slot.get('evidence_refs', [])]
        raw, extracted = [], []
        entity = slot['entity'].split(':', 1)[-1]
        for record in atlas.all_evidence():
            if (record['kind'] != 'observation' or record.get('terminal')
                    or record['revision'] < slot['min_revision']):
                continue
            if re.search(r'(?<!\w)' + re.escape(entity) + r'(?!\w)', record['text'], re.I):
                raw.append(record['id'])
            if record['id'] not in runner.fact_cache:
                runner.fact_cache[record['id']] = {(f.entity, f.predicate) for f in extractor(record)
                    if f.status == 'supported'} if extractor else set()
            if key in runner.fact_cache[record['id']]:
                extracted.append(record['id'])
        rows.append({'entity': slot['entity'], 'predicate': slot['predicate'], 'status': slot['status'],
                     'min_revision': slot['min_revision'], 'support_ids': supported,
                     'public_entity_mentions': raw[-4:], 'extracted_support_ids': extracted[-4:]})
    return rows


class AuditedRunner(EpisodeRunner):
    def __init__(self, *args, domain, arm, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain, self.arm = domain, arm
        self.live_atlas = None
        self.live_working = None
        self.live_goal = None
        self.action_menu = []
        self.turn_audits = []
        self.fact_cache = {}

    def _manager_request(self, atlas, observation, working, goal, budget, phase, audit=None):
        self.live_atlas, self.live_working, self.live_goal = atlas, working, goal
        if self.domain == 'alfworld' and self.cfg.get('decision_evidence'):
            # Count relevant history for every condition, including the legacy
            # branch whose policy input omits recovery targets.
            audit_working = dict(working, decision_recovery=True,
                decision_targets=self.public_projection.decision_targets(working.get('public_state', {}), goal))
            slots = history_slots(self, atlas, audit_working, goal)
        else:
            slots = []
        messages, room, skipped, menu = super()._manager_request(atlas, observation, working, goal, budget, phase, audit)
        self.manager_audit = {'slots': slots, 'skip': skipped, 'retrieval_budget': budget.retrievals,
            'menu': {label: {k: option[k] for k in ('operation', 'entity', 'predicate', 'source',
                       'query', 'evidence_id', 'start', 'end') if k in option} for label, option in (menu or {}).items()}}
        return messages, room, skipped, menu

    def _manager(self, *args, **kwargs):
        sample, result = super()._manager(*args, **kwargs)
        audit = self.manager_audit
        audit['selected'] = result.get('operation')
        audit['retrieval_offered'] = any(v['operation'] in RETRIEVAL for v in audit['menu'].values())
        audit['selected_finish_with_retrieval'] = audit['retrieval_offered'] and sample is not None and result.get('operation') == 'FINISH'
        audit['retrieval_slots_offered'] = [list(slot) for slot in dict.fromkeys(
            (v['entity'], v['predicate']) for v in audit['menu'].values()
            if v['operation'] in RETRIEVAL and v.get('entity') and v.get('predicate'))]
        result['decision_audit'] = audit
        return sample, result

    def _base_actor_messages(self, atlas, working, goal, observation, history):
        return super()._actor_messages(atlas, working, goal, observation, history)

    def _menu_order_key(self, atlas, working, goal, observation, history, command):
        return digest([atlas.scope.key, atlas.revision, command])

    def _actor_messages(self, atlas, working, goal, observation, history):
        messages, view_hash, view_id = self._base_actor_messages(atlas, working, goal, observation, history)
        self.action_menu = []
        if self.arm != 'current_interface':
            provider = getattr(self.env, 'public_action_candidates', None)
            if provider is not None:
                self.action_menu = provider(atlas, working, goal, observation, history)
            elif self.domain == 'babyai':
                self.action_menu = [{'command': command, 'reason': 'public_action_space', 'evidence_ids': []}
                                    for command in self.env.action_names]
                if self.arm == 'public_candidates_recovery':
                    # Suppress only a command repeated twice with identical public observations.
                    blocked = set()
                    for old in history[-2:]:
                        command = old.get('command')
                        if command and sum(x.get('command') == command and x.get('observation') == observation for x in history[-2:]) == 2:
                            blocked.add(command)
                    self.action_menu = [x for x in self.action_menu if x['command'] not in blocked]
            else:
                shadow = copy.deepcopy(working)
                # Candidate-only and recovery use the same public grounding logic.
                if self.arm == 'public_candidates':
                    shadow['recovery_attempts'] = []
                plan = prepare_recovery(self.public_projection, atlas, shadow, goal)
                choices = {x['command']: x for x in plan['candidates']}
                for command in ('look', 'inventory'):
                    if command not in plan['blocked_commands']:
                        choices.setdefault(command, {'command': command, 'reason': 'public_check', 'evidence_ids': []})
                self.action_menu = list(choices.values())[:8]
            if self.action_menu:
                ordered = sorted(self.action_menu, key=lambda x: self._menu_order_key(
                    atlas, working, goal, observation, history, x['command']))
                self.action_menu = [{**x, 'label': chr(65+i)} for i, x in enumerate(ordered)]
                suffix = '\nChoose one public action label only.\n' + '\n'.join(
                    f"{x['label']}: {x['command']}" for x in self.action_menu)
                messages = copy.deepcopy(messages)
                messages[-1]['content'] += suffix
                if len(self.policy.encode(messages)) > self.cfg.get('actor_max_input', 6144):
                    messages[-1]['content'] = messages[-1]['content'][:-len(suffix)]
                    self.action_menu = []
        self.turn_audits.append({'action_menu': copy.deepcopy(self.action_menu), 'atlas_revision': atlas.revision})
        return messages, digest([view_hash, self.action_menu]) if self.action_menu else view_hash, view_id

    def run(self, *args, **kwargs):
        self.turn_audits = []
        self.fact_cache = {}
        normal_sample = self.policy.sample

        def sample(messages, role, phase, boundary, max_input, max_new, greedy=False):
            if role != 'actor' or phase != 'execute' or not self.action_menu:
                return normal_sample(messages, role, phase, boundary, max_input, max_new, greedy)
            result = self.policy.actor_choice(messages, boundary, [x['label'] for x in self.action_menu], max_input)
            chosen = next(x for x in self.action_menu if x['label'] == result.text)
            self.turn_audits[-1]['selected_label'] = result.text
            result.text = 'Action: ' + chosen['command']
            if chosen['evidence_ids']:
                self.turn_audits[-1]['dispatcher_candidate_evidence_ids'] = chosen['evidence_ids']
            return result

        with patch.object(self.policy, 'sample', sample):
            outcome = super().run(*args, **kwargs)
        actor_events = [e for e in outcome.events if e['phase'] == 'execute']
        for step, audit in zip(actor_events, self.turn_audits):
            step['public_action_audit'] = audit
        return outcome


def bootstrap(left, right):
    delta = np.array([b-a for a, b in zip(left, right, strict=True)])
    values = np.random.default_rng(SEED).choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
    return {'delta': float(delta.mean()), 'ci95': np.quantile(values, [.025, .975]).tolist()}


def analyze(path):
    rows = [json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines()]
    actions, memory = Counter(), Counter()
    for row in rows:
        for step in row['history']:
            if step['phase'] != 'execute':
                continue
            actions[step['action_status']] += 1
            actions['sent'] += step['sent'] is True
            actions['turns'] += 1
            actions['public_state_changed'] += step.get('progress') == 'state_changed'
            actions['post_action_confirmation'] += step.get('public_confirmation', {}).get('status') == 'confirmed'
            memory['steps_with_retrieved_evidence_in_view'] += bool(step.get('retrieved_evidence_in_view'))
            memory['steps_citing_retrieved_evidence'] += bool(set(step.get('retrieved_evidence_in_view', [])) & set(step.get('cited_evidence_ids', [])))
        for event in row['manager_operations']:
            audit = event['result'].get('decision_audit', {})
            memory['decisions'] += 1
            memory['no_menu_candidate'] += audit.get('skip') == 'no_public_candidate'
            memory['any_extracted_slot_support'] += any(s['extracted_support_ids'] for s in audit.get('slots', []))
            memory['retrieval_offered'] += bool(audit.get('retrieval_offered'))
            memory['finish_with_retrieval_available'] += bool(audit.get('selected_finish_with_retrieval'))
            memory['retrieval_selected'] += audit.get('selected') in RETRIEVAL
    return rows, {'actions': dict(actions), 'memory': dict(memory)}


def run(root):
    m = shared.read(root / 'receipt.json')
    for file, sha in m['source_hashes'].items():
        if file_sha(root / 'source' / file) != sha:
            raise ContractError('Action development snapshot changed')
    for file, sha in m['assets'].items():
        if file_sha(file) != sha:
            raise ContractError('Action development asset changed')
    checkpoint = Path(m['checkpoint']['path'])
    if file_sha(checkpoint / 'state.pt') != m['checkpoint']['sha256']:
        raise ContractError('Actor checkpoint changed')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    from scope.policy import HFPolicy
    inner = HFPolicy(m['model'], gradient_checkpointing=False)
    load_checkpoint(checkpoint, inner, restore_random=False)
    policy = DevPolicy(inner)
    weights = policy.fingerprint()
    shared.BRIDGE = root / 'serve_alfworld.py'
    report = {}
    for domain in ('alfworld', 'babyai'):
        domain_tasks = m['tasks'][domain]
        ids = [x['id'] for x in domain_tasks]
        registry = shared.FixedTasks(ids)
        cells = {}
        cell_rows = {}
        for arm in ARMS:
            out = root / domain / arm
            out.mkdir(parents=True, exist_ok=True)
            runtime = {**DEFAULT_RUNTIME, 'state_graph': True, 'memory_choice': True, 'shared_memory': True,
                       'decision_evidence': True, 'action_mode': 'grounded' if domain == 'alfworld' else 'normalize',
                       'loop_policy': 'state', 'max_environment_steps': 50 if domain == 'alfworld' else 30,
                       'max_turns': 100 if domain == 'alfworld' else 60, 'total_tokens': 850000,
                       'decision_recovery': domain == 'alfworld' and arm == 'public_candidates_recovery'}
            config = {'environment': {'kind': 'alfworld_http' if domain == 'alfworld' else domain}, 'runtime': runtime,
                      'evaluation': {'seed': SEED, 'sequence_length': 1}}
            cell_receipt = {'domain': domain, 'arm': arm, 'tasks': ids, 'runtime': runtime, 'seed': SEED,
                            'weights': weights, 'target_updates': 0, 'teacher_calls': 0,
                            'experiment_receipt_sha256': file_sha(root / 'receipt.json')}
            path = out / 'receipt.json'
            if path.exists() and shared.read(path) != cell_receipt:
                raise ContractError('Development arm changed since collection')
            atomic_json(path, cell_receipt)
            atomic_json(root / 'status.json', {'phase': 'evaluating', 'domain': domain, 'arm': arm, 'expected': len(ids)})
            if domain == 'alfworld':
                endpoint = f'http://127.0.0.1:{shared.port()}'
                alf_manifest = root / 'alfworld_dev_manifest.json'
                config['environment'].update(base_url=endpoint, manifest=str(alf_manifest))
                env = ALFWorldEnvironment(endpoint, ids, alf_manifest)
                service = shared.alf_service(config, out)
            else:
                env = DevBabyAI(root, domain_tasks)
                service = nullcontext()
            try:
                with service, patch('scope.evaluation.EpisodeRunner', lambda *a, **kw: AuditedRunner(*a, domain=domain, arm=arm, **kw)):
                    if domain == 'alfworld':
                        env.health()
                    summary = evaluate(policy, env, registry, config, out, 'dev')
            finally:
                env.close()
            if policy.fingerprint() != weights or not summary['complete']:
                raise ContractError('Development evaluation must finish with frozen weights')
            rows, diagnostics = analyze(out)
            if len(rows) != len(ids) or {x['task_id'] for x in rows} != set(ids):
                raise ContractError('Unexpected development task IDs')
            cell_rows[arm] = {x['task_id']: x for x in rows}
            cells[arm] = {**diagnostics, **{k: summary[k] for k in ('n', 'success_count', 'strict_success_rate',
                        'mean_loose_reward', 'mean_total_tokens', 'results_sha256')}}
        comparisons = {}
        for before, after in zip(ARMS, ARMS[1:]):
            comparisons[after] = {'versus': before, 'strict_success': bootstrap(
                [cell_rows[before][x]['metrics']['success'] for x in ids],
                [cell_rows[after][x]['metrics']['success'] for x in ids])}
        report[domain] = {'cells': cells, 'paired': comparisons}
        atomic_json(root / 'summary.json', {'complete': len(report) == 2, 'teacher_calls': 0,
                                           'parameters_unchanged': True, 'domains': report})
    atomic_json(root / 'status.json', {'phase': 'complete'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as exc:
            atomic_json(root / 'error.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(root / 'status.json', {'phase': 'needs_attention'})
            raise


if __name__ == '__main__':
    main()
