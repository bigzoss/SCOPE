"""Compare public storefront and variant-aware checking with one frozen shopping SFT."""
import argparse
import fcntl
from pathlib import Path
import random
import traceback
from unittest.mock import patch

import numpy as np
import torch

from action_memory_dev import bootstrap
from experiments.memory_curriculum import shared_memory_eval as shared
from scope.contracts import ContractError
from scope.environments import ShopEnvironment, ShopPurchaseActions
from scope.evaluation import evaluate
from scope.io import atomic_json, canonical, digest, file_sha, read_jsonl
from scope.launcher import own_service
from scope.policy import HFPolicy
from scope.rollout import EpisodeRunner
from scope.shop_purchase import (read_panel, record_candidate_review, review_extended_purchase,
                                 review_for_purchase_prompt, review_purchase)


ARMS = ('public_storefront', 'purchase_panel', 'purchase_gate', 'verified_constraints')
VERIFIED_GATE = 'purchase_gate_verified'


class VerifiedShopActions(ShopPurchaseActions):
    def preflight(self, command, state, goal):
        conflict = super().preflight(command, state, goal)
        if conflict or command != "click[Buy Now]":
            return conflict
        review = review_extended_purchase(goal, state.get("purchase_panel"))
        for row in (review or {}).get("checks", []):
            if row["constraint"] in {"size", "material", "pack_quantity"} and row["status"] == "conflict":
                return f"公开购买表单的{row['constraint']}为 {row['observed']}，用户明确要求 {row['required']}。"
        return None


class CandidateReviewRunner(EpisodeRunner):
    def run(self, *args, **kwargs):
        self.candidate_audits = []
        outcome = super().run(*args, **kwargs)
        executed = (event for event in outcome.events if event["phase"] == "execute")
        for event, audit in zip(executed, self.candidate_audits):
            if audit is not None:
                event["public_candidate_review"] = audit
        return outcome

    def _actor_messages(self, atlas, working, goal, observation, history):
        messages, view_hash, view_id = super()._actor_messages(atlas, working, goal, observation, history)
        ledger = working.setdefault("shop_candidate_reviews", {})
        review = review_extended_purchase(goal, working.get("public_state", {}).get("purchase_panel"))
        prior_id = working.get("shop_review_id")
        current_id = (review["product_id"], review["selected"]) if review else None
        row = record_candidate_review(ledger, review, working["observation_id"], continuous=prior_id == current_id)
        working["shop_review_id"] = current_id
        self.candidate_audits.append(row)
        rejected = [value for value in ledger.values() if value["conflicts"]][-4:]
        recent = next((value for value in reversed(ledger.values()) if value["visits"] > 1), None)
        if not review and not rejected and not recent:
            return messages, view_hash, view_id
        details = {"current_review": review, "candidate_conflicts": rejected,
                   "unchanged_variant_revisited": row if row and row["visits"] > 1 else recent}
        note = ("\n仅在商品和已提交规格相同的情况下复用旧核对。重复查看且没有新证据时，调整规格、换候选或改搜索。"
                "待核验不等于冲突。\n" + canonical(details))
        previous = messages[-1]["content"]
        messages[-1]["content"] += note
        if len(self.policy.encode(messages)) > self.cfg.get("actor_max_input", 6144):
            messages[-1]["content"] = previous
            return messages, view_hash, view_id
        old_view = atlas.view(view_id)
        ids = {value["last_observation_id"] for value in rejected}
        if review:
            ids.add(working["observation_id"])
        if recent:
            ids.add(recent["last_observation_id"])
        saved = atlas.save_view(sorted(set(old_view["selected_ids"]) | ids), old_view["summary"])
        return messages, digest(messages), saved["id"]


def strip_purchase_panel(text):
    panel, match = read_panel(text)
    return text[:match.start()] if panel is not None else text


def purchase_card_is_due(panel, prior_command, warning):
    if panel is None:
        return False
    return bool(panel["selected"] or panel["unselected_fields"] == []
                or prior_command == "click[Buy Now]" or warning and "公开价格" in warning)


class PurchaseGateRunner(EpisodeRunner):
    """Preserve the storefront prompt before a public purchasable variant exists."""

    review = staticmethod(review_purchase)

    def run(self, *args, **kwargs):
        self.gate_audit = []
        outcome = super().run(*args, **kwargs)
        for event, shown in zip((event for event in outcome.events if event['phase'] == 'execute'), self.gate_audit):
            event['purchase_card_shown'] = shown
        return outcome

    def _actor_messages(self, atlas, working, goal, observation, history):
        cfg = self.cfg
        prior = history[-1].get("command") if history else None
        panel = working.get("public_state", {}).get("purchase_panel")
        shown = False
        try:
            self.cfg = {**cfg, "shop_purchase_flow": False}
            clean_history = [{**row, "observation": strip_purchase_panel(row["observation"])} for row in history]
            messages, view_hash, view_id = super()._actor_messages(
                atlas, working, goal, strip_purchase_panel(observation), clean_history)
        finally:
            self.cfg = cfg
        if purchase_card_is_due(panel, prior, working.get("warning")):
            review = self.review(goal, panel)
            context = {"规格": panel["selected"], "待选规格": panel["unselected_fields"],
                       "公开价格": panel["displayed_price_cny"],
                       "核对": [row for row in review["checks"] if row["constraint"] != "remaining_user_requirements"]}
            former = messages[-1]["content"]
            messages[-1]["content"] += ("\n购买前核对（待确认可继续查证，明确冲突应换规格或候选） "
                                        + canonical(context))
            if len(self.policy.encode(messages)) <= cfg.get("actor_max_input", 6144):
                view = atlas.view(view_id)
                view_id = atlas.save_view(sorted(set(view["selected_ids"]) | {working["observation_id"]}),
                                          view["summary"])["id"]
                view_hash, shown = digest(messages), True
            else:
                messages[-1]["content"] = former
        self.gate_audit.append(shown)
        return messages, view_hash, view_id


class VerifiedPurchaseGateRunner(PurchaseGateRunner):
    review = staticmethod(review_for_purchase_prompt)


def run(root, gpu_memory):
    receipt = shared.read(root / 'receipt.json')
    for name, expected in receipt['source_sha256'].items():
        if file_sha(root / 'source' / name) != expected:
            raise ContractError('Shop evaluation source differs')
    for name, expected in receipt['assets_sha256'].items():
        if file_sha(name) != expected:
            raise ContractError('Shop evaluation asset differs')
    config = shared.read(root / 'config.json')
    ids = receipt['task_ids']
    arms = tuple(receipt.get('arms', ARMS))
    if not ids or len(ids) != len(set(ids)):
        raise ContractError('Shop development task manifest mismatch')
    if not arms or len(set(arms)) != len(arms) or any(arm not in (*ARMS, VERIFIED_GATE) for arm in arms):
        raise ContractError('Shop evaluation configured unknown or duplicate conditions')
    seed = config.get('seed', 233)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(
        min(1., gpu_memory * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    model = HFPolicy(config['model']['path'], rank=config['model']['rank'], alpha=config['model']['alpha'],
                     gradient_checkpointing=False)
    frozen = model.fingerprint()
    by_arm, cells = {}, {}
    for arm in arms:
        output = root / arm
        output.mkdir(exist_ok=True)
        flow = arm != 'public_storefront'
        opt = {**config, 'runtime': {**config['runtime'], 'shop_purchase_flow': flow},
               'environment': {**config['environment'], 'base_url': f'http://127.0.0.1:{shared.port()}'}}
        env = ShopEnvironment(opt['environment']['base_url'], ids,
                              file_sha(opt['environment']['manifest']), purchase_flow=flow)
        if arm in {'verified_constraints', VERIFIED_GATE}:
            env.action_adapter = VerifiedShopActions()
        atomic_json(root / 'status.json', {'phase': 'evaluating', 'arm': arm, 'expected': len(ids)})
        with own_service(opt, output, True):
            try:
                env.health()
                runner = {'verified_constraints': CandidateReviewRunner,
                          'purchase_gate': PurchaseGateRunner,
                          VERIFIED_GATE: VerifiedPurchaseGateRunner}.get(arm, EpisodeRunner)
                with patch('scope.evaluation.EpisodeRunner', runner):
                    summary = evaluate(model, env, shared.FixedTasks(ids), opt, output, receipt['split'])
            finally:
                env.close()
        result = output / 'results.jsonl'
        rows = read_jsonl(result)
        if (not summary['complete'] or model.fingerprint() != frozen or not summary['parameters_unchanged_during_eval']
                or [row['task_id'] for row in rows] != ids or file_sha(result) != summary['results_sha256']):
            raise ContractError('Shop evaluation changed tasks or parameters')
        by_arm[arm] = rows
        buys = sum(row['metrics']['purchased'] for row in rows)
        wrong = sum(row['metrics']['purchased'] and not row['metrics']['success'] for row in rows)
        cells[arm] = {**{k:summary[k] for k in ('n', 'success_count', 'strict_success_rate', 'mean_total_tokens',
            'illegal_action_rate_per_turn', 'wrong_purchase_rate', 'public_interaction', 'results_sha256')},
            'purchases': buys, 'wrong_purchases': wrong, 'no_purchase': len(ids) - buys,
            'wrong_purchase_given_purchase': wrong / buys if buys else None,
            'purchase_card_turns': sum(e.get('purchase_card_shown', False) for row in rows for e in row['history'])}
    paired = {}
    ordered = [arm for arm in (*ARMS, VERIFIED_GATE) if arm in by_arm]
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            signal = bootstrap([row['metrics']['success'] for row in by_arm[left]],
                               [row['metrics']['success'] for row in by_arm[right]])
            signal['mean_token_delta'] = sum(b['costs']['charged_tokens'] - a['costs']['charged_tokens']
                                               for a, b in zip(by_arm[left], by_arm[right])) / len(ids)
            def first_command(row):
                return next((e.get('normalized_command') for e in row['history'] if e['phase'] == 'execute'), None)
            signal['same_first_command'] = sum(first_command(a) == first_command(b)
                                               for a, b in zip(by_arm[left], by_arm[right]))
            paired[f'{right}_vs_{left}'] = signal
    atomic_json(root / 'summary.json', {'complete': True, 'n': len(ids), 'split': receipt['split'], 'cells': cells,
                'paired': paired, 'target_updates': 0, 'teacher_calls': 0})
    atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, default=26)
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve(), args.memory_gib)
        except Exception as exc:
            atomic_json(args.root / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
