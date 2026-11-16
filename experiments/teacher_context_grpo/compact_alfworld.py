"""Common actor prompt and measured one-decision public-evidence interventions."""
import copy

from public_memory_ablation import MemoryAblationRunner
from public_prefix_repair import cell, prefix_identity
from scope.alfworld import ALFWorldActions
from scope.alfworld_context import decision_card, historical_quote, treatment_candidates, treatment_stage
from scope.candidate_actions import candidate_actions, choice_payload
from scope.contracts import ContractError
from scope.io import digest
from scope.state_graph import compression_due


class CompactALFRunner(MemoryAblationRunner):
    def __init__(self, *args, compact=True, gated_manager=False, **kwargs):
        super().__init__(*args, public_history_only=True, **kwargs)
        self.compact = compact
        self.gated_manager = gated_manager
        self.intervention = None
        self.scan = False

    def _decision_card(self, atlas, working, goal):
        rows = decision_card(atlas, working, goal)
        if not self.cfg.get("alf_treatment_stages"):
            return rows, ()
        stage = treatment_stage(working.get("public_state", {}), goal)
        if not stage:
            return rows, ()
        priority = {slot: index for index, slot in enumerate(stage["slots"])}
        rows.sort(key=lambda r: priority.get((r["claim"]["entity"], r["claim"]["predicate"]), len(priority))
                  if "claim" in r else len(priority))
        return rows[:4] + [{"text": stage["text"], "source_refs": []}], stage["slots"]

    def _manager_request(self, atlas, observation, working, goal, budget, phase, audit=None):
        if not self.gated_manager:
            return super()._manager_request(atlas, observation, working, goal, budget, phase, audit)
        rows, slots = self._decision_card(atlas, working, goal)
        targets = [(row["claim"]["entity"], row["claim"]["predicate"]) for row in rows if "claim" in row]
        working.update(domain_decision_slots=True, decision_targets=list(dict.fromkeys([*slots, *targets]))[:4])
        recent = [record["text"] for record in atlas.all_evidence() if record["kind"] == "observation"][-4:]
        quote = historical_quote(atlas, working, rows, [observation, *recent])
        needed = quote and (working.get("interaction_feedback") or
                            any(row["text"].endswith("pending") for row in rows) or
                            quote["predicate"] in {"held", "container_open", "object_location"})
        if slots and quote:
            needed = (quote["entity"], quote["predicate"]) in slots
        limit = self.cfg.get("manager_max_input", 2048)
        menu = {}
        if budget.manager_tokens > 0 and budget.total_tokens > 16:
            if phase == "state" and needed and not working.get("state_lookup_used"):
                menu = candidate_actions(atlas, working, goal, phase, budget, self.tools, audit)
                menu = {label: op for label, op in menu.items() if op["operation"] == "FINISH" or
                        (op["operation"] == "RETRIEVE" and op.get("query") == quote["id"] and
                         (op.get("entity"), op.get("predicate")) == (quote["entity"], quote["predicate"]))}
            if phase == "context" and not working.get("summary") and compression_due(atlas, working):
                menu = candidate_actions(atlas, working, goal, phase, budget, self.tools, audit)
                menu = {label: op for label, op in menu.items() if op["operation"] in {"FINISH", "SUMMARIZE"}}
        reason, messages, room = "no_public_decision_gap", None, 0
        if len(menu) > 1:
            messages = choice_payload({"goal": goal, "phase": phase, "state": [r["text"] for r in rows],
                                       "historical_span": quote if phase == "state" else None}, menu)
            size = len(self.policy.encode(messages))
            room = min(budget.manager_tokens, budget.total_tokens - size)
            reason = None if size <= limit and room > 0 else "manager_input_budget"
        self.manager_audit = {"slots": [], "skip": reason, "retrieval_budget": budget.retrievals, "menu": menu}
        return messages, room, reason, menu

    def _base_actor_messages(self, atlas, working, goal, observation, history):
        window = self.cfg.get("paired_history_window")
        if window is not None:
            history = history[-window:]
        if self.gated_manager and working.get("summary") and self.tools.validator.summary_valid(
                atlas, working["summary"], goal):
            card, _ = self._decision_card(atlas, working, goal)
            card_ids = {row["claim"]["id"] for row in card if "claim" in row}
            protected = {claim["id"] for claim in atlas.nodes("Claim") if claim["predicate"] != "requirement"
                         and (claim.get("protected") or claim.get("conflicted") or claim["support_status"] == "pending")}
            if protected <= card_ids:
                history = history[-4:]
        return super()._base_actor_messages(atlas, working, goal, observation, history)

    @staticmethod
    def _insert(messages, card, quote, suffix):
        messages = copy.deepcopy(messages)
        text = messages[-1]["content"]
        if suffix and not text.endswith(suffix):
            raise ContractError("Action menu changed before evidence insertion")
        start = text[:-len(suffix)] if suffix else text
        extra = "\nDecision state (public):\n" + "\n".join(row["text"] for row in card) if card else ""
        if quote:
            extra += "\nHistorical public evidence: " + quote["text"]
        messages[-1]["content"] = start + extra + suffix
        return messages

    def _actor_messages(self, atlas, working, goal, observation, history):
        messages, view_hash, view_id = super()._actor_messages(atlas, working, goal, observation, history)
        if not (self.scan or self.intervention or self.compact) or not self.action_menu:
            return messages, view_hash, view_id
        turn = len(self.turn_audits) - 1
        base_messages = messages
        base_sha = digest(messages)
        limit = self.cfg.get("actor_max_input", 6144)
        card, _ = self._decision_card(atlas, working, goal)
        quote = historical_quote(atlas, working, card, [m["content"] for m in messages])
        suffix = "\nChoose one public action label only.\n" + "\n".join(
            f"{act['label']}: {act['command']}" for act in self.action_menu)
        with_card = self._insert(messages, card, None, suffix)
        with_quote = self._insert(messages, card, quote, suffix) if quote else None
        if self.scan and self.anchor is None and card and len(self.action_menu) > 1 and len(self.policy.encode(with_card)) <= limit:
            labels = [x["label"] for x in self.action_menu]
            prompts = {"history": messages, "card": with_card}
            if with_quote and len(self.policy.encode(with_quote)) <= limit:
                prompts["card_quote"] = with_quote
            choices = {}
            for name, prompt in prompts.items():
                action = self.policy.actor_choice(prompt, base_sha, labels, limit)
                choices[name] = action.text
                self.discovery_tokens += len(action.prompt_ids) + len(action.completion_ids)
            if self.cfg.get("paired_require_quote"):
                eligible = "card_quote" in choices
            else:
                eligible = any(label != choices["history"] for label in choices.values())
            if eligible:
                self.anchor = {"turn": turn, "prompt_hash": base_sha, "choices": choices,
                               "card": [row["text"] for row in card],
                               "quote": quote if "card_quote" in choices else None}
        if self.intervention and turn == self.intervention["turn"]:
            if base_sha != self.intervention["prompt_hash"]:
                raise ContractError("Replay reached a different prompt or menu")
            quote = self.intervention.get("quote") if self.intervention["mode"] == "card_quote" else None
            if quote and not self._quote_matches(atlas, quote):
                raise ContractError("Replay reached a different historical evidence span")
            messages = self._insert(messages, card, quote, suffix)
            self.reached_anchor = True
        elif self.compact:
            chosen = quote if self.gated_manager and quote and quote["id"] in working.get("selected_ids", []) else None
            messages = self._insert(messages, card, chosen, suffix)
            quote = chosen
        else:
            return messages, view_hash, view_id
        if len(self.policy.encode(messages)) > limit:
            if self.intervention:
                raise ContractError("Paired intervention exceeds the frozen actor budget")
            messages = with_card if len(self.policy.encode(with_card)) <= limit else base_messages
            if messages is base_messages:
                return base_messages, view_hash, view_id
            quote = None
        refs = [r["claim"]["id"] for r in card if "claim" in r]
        if quote:
            refs.append(quote["id"])
        saved = atlas.save_view(refs, None)
        return messages, digest(messages), saved["id"]

    @staticmethod
    def _quote_matches(atlas, quote):
        return atlas.evidence(quote["id"])["text"][quote["start"]:quote["end"]] == quote["text"]

    def trace(self, task_id, *, scan=False, intervention=None):
        self.scan, self.intervention = scan, intervention
        self.anchor, self.discovery_tokens, self.reached_anchor = None, 0, False
        outcome = self.run(task_id, "compact-alf:" + task_id, greedy=True, pair_seed=233)
        steps = [event for event in outcome.events if event["phase"] == "execute"]
        if intervention and not self.reached_anchor:
            raise ContractError("Continuation did not reach the paired intervention")
        return outcome, steps


class TreatmentALFActions(ALFWorldActions):
    def recovery_candidates(self, state, goal):
        return treatment_candidates(state, goal, super().recovery_candidates(state, goal))


def paired_trace(runner, task_id):
    baseline, original_steps = runner.trace(task_id, scan=True)
    anchor, discovery_tokens = runner.anchor, runner.discovery_tokens
    row = {"task_id": task_id, "baseline": baseline.metrics, "baseline_costs": baseline.costs,
           "discovery_tokens": discovery_tokens}
    if anchor is None:
        return {**row, "status": ("no_novel_historical_quote" if runner.cfg.get("paired_require_quote")
                                  else "no_eligible_divergence")}
    turn = anchor["turn"]
    branches = {"history": branch_cell(baseline, original_steps, turn)}
    if original_steps[turn]["public_action_audit"].get("selected_label") != anchor["choices"]["history"]:
        raise ContractError("Scanned history action differs from the executed action")
    for mode in [name for name in ("card", "card_quote") if name in anchor["choices"]]:
        result, steps = runner.trace(task_id, intervention={**anchor, "mode": mode})
        if prefix_identity(steps, turn) != prefix_identity(original_steps, turn):
            raise ContractError("Environment replay changed before the paired action")
        if steps[turn]["public_action_audit"].get("selected_label") != anchor["choices"][mode]:
            raise ContractError("Scanned intervention action differs from the replayed action")
        branches[mode] = branch_cell(result, steps, turn)
    return {**row, "status": "paired", "anchor": anchor, "branches": branches,
            "prefix_verified": True, "same_menu": True}


def branch_cell(outcome, steps, turn):
    return {**cell(outcome, steps, turn),
            "public_confirmed_effect": steps[turn].get("new_public_action_confirmation", False),
            "public_progress": steps[turn].get("progress")}
