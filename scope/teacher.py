"""Offline, public-prefix supervision for the SCOPE memory manager."""
from collections import Counter
import copy
import json
import re
from pathlib import Path

from .atlas import StateAtlas
from .contracts import ContractError
from .io import canonical, digest, file_sha
from .state_graph import management_reason
from .tools import parse_tool


_ID = re.compile(r"\bobs_\d+_[a-f0-9]+\b")
_CASES = {"evidence_gap", "wrong_or_stale_link", "loop", "compression", "experience_recall", "sufficient_state"}
_GUIDE = """You teach a smaller memory manager using only a public task prefix. Return one JSON object:
{"case":"evidence_gap|wrong_or_stale_link|loop|compression|experience_recall|sufficient_state",
 "proposal":{"operation":"..."}, "target_ids":[]}
Use the tools and revision in the student prompt. A click or an available option never confirms a selection.
For a loop, identify previously tried candidates or commands. For compression, retain exact quotations and
unresolved hard constraints. Task-specific facts stay within this task. An experience is a reusable procedure,
not a fact about the current task. FINISH when more memory work would not help.
The teacher has extra historical evidence. For a missing item, teach a query based on the student's visible
information, and list expected evidence in target_ids. Do not put unseen IDs, source text or offsets in the
student tool call. Any state update or summary must cite a span already shown in the student prompt.
For RETRIEVE or FETCH_EVIDENCE, target_ids must name expected results. For other operations, use []."""

_CHOICE = """Select a useful candidate from verified_options, or propose a missing legal tool call.
Return one JSON object, no markdown:
{"case":"evidence_gap|wrong_or_stale_link|loop|compression|experience_recall|sufficient_state",
 "candidate_id":"option_N"}
If the menu is incomplete, instead return {"case":"...","proposal":{...},"target_ids":[]}.
For retrieval, target_ids names expected results in public_history; the query itself must be
constructible from student-visible information. Writes and summaries must cite student-visible spans.
Optionally add diagnosis: evidence_absent, extractor_gap, stale_link, menu_gap,
execution_gap, compression_risk, sufficient, or undetermined. A truncated history cannot prove absence.
Optionally add probe_worthy: true only when comparing memory operations at this prefix could change a later decision.
Each option was executed on a copy of the current public state. Compare its actual output with the goal
and the student's missing or stale state. Prefer the action that restores the useful evidence or preserves
an unresolved requirement. Treat attempted clicks and available options as separate from confirmed selections.
FINISH when further memory work would not help. Copy candidate_id exactly when selecting."""


def visible_spans(messages):
    payload = json.loads(messages[-1]["content"])
    last = payload.get("last_tool_result") or {}
    snippets = [*payload["recent_evidence"], *last.get("retrieved", [])]
    if payload.get("fetched_span"):
        snippets.append(payload["fetched_span"])
    spans = {}
    for snippet in snippets:
        for piece in (snippet, snippet.get("tail")):
            if piece:
                start = piece.get("span_start", 0)
                spans.setdefault(snippet["id"], []).append((start, start + len(piece["text"])))
    return spans


class LocalTeacher:
    """Optional HF backend. Load it only in the offline example-building process."""

    def __init__(self, path, device):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.path = Path(path).resolve(strict=True)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        loader = AutoModelForImageTextToText if getattr(config, "vision_config", None) else AutoModelForCausalLM
        self.model = loader.from_pretrained(self.path, local_files_only=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32).to(self.device).eval()
        self.torch = torch
        self.identity = {"path": str(self.path), "config_sha256": file_sha(self.path / "config.json"),
                         "weights": {f.name:[f.stat().st_size, f.stat().st_mtime_ns]
                                     for f in sorted(self.path.glob("*.safetensors"))}}

    def __call__(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False, return_dict=False)
        tokens = self.torch.tensor([ids], device=self.device)
        with self.torch.inference_mode():
            output = self.model.generate(tokens, do_sample=False, max_new_tokens=384,
                pad_token_id=self.tokenizer.pad_token_id, attention_mask=self.torch.ones_like(tokens))
        return self.tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True)

    def select_witness(self, target, spans):
        alternatives = [*spans[:4], None]
        alternatives.sort(key=lambda value:digest([target,value]))
        choices = dict(zip("ABCDE", alternatives))
        messages = [{"role":"system","content":"Choose the public span that supports the named slot for the same "
                     "entity. A public pickup confirmation supports held=true; merely seeing an object or attempting "
                     "a command does not. Open and closed refer to the exact numbered container. "
                     "Choose null only if no span supports the requested value. Answer with one option letter."},
                    {"role":"user","content":canonical({"target":target,"options":choices})}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False, return_dict=False)
        tokens = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in choices]
        if any(len(t) != 1 for t in tokens) or len({t[0] for t in tokens}) != len(choices):
            raise ContractError("Teacher witness options must have distinct single-token IDs")
        x = self.torch.tensor([prompt],device=self.device)
        with self.torch.inference_mode():
            logits = self.model(input_ids=x,attention_mask=self.torch.ones_like(x),use_cache=False,logits_to_keep=1).logits[0,-1]
        return list(choices.values())[int(logits[[t[0] for t in tokens]].argmax())]


class TeacherDistillation:
    def __init__(self, generate, max_calls):
        self.generate = generate
        self.max_calls = max_calls
        self.counts = Counter()

    @staticmethod
    def history(atlas, working, goal, max_chars=16000):
        """Rank public records and disclose omitted history rather than silently cutting quotes."""
        terms = set(re.findall(r"[\w]+", goal.lower()))
        for entity, predicate in [*working.get("decision_targets", []), *working.get("recovery_slots", [])]:
            terms.update(re.findall(r"[\w]+", (entity + " " + predicate).lower()))
        records = [r for r in atlas.all_evidence() if not r.get("terminal")]
        ranked = sorted(records, key=lambda r:(sum(t in r["text"].lower() for t in terms), r["revision"]), reverse=True)
        selected, used = [], 0
        for row in ranked:
            if used + len(row["text"]) > max_chars:
                continue
            selected.append({k:row[k] for k in ("id", "entity", "kind", "revision", "text")})
            used += len(row["text"])
        return selected, {"complete":len(selected)==len(records), "records":len(records),
                          "shown":len(selected), "characters":used}

    @staticmethod
    def verify(atlas, working, goal, phase, budget, messages, tools, proposal):
        visible = set(_ID.findall(canonical(messages)))
        if set(_ID.findall(canonical(proposal))) - visible:
            raise ContractError("teacher action refers to evidence unavailable to the student")
        spans = visible_spans(messages)
        refs = list(proposal.get("evidence_refs", []))
        refs.extend(item["evidence_ref"] for item in proposal.get("items", []))
        if proposal["operation"] == "FETCH_EVIDENCE":
            refs.append({"id":proposal["evidence_id"], "start":proposal["start"], "end":proposal["end"]})
        for ref in refs:
            if not any(a <= ref["start"] < ref["end"] <= b for a, b in spans.get(ref["id"], [])):
                raise ContractError("teacher action quotes a span unavailable to the student")
        clone = StateAtlas.restore(atlas.export())
        try:
            return tools.execute(clone, copy.deepcopy(working), proposal, phase, copy.deepcopy(budget), goal)
        finally:
            clone.close()

    def options(self, atlas, working, goal, phase, budget, messages, tools, rule, reason):
        entity = working.get("current_entity", "task")
        revision = atlas.revision
        candidates = [rule]
        if working.get("decision_recovery"):
            from .candidate_actions import candidate_actions
            candidates.extend(candidate_actions(atlas, working, goal, phase, budget, tools).values())
        if entity.startswith("product:") and budget.retrievals:
            identifier = entity.split(":", 1)[1]
            if identifier in messages[-1]["content"]:
                candidates.append({"operation":"RETRIEVE", "query":identifier, "entity":"task", "top_k":1})
            candidates.extend({"operation":"RETRIEVE", "query":predicate, "entity":entity, "predicate":predicate, "top_k":1}
                              for predicate in ("price", "selected_color", "selected_spec"))
        payload = json.loads(messages[-1]["content"])
        for evidence_id in (working.get("interaction_feedback") or {}).get("evidence_ids", [])[:2]:
            candidates.append({"operation":"RETRIEVE", "query":evidence_id, "top_k":1})
        if phase == "context":
            if tools.experience:
                candidates.append({"operation":"RETRIEVE", "query":goal, "source":"experience", "top_k":2})
            for record in payload["recent_evidence"][:-1]:
                if record["kind"] in {"observation", "action_receipt"} and record["text"]:
                    start = record["span_start"]
                    quote = record["text"][:100]
                    candidates.append({"operation":"SUMMARIZE","items":[{"text":quote,
                        "evidence_ref":{"id":record["id"],"start":start,"end":start+len(quote)}}]})
        unique = {canonical(candidate): candidate for candidate in candidates}
        options = []
        for candidate in sorted(unique.values(), key=lambda c: digest([atlas.scope.key, phase, c])):
            proposal = {**candidate, "expected_revision":revision}
            if rule["operation"] != "FINISH" and reason == "context_threshold" and proposal["operation"] != "SUMMARIZE":
                continue
            try:
                result = self.verify(atlas, working, goal, phase, budget, messages, tools, proposal)
            except (ContractError, KeyError, TypeError, ValueError):
                continue
            retrieved = result.get("retrieved", [result["evidence"]] if "evidence" in result else [])
            if proposal["operation"] in {"RETRIEVE", "FETCH_EVIDENCE"} and not retrieved:
                continue
            options.append({"candidate_id":f"option_{len(options) + 1}", "proposal":proposal,
                            "retrieved":[{"id":r["id"],"text":r["text"][:160]} for r in retrieved],
                            "support_status":result.get("support_status"),
                            "gap_diagnostic":result.get("gap_diagnostic")})
        return options

    def annotate(self, atlas, working, goal, phase, budget, messages, tools, rule):
        reason = management_reason(atlas, working, goal, phase)
        if not reason or self.counts["calls"] >= self.max_calls:
            return None
        # Preserve exact offsets in the teacher's read-only history. Student input remains unchanged.
        history, coverage = self.history(atlas, working, goal)
        experience = tools.experience.search(goal) if tools.experience and phase == "context" else []
        options = self.options(atlas, working, goal, phase, budget, messages, tools, rule, reason)
        request = [{"role":"system", "content":_CHOICE if options else _GUIDE}, {"role":"user", "content":canonical({
            "reason":reason, "student_messages":messages,
            "public_history":history, "history_coverage":coverage, "training_experience":experience,
            "verified_options":options})}]
        self.counts["calls"] += 1
        response = self.generate(request)
        try:
            answer = parse_tool(response)
            probe_worthy = answer.pop("probe_worthy", False)
            if type(probe_worthy) is not bool:
                raise ContractError("probe_worthy must be boolean")
            diagnosis = answer.pop("diagnosis", "undetermined")
            if diagnosis not in {"evidence_absent", "extractor_gap", "stale_link", "menu_gap",
                                 "execution_gap", "compression_risk", "sufficient", "undetermined"}:
                raise ContractError("invalid teacher diagnosis")
            if diagnosis == "evidence_absent" and not coverage["complete"]:
                diagnosis = "undetermined"
            if answer.get("case") not in _CASES:
                raise ContractError("invalid teacher response schema")
            if set(answer) == {"case", "candidate_id"}:
                chosen = next((o for o in options if o["candidate_id"] == answer["candidate_id"]), None)
                if not chosen:
                    raise ContractError("teacher selected an unknown option")
                proposal, targets = chosen["proposal"], [r["id"] for r in chosen["retrieved"]]
            elif set(answer) == {"case", "proposal", "target_ids"}:
                proposal, targets = answer["proposal"], answer["target_ids"]
                if not isinstance(proposal, dict) or not isinstance(targets, list) or any(not isinstance(x, str) for x in targets):
                    raise ContractError("invalid teacher proposal or target IDs")
            else:
                raise ContractError("invalid teacher response schema")
            proposal = {**proposal, "expected_revision":atlas.revision}
            operation = proposal["operation"]
            result = self.verify(atlas, working, goal, phase, budget, messages, tools, proposal)
            if operation in {"RETRIEVE", "FETCH_EVIDENCE"}:
                retrieved = result.get("retrieved", [result.get("evidence", {})])
                if not targets or not set(targets) <= {r.get("id") for r in retrieved}:
                    raise ContractError("teacher retrieval does not recover the declared evidence")
            elif targets:
                raise ContractError("only retrieval may declare target IDs")
            if operation == "FINISH" and answer["case"] != "sufficient_state":
                raise ContractError("teacher FINISH requires sufficient_state")
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            self.counts["rejected"] += 1
            self.counts["rejection:" + (str(exc) if isinstance(exc, ContractError) else type(exc).__name__)] += 1
            return None
        self.counts["accepted"] += 1
        self.counts["operation:" + operation] += 1
        if proposal != rule:
            self.counts["different_from_rule"] += 1
        menu_gap = not any(o["proposal"] == proposal for o in options)
        self.counts["validated_menu_additions"] += int(menu_gap)
        from .probe_selection import prefix_signature
        current = atlas.evidence(working["observation_id"])["text"] if working.get("observation_id") else ""
        return {"proposal":proposal, "teacher_case":answer["case"], "verified_target_ids":targets,
                "teacher_trigger":reason, "teacher_diagnosis":diagnosis,
                "diagnosis_verified":False, "menu_addition":menu_gap, "history_coverage":coverage,
                "teacher_probe":{"worthy":probe_worthy,
                                 "prefix_hash":prefix_signature(atlas,working,goal,current,phase)}}
