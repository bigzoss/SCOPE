"""Deterministic execution of manager proposals. Rejected writes never change facts."""
from __future__ import annotations
import json
from .contracts import ContractError
from .decision_evidence import lookup_key, raw_lookup_key, review
from .io import digest
from .retrieval import Retriever
from .validation import Validator
from .atlas import RELATIONS
from .state_graph import action_digest, core_claims, historical_candidate_evidence, shopping_checklist

STATE_OPS = {"PROPOSE_STATE_UPDATE", "FINISH"}
GRAPH_STATE_OPS = STATE_OPS | {"RETRIEVE", "FETCH_EVIDENCE"}
CONTEXT_OPS = {"RETRIEVE", "FETCH_EVIDENCE", "ADD_MEMORY", "UPDATE_MEMORY", "RETIRE_MEMORY", "SUMMARIZE", "FILTER", "FINISH"}
_STATE_FIELDS = {"entity", "predicate", "value", "support_status", "evidence_refs", "unit", "polarity"}
TOOL_FIELDS = {"FINISH": set(), "PROPOSE_STATE_UPDATE": _STATE_FIELDS, "ADD_MEMORY": _STATE_FIELDS,
               "UPDATE_MEMORY": _STATE_FIELDS | {"node_id"},
               "RETRIEVE": {"query", "entity", "top_k", "max_hops", "source", "relation_types", "predicate", "as_of_revision"},
               "FETCH_EVIDENCE": {"evidence_id", "start", "end"}, "RETIRE_MEMORY": {"node_id"},
               "SUMMARIZE": {"items"}, "FILTER": {"selected_ids"}}
SCHEMA_HELP = '''Return one JSON object, no markdown. Always operation and expected_revision.
state phase: PROPOSE_STATE_UPDATE or FINISH. context phase: RETRIEVE, ADD_MEMORY,
UPDATE_MEMORY, RETIRE_MEMORY, SUMMARIZE, FILTER, FETCH_EVIDENCE, FINISH.
Updates require entity, predicate, scalar value, support_status (supported/pending),
evidence_refs:[{id,start,end}] with exact character offsets in the original source.
Predicates: quoted_text, requirement, price, available_option, selected_color, attempted_action.
Seeing a color button or a click accepted does NOT confirm selected_color.
RETRIEVE: query, optional entity, top_k<=8, max_hops<=2, source=evidence/experience.
Optional relation_types restricts graph expansion to the registered atlas relations.
FETCH_EVIDENCE: evidence_id, start, end, at most 768 characters; returns the exact original span.
SUMMARIZE: items:[{text:exact source substring,evidence_ref:{id,start,end}}].
FILTER: selected_ids (at most 4 evidence IDs). RETIRE_MEMORY: node_id.
FINISH is valid when no management is needed. Never access another task or hidden scores.'''


def graph_tool_prompt(phase, revision, lookup_used=False):
    finish = json.dumps({"operation":"FINISH","expected_revision":revision}, separators=(",",":"))
    lookup = ("RETRIEVE fields: query, optional entity, predicate, as_of_revision, top_k<=8, max_hops<=2, "
              "relation_types, source=evidence or experience. FETCH_EVIDENCE fields: evidence_id, start, end. ")
    if phase == "state":
        allowed = "PROPOSE_STATE_UPDATE, FINISH" if lookup_used else "PROPOSE_STATE_UPDATE, RETRIEVE, FETCH_EVIDENCE, FINISH"
        fields = ("PROPOSE_STATE_UPDATE fields: entity, predicate, value, support_status, evidence_refs, optional unit. "
                  "support_status is supported or pending. evidence_refs is a list of {id,start,end}. "
                  "Copy the exact entity and cite a public source. An attempted click does not confirm selection. ")
        fields += "This turn's lookup was already used. " if lookup_used else lookup.replace(" or experience", "")
    else:
        allowed = "RETRIEVE, FETCH_EVIDENCE, ADD_MEMORY, UPDATE_MEMORY, RETIRE_MEMORY, SUMMARIZE, FILTER, FINISH"
        fields = (lookup + "SUMMARIZE fields: items, a list of {text,evidence_ref:{id,start,end}} using exact source quotes. "
                  "FILTER fields: selected_ids. ADD_MEMORY fields: entity, predicate, value, support_status, evidence_refs, optional unit. "
                  "UPDATE_MEMORY uses the same fields plus node_id. RETIRE_MEMORY fields: node_id. ")
    return (f"Current phase is {phase}. Return exactly one JSON object with operation and expected_revision. "
            f"Allowed operation values: {allowed}. Expected revision: {revision}. "
            f"Use only the fields for your chosen operation. Use value for the state value. {fields}"
            f"Use FINISH if no memory action is needed. Valid example: {finish}")


def parse_tool(text):
    text = text.strip()
    # Strip only a completed model reasoning prefix; never invent a missing JSON suffix.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ContractError("Tool output must be an object")
    return value


class ToolExecutor:
    def __init__(self, experience=None,semantic_validation=True,adapter=None):
        self.validator = Validator(semantic_validation, adapter)
        self.retriever = Retriever()
        self.experience = experience

    def execute(self, atlas, working, proposal, phase, budget, goal):
        op = proposal.get("operation")
        if not isinstance(op,str):raise ContractError("Operation must be text")
        state_ops = GRAPH_STATE_OPS if working.get("state_graph") else STATE_OPS
        if op not in (state_ops if phase == "state" else CONTEXT_OPS):
            raise ContractError("Operation not admitted in this phase")
        if set(proposal)-TOOL_FIELDS[op]-{"operation","expected_revision","proposal_id"}:raise ContractError("Unknown tool arguments")
        if type(proposal.get("expected_revision")) is not int:raise ContractError("Expected revision must be an integer")
        if "proposal_id" in proposal and (not isinstance(proposal["proposal_id"],str) or not proposal["proposal_id"]):raise ContractError("Proposal ID must be nonempty text")
        pid = proposal.get("proposal_id") or digest([atlas.scope.key, phase, proposal])
        saved=atlas.db.execute("SELECT id,body FROM events WHERE proposal_id=?",(pid,)).fetchone()
        if saved:
            event=json.loads(saved[1])
            if event["payload"].get("proposal_hash")!=digest(proposal):raise ContractError("Proposal ID reused with different content")
            return {"accepted":True,"operation":op,"idempotent":True,"commit":{"id":saved[0],**event}}
        if proposal.get("expected_revision") != atlas.revision:
            raise ContractError("STALE_REVISION")
        state_lookup = phase == "state" and op in {"RETRIEVE", "FETCH_EVIDENCE"}
        if state_lookup and working.get("state_lookup_used"):
            raise ContractError("STATE_LOOKUP_ALREADY_USED_THIS_TURN")
        if op == "FINISH":
            if phase == "state" and working.get("state_graph") and historical_candidate_evidence(atlas, working):
                working["candidate_identity_checked"] = working.get("candidate_since", 0)
            return {"accepted": True, "finish": True, "operation": op}
        if op in {"PROPOSE_STATE_UPDATE", "ADD_MEMORY", "UPDATE_MEMORY"}:
            checked = self.validator.check(atlas, proposal)
            if op == "UPDATE_MEMORY":
                old = atlas.node(proposal.get("node_id"))
                if old["kind"] != "Claim" or not old["active"] or old.get("protected") or (old["entity"], old["predicate"]) != (checked["entity"], checked["predicate"]):
                    raise ContractError("Invalid memory update target")
                if checked["support_status"]!="supported":raise ContractError("An unsupported hypothesis cannot supersede memory")
                checked["supersedes"]=old["id"]
            result = atlas.commit(checked, proposal["expected_revision"], pid,digest(proposal))
            return {"accepted": True, "operation": op, "commit": result, "revision": atlas.revision}
        if op == "RETIRE_MEMORY":
            event = atlas.retire(proposal["node_id"], atlas.revision, pid,digest(proposal))
            working["summary"] = None
            return {"accepted": True, "operation": op, "event": event}
        if op == "FETCH_EVIDENCE":
            if budget.retrievals<=0:raise ContractError("RETRIEVAL_BUDGET_EXHAUSTED")
            record=atlas.evidence(proposal.get("evidence_id"))
            start,end=proposal.get("start"),proposal.get("end")
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(record["text"]) or end-start>768:
                raise ContractError("Invalid fetch span/budget")
            budget.retrievals-=1
            if state_lookup:
                working["state_lookup_used"] = True
                if historical_candidate_evidence(atlas, working):
                    working["candidate_identity_checked"] = working.get("candidate_since", 0)
            working["fetched_span"]={"id":record["id"],"span_start":start,"text":record["text"][start:end],
                                     "entity":record["entity"],"support_status":"source_quote"}
            return {"accepted":True,"operation":op,"evidence":working["fetched_span"]}
        if op == "RETRIEVE":
            if any(type(proposal.get(k,d)) is not int for k,d in (("top_k",8),("max_hops",0))):raise ContractError("Retrieval limits must be integers")
            if budget.retrievals <= 0:
                raise ContractError("RETRIEVAL_BUDGET_EXHAUSTED")
            if proposal.get("source","evidence") not in {"experience","evidence","raw"}:raise ContractError("Unknown retrieval source")
            relations=proposal.get("relation_types")
            if relations is not None and (not isinstance(relations,list) or any(not isinstance(x,str) for x in relations) or not set(relations)<=RELATIONS):raise ContractError("Unknown graph relations")
            if proposal.get("source", "evidence") == "experience":
                if state_lookup or any(k in proposal for k in ("predicate", "as_of_revision")):
                    raise ContractError("State-slot lookup requires task evidence")
                budget.retrievals -= 1
                found=self.experience.search(str(proposal.get("query",""))) if self.experience else []
                working["experiences"]=found
                return {"accepted":True,"operation":op,"retrieved":found,"source":"training_experience"}
            predicate = proposal.get("predicate")
            if predicate is not None:
                from .validation import PREDICATES
                if not isinstance(predicate, str) or predicate not in PREDICATES | self.validator.domain_predicates:
                    raise ContractError("Unknown state-slot predicate")
            if proposal.get("source") == "raw":
                if not working.get("decision_recovery"):
                    raise ContractError("Raw decision retrieval requires decision_recovery")
                if proposal.get("max_hops", 0) or relations:
                    raise ContractError("Raw lookup cannot traverse unverified graph links")
                target = next((r for r in review(atlas, working, goal) if
                               (r["entity"], r["predicate"]) == (proposal.get("entity"), predicate)), None)
                if target is None:
                    raise ContractError("Raw lookup needs a current decision target")
                records = self.retriever.search_raw(atlas, str(proposal.get("query", "")), target["entity"],
                    proposal.get("top_k", 2), proposal.get("as_of_revision"), target["min_revision"])
                keys = [raw_lookup_key(target, working.get("public_state"), r) for r in records]
                history = working.get("decision_lookup_history", [])
                if keys and all(k in history for k in keys):
                    raise ContractError("NO_NEW_DECISION_EVIDENCE")
                fresh = [(r, k) for r, k in zip(records, keys) if k not in history]
                records, keys = [r for r, _ in fresh], [k for _, k in fresh]
                budget.retrievals -= 1
                if state_lookup:
                    working["state_lookup_used"] = True
                working["decision_lookup_history"] = (history + keys)[-64:]
                working["selected_ids"] = [r["id"] for r in records[:4]]
                working["selected_spans"] = {r["id"]: r["matched_spans"][0] for r in records[:4]}
                working["selected_decision_slot"] = None
                working["selected_decision_entity"] = working.get("current_entity", "task")
                working["raw_selection_state"] = digest(working.get("public_state"))
                working["summary"] = None
                extracted = [f for r in records for f in self.validator.public_facts(r)
                             if (f.entity, f.predicate) == (target["entity"], target["predicate"])]
                gap = "extractor_gap" if extracted and target["status"] != "supported" else "unverified_raw_evidence"
                return {"accepted": True, "operation": op, "source": "raw", "support_status": "unverified_source",
                        "gap_diagnostic": {"class": gap, "entity": target["entity"], "predicate": predicate},
                        "retrieved": [public_snippet(r, 256, exact_span=True) for r in records],
                        "empty_reason": None if records else "NO_RAW_MATCH_IN_PUBLIC_PREFIX"}
            records = self.retriever.search(atlas, str(proposal.get("query", "")), proposal.get("entity"),
                                            int(proposal.get("top_k", 8)), int(proposal.get("max_hops", 0)),relations,
                                            predicate, proposal.get("as_of_revision"))
            targeted = working.get("decision_evidence") and predicate is not None
            slot_binding = None
            if targeted and records:
                target = next((row for row in review(atlas, working, goal) if row["status"] == "supported"
                               and (row["entity"],row["predicate"]) == (proposal.get("entity"),predicate)), None)
                if target is None or any(r["id"] not in {ref["id"] for ref in target["evidence_refs"]} for r in records):
                    raise ContractError("Retrieval does not match the current public decision slot")
                keys = [lookup_key(target, working.get("public_state"), r) for r in records]
                history = working.get("decision_lookup_history", [])
                if any(key in history for key in keys):
                    raise ContractError("NO_NEW_DECISION_EVIDENCE")
                working["decision_lookup_history"] = (history + keys)[-64:]
                slot_binding = (target["entity"],target["predicate"])
            budget.retrievals -= 1
            if state_lookup:
                working["state_lookup_used"] = True
                if historical_candidate_evidence(atlas, working):
                    working["candidate_identity_checked"] = working.get("candidate_since", 0)
            working["selected_ids"] = [r["id"] for r in records[:4]]
            working.pop("raw_selection_state", None)
            if working.get("state_graph"):
                working["selected_spans"] = {r["id"]: r["matched_spans"][0] for r in records[:4] if r.get("matched_spans")}
            if working.get("decision_evidence"):
                working["selected_decision_slot"] = slot_binding
                working["selected_decision_entity"] = working.get("current_entity","task")
            working["summary"] = None
            return {"accepted": True, "operation": op, "retrieved": [
                        public_snippet(r, 160, exact_span=True) if targeted else public_snippet(r) for r in records],
                    "empty_reason": None if records else "NO_IN_SCOPE_EVIDENCE"}
        if op == "FILTER":
            ids = proposal.get("selected_ids", [])
            if not isinstance(ids, list) or len(ids) > 4:
                raise ContractError("Invalid selected IDs")
            for eid in ids:
                atlas.evidence(eid)
            working.update(selected_ids=ids, summary=None)
            return {"accepted": True, "operation": op, "included": ids}
        if op == "SUMMARIZE":
            summary = self.validator.summary(atlas, proposal, goal)
            if working.get("state_graph") and not summary["items"]:
                raise ContractError("A state-graph summary needs at least one source span")
            working["summary"] = summary
            if working.get("state_graph"):
                working["observations_since_summary"] = 0
                working["summary_upto_revision"] = atlas.revision
            return {"accepted": True, "operation": op, "summary": summary}
        raise ContractError("Unknown operation")


def public_snippet(record, limit=500, exact_span=False):
    # Exact offsets always refer to original full text, including when prompts use snippets.
    span = next(iter(record.get("matched_spans", [])), None)
    start = span["start"] if span else 0
    end = min(span["end"] if exact_span and span else len(record["text"]), start + limit)
    return {"id": record["id"], "entity": record["entity"], "kind": record["kind"],
            "source_length": len(record["text"]), "span_start": start,
            "text": record["text"][start:end], "revision": record["revision"],"retrieval_path":record.get("retrieval_path",[record["id"]])}


def context_view(atlas, working, goal):
    graph = working.get("state_graph")
    decision_rows = review(atlas,working,goal) if working.get("decision_evidence") else []
    selected_ids = working.get("selected_ids", [])
    if working.get("raw_selection_state") is not None and working["raw_selection_state"] != digest(working.get("public_state")):
        selected_ids = []
    if working.get("decision_evidence"):
        binding = working.get("selected_decision_slot")
        if binding:
            live = {ref["id"] for row in decision_rows if row["status"]=="supported"
                    and (row["entity"],row["predicate"])==tuple(binding) for ref in row["evidence_refs"]}
            selected_ids = [eid for eid in selected_ids if eid in live]
        elif working.get("selected_decision_entity") != working.get("current_entity","task"):
            selected_ids = []
    selected = []
    for eid in selected_ids:
        record = atlas.evidence(eid)
        span = working.get("selected_spans", {}).get(eid)
        source = {**record, "matched_spans": [span]} if span else record
        selected.append(public_snippet(source,256 if working.get("raw_selection_state") else 160,exact_span=True)
                        if working.get("decision_evidence") else public_snippet(source))
    claims = core_claims(atlas, working) if graph else atlas.nodes("Claim")
    if len(claims) > 64:
        # Do not silently discard supported claims or pending/conflict markers.
        raise ContractError("Active claim capacity exhausted; retire redundant records")
    summary = working.get("summary")
    if summary and not Validator.summary_valid(atlas,summary,goal):
        summary = None
        working["summary"] = None
    if graph and summary:
        covered = set(summary["source_hashes"])
        selected = [record for record in selected if record["id"] not in covered]
    view = {"goal": goal, "atlas_revision": atlas.revision,
            "claims": claims, "selected_evidence": selected, "summary": summary,
            "fetched_evidence":working.get("fetched_span"),
            "instruction": "pending/attempted and source quotes are not confirmed environment facts"}
    if graph:
        entity = working.get("current_entity", "task")
        view["current_entity"] = entity
        view["recent_actions"] = action_digest(atlas, entity)
        view["constraint_checklist"] = shopping_checklist(atlas, goal, entity, working.get("candidate_since", 0))
        if working.get("confirmation"):
            view["public_confirmation"] = working["confirmation"]
        if working.get("decision_evidence"):
            view["decision_evidence"] = decision_rows
        if working.get("decision_recovery"):
            view["recovery_plan"] = working.get("recovery_plan", {})
        if summary and (view["fetched_evidence"] or {}).get("id") in summary["source_hashes"]:
            view["fetched_evidence"] = None
    view["content_hash"] = digest(view)
    return view
