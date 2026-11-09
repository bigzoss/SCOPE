"""Deterministic execution of manager proposals. Rejected writes never change facts."""
from __future__ import annotations
import json
from .contracts import ContractError
from .io import digest
from .retrieval import Retriever
from .validation import Validator
from .atlas import RELATIONS

STATE_OPS = {"PROPOSE_STATE_UPDATE", "FINISH"}
CONTEXT_OPS = {"RETRIEVE", "FETCH_EVIDENCE", "ADD_MEMORY", "UPDATE_MEMORY", "RETIRE_MEMORY", "SUMMARIZE", "FILTER", "FINISH"}
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
    def __init__(self, experience=None,semantic_validation=True):
        self.validator = Validator(semantic_validation)
        self.retriever = Retriever()
        self.experience = experience

    def execute(self, atlas, working, proposal, phase, budget, goal):
        op = proposal.get("operation")
        if not isinstance(op,str):raise ContractError("Operation must be text")
        if op not in (STATE_OPS if phase == "state" else CONTEXT_OPS):
            raise ContractError("Operation not admitted in this phase")
        fields={"FINISH":set(),"PROPOSE_STATE_UPDATE":{"entity","predicate","value","support_status","evidence_refs","unit","polarity"},
                "ADD_MEMORY":{"entity","predicate","value","support_status","evidence_refs","unit","polarity"},
                "UPDATE_MEMORY":{"entity","predicate","value","support_status","evidence_refs","unit","polarity","node_id"},
                "RETRIEVE":{"query","entity","top_k","max_hops","source","relation_types"},
                "FETCH_EVIDENCE":{"evidence_id","start","end"},"RETIRE_MEMORY":{"node_id"},
                "SUMMARIZE":{"items"},"FILTER":{"selected_ids"}}
        if set(proposal)-fields[op]-{"operation","expected_revision","proposal_id"}:raise ContractError("Unknown tool arguments")
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
        if op == "FINISH":
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
            working["fetched_span"]={"id":record["id"],"span_start":start,"text":record["text"][start:end],
                                     "entity":record["entity"],"support_status":"source_quote"}
            return {"accepted":True,"operation":op,"evidence":working["fetched_span"]}
        if op == "RETRIEVE":
            if any(type(proposal.get(k,d)) is not int for k,d in (("top_k",8),("max_hops",0))):raise ContractError("Retrieval limits must be integers")
            if budget.retrievals <= 0:
                raise ContractError("RETRIEVAL_BUDGET_EXHAUSTED")
            budget.retrievals -= 1
            if proposal.get("source","evidence") not in {"experience","evidence"}:raise ContractError("Unknown retrieval source")
            relations=proposal.get("relation_types")
            if relations is not None and (not isinstance(relations,list) or any(not isinstance(x,str) for x in relations) or not set(relations)<=RELATIONS):raise ContractError("Unknown graph relations")
            if proposal.get("source", "evidence") == "experience":
                found=self.experience.search(str(proposal.get("query",""))) if self.experience else []
                working["experiences"]=found
                return {"accepted":True,"operation":op,"retrieved":found,"source":"training_experience"}
            records = self.retriever.search(atlas, str(proposal.get("query", "")), proposal.get("entity"),
                                            int(proposal.get("top_k", 8)), int(proposal.get("max_hops", 0)),relations)
            working["selected_ids"] = [r["id"] for r in records[:4]]
            working["summary"] = None
            return {"accepted": True, "operation": op, "retrieved": [public_snippet(r) for r in records],
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
            working["summary"] = summary
            return {"accepted": True, "operation": op, "summary": summary}
        raise ContractError("Unknown operation")


def public_snippet(record, limit=500):
    # Exact offsets always refer to original full text, including when prompts use snippets.
    return {"id": record["id"], "entity": record["entity"], "kind": record["kind"],
            "source_length": len(record["text"]), "span_start": 0,
            "text": record["text"][:limit], "revision": record["revision"],"retrieval_path":record.get("retrieval_path",[record["id"]])}


def context_view(atlas, working, goal):
    selected = []
    for eid in working.get("selected_ids", []):
        selected.append(public_snippet(atlas.evidence(eid)))
    claims = atlas.nodes("Claim")
    if len(claims) > 64:
        # Do not silently discard supported claims or pending/conflict markers.
        raise ContractError("Active claim capacity exhausted; retire redundant records")
    summary = working.get("summary")
    if summary and not Validator.summary_valid(atlas,summary,goal):
        summary = None
        working["summary"] = None
    view = {"goal": goal, "atlas_revision": atlas.revision,
            "claims": claims, "selected_evidence": selected, "summary": summary,
            "fetched_evidence":working.get("fetched_span"),
            "instruction": "pending/attempted and source quotes are not confirmed environment facts"}
    view["content_hash"] = digest(view)
    return view
