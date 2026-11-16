"""Observed source-span flow, deliberately without causal-success edges."""
import json

from .contracts import ContractError
from .io import digest


def memory_flow(trace):
    archive = {key: json.loads(raw) for key, raw in trace["atlas"]["tables"]["evidence"]}
    nodes, edges = {}, []

    def span(ref, revision=None):
        source = archive.get(ref["id"])
        if (source is None or source["kind"] != "observation"
                or source["scope"]["task"] != trace["task_id"]
                or not 0 <= ref["start"] < ref["end"] <= len(source["text"])
                or (revision is not None and source["revision"] > revision)):
            raise ContractError("Memory flow references invalid, out-of-scope, or future public evidence")
        key = "span_" + digest([ref["id"], ref["start"], ref["end"]])[:24]
        nodes[key] = {"id": key, "kind": "public_source_span", "evidence_ref": {
            field: ref[field] for field in ("id", "start", "end")}, "entity": source["entity"],
            "source_revision": source["revision"], "source_hash": source["content_hash"]}
        return key

    previous = None
    for index, action in enumerate(trace["actions"]):
        op_id = f"operation_{index}"
        view_id = f"working_view_{index}"
        nodes[op_id] = {"id": op_id, "kind": "management_operation", "operation": action["operation"],
                        "boundary": action["boundary"], "stage": action["stage"]}
        nodes[view_id] = {"id": view_id, "kind": "working_view", "stage": action["stage"]}
        if previous is not None:
            edges.append({"src": previous, "relation": "INPUT_TO", "dst": op_id})
        option = action["menu"][action["choice"]]
        if option.get("evidence"):
            edges.append({"src": span(option["evidence"], action["atlas_revision"]),
                          "relation": "SELECTED_BY", "dst": op_id})
        for ref in action["working_after"]:
            edges.append({"src": span(ref), "relation": "PRESENT_IN", "dst": view_id})
        edges.append({"src": op_id, "relation": "PRODUCES", "dst": view_id})
        previous = view_id
    nodes["reader_view"] = {"id": "reader_view", "kind": "reader_input",
                             "view_hash": digest(trace["answer_view"])}
    for ref in trace["answer_view"]:
        edges.append({"src": span(ref), "relation": "DELIVERED_TO", "dst": "reader_view"})
    if previous:
        edges.append({"src": previous, "relation": "RENDERED_AS", "dst": "reader_view"})
    return {"schema": "scope-public-memory-flow-v1", "nodes": list(nodes.values()), "edges": edges,
            "semantics": "observed selection and delivery only; no attribution of causal task success",
            "contains_gold": False}
