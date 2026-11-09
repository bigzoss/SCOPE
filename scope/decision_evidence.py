"""Public slot evidence for the next decision, with task and version boundaries."""
from .io import digest
from .state_graph import shopping_checklist


def targets(atlas, working, goal):
    entity = working.get("current_entity", "task")
    result = [tuple(slot) for slot in working.get("recovery_slots", ())]
    if working.get("decision_recovery"):
        result += [tuple(slot) for slot in working.get("decision_targets", ())]
    confirmation = working.get("confirmation") or {}
    if confirmation.get("status") in {"pending", "conflict"}:
        result.append((confirmation["entity"], confirmation["predicate"]))
    for row in shopping_checklist(atlas, goal, entity, working.get("candidate_since", 0)):
        if row["status"] != "confirmed" and row["constraint"] in {"color", "maximum_price_cny"}:
            result.append((entity, {"color":"selected_color","maximum_price_cny":"price"}[row["constraint"]]))
    return list(dict.fromkeys(result))[:4]


def review(atlas, working, goal):
    """A missing or conflicting slot cannot be promoted by a retrieval receipt."""
    claims = atlas.nodes("Claim")
    result = []
    for entity, predicate in targets(atlas, working, goal):
        floor = working.get("candidate_since", 0) if entity.startswith("product:") else 0
        pending = working.get("confirmation") or {}
        if (entity, predicate) == (pending.get("entity"), pending.get("predicate")) and pending.get("status") == "pending":
            floor = max(floor, pending["attempt_revision"] + 1)
        candidates = [c for c in claims if (c["entity"], c["predicate"]) == (entity, predicate)
                      and c["observed_at_revision"] >= floor and c["support_status"] == "supported"]
        claim = max(candidates, key=lambda c:c["observed_at_revision"], default=None)
        row = {"entity":entity,"predicate":predicate,"status":"pending","as_of_revision":atlas.revision,
               "min_revision": floor}
        if claim:
            row.update(status="conflict" if claim.get("conflicted") else "supported", value=claim["value"],
                       observed_at_revision=claim["observed_at_revision"],
                       evidence_refs=claim["evidence_refs"])
            if predicate == "price":
                row["unit"] = claim.get("unit")
                if row["unit"] is None:
                    row["status"] = "pending"
        result.append(row)
    return result


def lookup_key(row, state, record):
    quotes = [record["text"][ref["start"]:ref["end"]] for ref in row["evidence_refs"] if ref["id"] == record["id"]]
    return digest([state,row["entity"],row["predicate"],row.get("value"),row.get("unit"),quotes])


def proposals(atlas, working, goal):
    for row in review(atlas, working, goal):
        if row["status"] != "supported":
            if working.get("decision_recovery"):
                from .retrieval import Retriever
                query = row["entity"].split(":", 1)[-1] + " " + row["predicate"].replace("_", " ")
                records = Retriever().search_raw(atlas, query, row["entity"], 1, atlas.revision, row["min_revision"])
                if records and records[0]["id"] != working.get("observation_id"):
                    key = raw_lookup_key(row, working.get("public_state"), records[0])
                    if key not in working.get("decision_lookup_history", []):
                        yield {"operation":"RETRIEVE", "expected_revision":atlas.revision, "query":query,
                               "source":"raw", "entity":row["entity"], "predicate":row["predicate"],
                               "as_of_revision":atlas.revision, "top_k":1}
            continue
        for ref in row["evidence_refs"]:
            eid = ref["id"]
            if eid == working.get("observation_id") or eid in working.get("selected_ids", []):
                continue
            key = lookup_key(row, working.get("public_state"), atlas.evidence(eid))
            if key in working.get("decision_lookup_history", []):
                continue
            yield {"operation":"RETRIEVE","expected_revision":atlas.revision,"query":eid,
                   "entity":row["entity"],"predicate":row["predicate"],
                   "as_of_revision":atlas.revision,"top_k":1}
            break


def raw_lookup_key(row, state, record):
    return digest(["raw", state, row["entity"], row["predicate"], row.get("min_revision", 0), record["text"]])
