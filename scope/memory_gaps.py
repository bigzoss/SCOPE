"""Evidence-gap decisions use a single named target and an explicit public prefix."""
import re

from .candidate_actions import LABELS
from .io import canonical, digest
from .state_graph import shopping_checklist
from .validation import public_prices


GAP_TYPES = ("missing_evidence", "extractor_gap", "menu_gap", "budget_limited", "undetermined")


def diagnose(*, target, public_witness, extracted, menu_covers, allowed, prefix_complete=True):
    """Unknown is a valid outcome. An exhausted budget is independent of evidence coverage."""
    if not allowed:
        return "budget_limited"
    if not target or not prefix_complete or (extracted and not public_witness):
        return "undetermined"
    if not public_witness:
        return "missing_evidence"
    if not extracted:
        return "extractor_gap"
    return "undetermined" if menu_covers else "menu_gap"


def diagnosis_messages(prefix, identity):
    choices = sorted(GAP_TYPES, key=lambda kind:digest([identity, kind]))
    options = {label:{"diagnosis":kind} for label, kind in zip(LABELS, choices)}
    prompt = ("Diagnose the gap for the named target in this public task prefix. "
              "Missing evidence: no public observation supports the target. Extractor gap: the public text supports the "
              "target but the projection omitted it. Menu gap: the fact is projected, an operation is allowed, but the menu "
              "omits it. Budget limited: the operation is blocked by its budget or phase. If the target or evidence is unclear, "
              "choose undetermined. A goal, click attempt, or available option is not proof of a selected state. "
              "Answer with only the option letter.")
    return [{"role":"system","content":prompt},
            {"role":"user","content":canonical({**prefix,"verified_options":options})}], options


def confirmation_status(atlas, tools, target):
    """Compare the post-action public observations against live, versioned claims."""
    entity, predicate, expected = target["entity"], target["predicate"], target["value"]
    current = [c for c in atlas.nodes("Claim") if (c["entity"], c["predicate"]) == (entity, predicate)
               and c["support_status"] == "supported"]
    witnesses = [(record, fact) for record in atlas.all_evidence() if record["revision"] > target["attempt_revision"]
                 for fact in tools.validator.public_facts(record)
                 if (fact.entity, fact.predicate) == (entity, predicate) and fact.status == "supported"]
    if not witnesses:
        return {**target, "status":"pending", "class":"missing_evidence", "evidence_ids":[]}
    record, fact = max(witnesses, key=lambda witness:witness[0]["revision"])
    # A repeated public confirmation may leave an identical live slot unchanged.
    matching = next((c for c in current if c["value"] == fact.value and
                     c["observed_at_revision"] <= record["revision"]), None)
    status = "pending" if not matching else ("conflict" if matching.get("conflicted") or fact.value != expected else "confirmed")
    return {**target,"status":status,"class":"extractor_gap" if not matching else ("conflicting_evidence" if
                status == "conflict" else "confirmed"),"evidence_ids":[record["id"]]}


def skipped_candidate_reason(working, budget, phase, atlas=None, goal=""):
    confirmation = working.get("confirmation")
    if working.get("interaction_feedback") or (phase == "state" and confirmation and confirmation["status"] != "confirmed"):
        if budget.retrievals == 0:
            return {"class":"budget_limited","detail":"episode_retrieval_budget"}
        if phase == "state" and working.get("state_lookup_used"):
            return {"class":"budget_limited","detail":"state_lookup_used_this_turn"}
    if phase == "state" and confirmation and confirmation["status"] != "confirmed":
        return {**confirmation,"detail":"post_action_public_confirmation"}
    entity = working.get("current_entity", "task")
    if atlas is not None and entity.startswith("product:"):
        since = working.get("candidate_since", 0)
        rows = shopping_checklist(atlas, goal, entity, since)
        records = [e for e in atlas.all_evidence() if e["entity"] == entity and e["kind"] == "observation"
                   and e["revision"] >= since and not e.get("terminal")]
        for row in rows:
            if row["constraint"] == "color" and row["status"] == "pending":
                for record in reversed(records):
                    match = re.search(r"(?:^|\n|\[SEP\])\s*(Selected colou?r\s*:\s*[^\n\[.,;]+)",record["text"],re.I)
                    if match:
                        return {"class":"extractor_gap","detail":"explicit_selection_alias_not_projected",
                                "evidence_ids":[record["id"]]}
            if row["constraint"] == "maximum_price_cny" and row["status"] == "pending":
                prices = [(record["id"],match.group(0)) for record in records for match in public_prices(record["text"])
                          if re.search(r"(?:^|\n|\[SEP\])\s*$",record["text"][:match.start()])]
                if prices and not any(any(unit in text for unit in ("元","人民币","￥")) for _,text in prices):
                    return {"class":"missing_evidence","detail":"displayed_price_has_no_currency",
                            "evidence_ids":list(dict.fromkeys(eid for eid,_ in prices))}
    return {"class":"undetermined","detail":"no_verified_candidate_for_current_phase"}
