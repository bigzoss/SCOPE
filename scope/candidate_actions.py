"""Bounded memory actions built from public, versioned state evidence."""
import copy
import json
import string

from .atlas import StateAtlas
from .contracts import ContractError
from .decision_evidence import proposals as decision_proposals
from .io import canonical, digest
from .state_graph import compression_due, historical_candidate_evidence, shopping_checklist


LABELS = tuple(string.ascii_uppercase[:8])


def summary_sources(observations, current, entity):
    sources = current[-2:] if current else observations[-2:]
    if len(sources) == 1 and entity.startswith("product:"):
        identifier = entity.split(":", 1)[1]
        prior = next((e for e in reversed(observations) if e["entity"] == "task" and
                      e["revision"] < sources[0]["revision"] and identifier in e.get("clickables", []) and
                      identifier in e["text"]), None)
        if prior:
            sources = [prior, *sources]
    return sources


def quote_item(record, start=0, width=160):
    text = record["text"][start:start+width]
    return {"text":text,"evidence_ref":{"id":record["id"],"start":start,"end":start+len(text)}}


def candidate_actions(atlas, working, goal, phase, budget, tools, audit=None):
    revision = atlas.revision
    entity = working.get("current_entity", "task")
    entities = {entity, "task", *working.get("focus_entities", [])}
    observations = [e for e in atlas.all_evidence() if e["kind"] == "observation" and not e.get("terminal")]
    current = [e for e in observations if e["entity"] == entity and e["revision"] >= working.get("candidate_since", 0)]
    options = []

    shared = working.get("shared_memory")
    if phase == "state":
        recent = observations[-4:] if entity == "task" else current[-3:]
        if working.get("decision_recovery"):
            retrieved = [atlas.evidence(eid) for eid in working.get("selected_ids", [])]
            recent = list({r["id"]: r for r in recent + retrieved}.values())
            entities.update(e for e, _ in working.get("decision_targets", []))
            entities.update(e for e, _ in working.get("recovery_slots", []))
        slots = set()
        active = {(c["entity"],c["predicate"],c["value"],c["support_status"]) for c in atlas.nodes("Claim")} if shared else set()
        for record in reversed(recent):
            for fact in tools.validator.public_facts(record):
                pending = working.get("confirmation") or {}
                if working.get("decision_recovery") and pending.get("status") == "pending" and (
                        fact.entity, fact.predicate) == (pending.get("entity"), pending.get("predicate")) and (
                        record["revision"] <= pending["attempt_revision"]):
                    continue
                field = fact.value.split(":", 1)[0] if fact.predicate == "selected_spec" else None
                slot = (fact.entity, fact.predicate, field)
                if slot not in slots and fact.entity in entities and fact.predicate != "current_candidate":
                    slots.add(slot)
                    if (fact.entity,fact.predicate,fact.value,fact.status) not in active:
                        options.append(fact.proposal(record, revision))
        rows = shopping_checklist(atlas, goal, entity, working.get("candidate_since", 0))
        for row in rows:
            if row["constraint"] != "color" or row["status"] != "pending":
                continue
            color = row["expected"]
            page = next((e for e in reversed(current) if color in e.get("clickables", []) and color in e["text"]), None)
            if page:
                start = page["text"].find(color)
                ref = {"id":page["id"],"start":start,"end":start+len(color)}
                for predicate, status in (("selected_color", "pending"), ("available_option", "supported")):
                    options.append({"operation":"PROPOSE_STATE_UPDATE", "expected_revision":revision, "entity":entity,
                                    "predicate":predicate,"value":color,"support_status":status,"evidence_refs":[ref]})

    if budget.retrievals and (phase == "context" or not working.get("state_lookup_used")):
        historical = historical_candidate_evidence(atlas, working)
        feedback = working.get("interaction_feedback") or {}
        ids = ([historical["id"]] if historical else []) + feedback.get("evidence_ids", [])[:2]
        target = working.get("confirmation") if shared and phase == "state" else None
        if target and target["status"] != "confirmed":
            ids = target.get("evidence_ids", []) + ids
        if working.get("decision_evidence"):
            options[0:0] = list(decision_proposals(atlas, working, goal))
            # A generic loop report is not a slot-specific lookup.
            ids = [historical["id"]] if historical and not working.get("recovery_slots") else []
        for eid in ids:
            if not shared or eid not in working.get("selected_ids", []):
                options.append({"operation":"RETRIEVE","expected_revision":revision,"query":eid,"top_k":1})
        if phase == "context" and feedback and tools.experience:
            if tools.experience.search(goal):
                options.append({"operation":"RETRIEVE","expected_revision":revision,"query":goal,
                                "source":"experience","top_k":2})

    if phase == "context" and compression_due(atlas, working):
        sources = summary_sources(observations, current, entity)
        if len(sources) == 2:
            items = []
            search_then_product = sources[0]["entity"] == "task" and entity.startswith("product:")
            for record in sources:
                start = 0
                if record["entity"] == "task" and entity.startswith("product:"):
                    start = max(0, record["text"].find(entity.split(":", 1)[1]))
                elif search_then_product and record["text"].startswith("Instruction:") and "[SEP]" in record["text"]:
                    start = record["text"].index("[SEP]") + len("[SEP]")
                if record["text"]:
                    items.append(quote_item(record, start))
            if items:
                options.append({"operation":"SUMMARIZE","expected_revision":revision,"items":items})

    valid = [{"operation":"FINISH","expected_revision":revision}]
    seen = {canonical(valid[0])}
    disabled = working.get("disabled_memory_tools", ())
    for proposal in options:
        if proposal["operation"] in disabled:
            continue
        key = canonical(proposal)
        if key in seen:
            continue
        seen.add(key)
        clone = StateAtlas.restore(atlas.export())
        try:
            result = tools.execute(clone, copy.deepcopy(working), proposal, phase, copy.deepcopy(budget), goal)
            if result["accepted"] and (proposal["operation"] != "RETRIEVE" or result.get("retrieved")):
                if len(valid) == len(LABELS):
                    if audit is not None:
                        audit.append({"class":"menu_gap","detail":"verified_action_excluded_by_menu_capacity",
                                      "operation":proposal["operation"]})
                    break
                valid.append(proposal)
        except ContractError as exc:
            if audit is not None:
                audit.append({"class":"validation_rejected", "operation":proposal["operation"], "detail":str(exc)})
        finally:
            clone.close()
        if len(valid) == len(LABELS) and audit is None:
            break
    valid.sort(key=lambda p:digest([atlas.scope.key, phase, p]))
    return dict(zip(LABELS, valid))


def choice_payload(payload, menu):
    payload = dict(payload)
    payload["verified_options"] = menu
    return [{"role":"system","content":"Choose one verified memory operation for the current public state. "
             "Prefer the current entity and current evidence. An available option does not confirm a selection. "
             "Use a pending update when confirmation is missing, preserve hard constraints in summaries, "
             "and finish when evidence already suffices. Answer with only the option letter."},
            {"role":"user","content":canonical(payload)}]


def choice_messages(messages, menu):
    return choice_payload(json.loads(messages[-1]["content"]), menu)
