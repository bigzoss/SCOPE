"""Public-evidence projections and shared working-view policy for the state atlas."""
from collections import Counter
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PublicFact:
    entity: str
    predicate: str
    value: str | float | bool
    start: int
    end: int
    status: str = "supported"

    def proposal(self, record, revision):
        return {"operation": "PROPOSE_STATE_UPDATE", "expected_revision": revision,
                "entity": self.entity, "predicate": self.predicate, "value": self.value,
                "support_status": self.status,
                "evidence_refs": [{"id": record["id"], "start": self.start, "end": self.end}]}


_ALF_OBJECT = r"[a-z][a-z0-9_]* \d+"
_ALF_RULES = tuple(re.compile(pattern, re.I) for pattern in (
    rf"\byou (?:arrive at|are facing) (?:the )?({_ALF_OBJECT})\b",
    rf"\byou (?:pick up|take) (?:the )?({_ALF_OBJECT}) from (?:the )?({_ALF_OBJECT})\b",
    rf"\byou put (?:the )?({_ALF_OBJECT}) (?:in/on|in|on|into) (?:the )?({_ALF_OBJECT})\b",
    rf"\byou (open|close) (?:the )?({_ALF_OBJECT})\b",
    rf"\b({_ALF_OBJECT}) is (open|closed)\b",
    rf"\byou (clean|heat|cool) (?:the )?({_ALF_OBJECT})\b",
    rf"\byou (?:are carrying|have in your inventory)\s*:?\s*([^\n.]+)",
))
ALF_PREDICATES = frozenset({"agent_location", "object_location", "held", "container_open", "cleaned", "temperature"})
SHOP_PREDICATES = frozenset({"current_candidate", "selected_spec"})
_SELECTION = re.compile(r"(?:^|\n|\[SEP\])\s*(已选(?:择)?(?:颜色|尺码|规格|款式)|selected colou?r)\s*[:：]\s*([^\n\[，,;；。]+)", re.I)
_BUDGET = re.compile(r"(?:预算\s*)?(?:不超过|不高于|最多|上限(?:为|是)?)\s*[¥￥]?(\d+(?:\.\d+)?)\s*元")
_COLOR = re.compile(r"(?:颜色\s*(?:要求(?:为|是)?|要|为|是|[:：])\s*|(?:需要|想要|要求)(?:一款|一件)?\s*)(黑色|白色|红色|蓝色|灰色|绿色|银色|金色|透明)")


def alfworld_facts(record):
    text = re.split(r"(?m)^\s*Your task is to\b", record["text"], maxsplit=1, flags=re.I)[0]
    # ALFWorld lists objects after a container, rather than repeating the container
    # for every object. Preserve the complete clause as the support span.
    for match in re.finditer(rf"\b(?:on|in|inside) (?:the )?({_ALF_OBJECT}),?\s+you see\s+([^\n.!]+)", text, re.I):
        for obj in re.findall(_ALF_OBJECT, match[2], re.I):
            yield PublicFact("alf:" + obj.lower(), "object_location", "alf:" + match[1].lower(), *match.span())
    for match in re.finditer(rf"\b({_ALF_OBJECT}) (?:is )?(?:in/on|on|in|inside) (?:the )?({_ALF_OBJECT})\b", text, re.I):
        yield PublicFact("alf:" + match[1].lower(), "object_location", "alf:" + match[2].lower(), *match.span())
    for index, pattern in enumerate(_ALF_RULES):
        for match in pattern.finditer(text):
            first = match[1].lower()
            def fact(entity, predicate, value):
                return PublicFact(entity, predicate, value, *match.span())
            if index == 0:
                yield fact("task", "agent_location", "alf:" + first)
            elif index in (1, 2):
                taken = index == 1
                yield fact("alf:" + first, "held", taken)
                yield fact("alf:" + first, "object_location", "agent" if taken else "alf:" + match[2].lower())
            elif index == 3:
                yield fact("alf:" + match[2].lower(), "container_open", first == "open")
            elif index == 4:
                yield fact("alf:" + first, "container_open", match[2].lower() == "open")
            elif index == 5:
                yield fact("alf:" + match[2].lower(), "cleaned" if first == "clean" else "temperature",
                           True if first == "clean" else {"heat": "hot", "cool": "cold"}[first])
            else:
                for item in re.findall(_ALF_OBJECT, first):
                    yield fact("alf:" + item, "held", True)


def shop_facts(record):
    from .validation import public_prices

    entity, text = record["entity"], record["text"]
    if not entity.startswith("product:"):
        return
    marker = re.search(r"\bBuy Now\b", text, re.I)
    if marker and any(label.casefold() == "buy now" for label in record.get("clickables", [])):
        product = re.search(r"(?<!\w)" + re.escape(entity.split(":", 1)[1]) + r"(?!\w)", text, re.I)
        yield PublicFact("task", "current_candidate", entity, *(product or marker).span(),
                         "supported" if product else "pending")
    if marker or any(label.casefold() == "buy now" for label in record.get("clickables", [])):
        for match in public_prices(text):
            if re.search(r"(?:^|\n|\[SEP\])\s*$", text[:match.start()]):
                yield PublicFact(entity, "price", float(match[1] or match[2]), *match.span())
    for match in _SELECTION.finditer(text):
        field = match[1].removeprefix("已选择") if match[1].startswith("已选择") else match[1].removeprefix("已选")
        color = field == "颜色" or field.lower().startswith("selected colo")
        yield PublicFact(entity, "selected_color" if color else "selected_spec",
                         match[2].strip() if color else field + ":" + match[2].strip(), *match.span())


def apply_observation(tools, atlas, working, record, budget, goal):
    """The same transition is used by rollout and the offline prefix builder."""
    if not working.get("state_graph") or record["kind"] != "observation" or record.get("terminal"):
        return
    working["observations_since_summary"] = working.get("observations_since_summary", 0) + 1
    facts = tools.validator.public_facts(record)
    state = working.get("public_state", {})
    adapter = tools.validator.adapter
    working["focus_entities"] = sorted({f.entity for f in facts} | set(adapter.focus_entities(state) if adapter else ()))
    for fact in facts:
        tools.execute(atlas, working, fact.proposal(record, atlas.revision), "state", budget, goal)


def shopping_checklist(atlas, goal, entity, since=0):
    if not entity.startswith("product:"):
        return []
    rows = []
    claims = atlas.nodes("Claim")
    goal_id = next(c for c in claims if c["predicate"] == "requirement")["evidence_refs"][0]["id"]
    for constraint, pattern, predicate in (("maximum_price_cny", _BUDGET, "price"), ("color", _COLOR, "selected_color")):
        expected = pattern.search(goal)
        if not expected:
            continue
        value = float(expected[1]) if predicate == "price" else expected[1]
        relevant = [c for c in claims if c["entity"] == entity and c["predicate"] == predicate
                    and c["observed_at_revision"] >= since
                    and (predicate != "price" or c.get("unit") == "CNY")]
        latest = max(relevant, key=lambda c: c["observed_at_revision"], default=None)
        status = "pending"
        if latest and latest.get("conflicted"):
            status = "conflict"
        elif latest and latest["support_status"] == "supported":
            matches = latest["value"] <= value if predicate == "price" else latest["value"] == value
            status = "confirmed" if matches else "conflict"
        row = {"constraint": constraint, "expected": value, "status": status,
               "evidence_ids": [goal_id] + ([r["id"] for r in latest["evidence_refs"]] if latest else [])}
        if status == "conflict":
            row["rejection_reason"] = "public evidence disagrees with this explicit constraint"
        rows.append(row)
    rows.append({"constraint": "remaining_user_requirements", "status": "pending", "evidence_ids": [goal_id]})
    return rows


def core_claims(atlas, working):
    entity = working.get("current_entity", "task")
    relevant = {"task", entity, *working.get("focus_entities", [])}
    return [c for c in atlas.nodes("Claim") if c.get("protected") or (
        c["predicate"] != "attempted_action" and c["entity"] in relevant and
        (not c["entity"].startswith("product:") or c["observed_at_revision"] >= working.get("candidate_since", 0)) and
        (c["predicate"] != "current_candidate" or c["value"] == entity) and
        (c.get("state_slot") or c.get("conflicted") or c["support_status"] == "pending"))]


def historical_candidate_evidence(atlas, working):
    entity = working.get("current_entity", "task")
    if not entity.startswith("product:") or working.get("candidate_identity_checked") == working.get("candidate_since", 0):
        return None
    if not any(c["predicate"] == "current_candidate" and c["value"] == entity and c["support_status"] == "pending"
               for c in atlas.nodes("Claim")):
        return None
    identifier = entity.split(":", 1)[1]
    return next((e for e in reversed(atlas.all_evidence()) if e["kind"] == "observation" and e["entity"] == "task"
                 and identifier.casefold() in {label.casefold() for label in e.get("clickables", [])}), None)


def compression_due(atlas, working):
    if working.get("observations_since_summary", 0) >= working.get("compression_after", 6):
        return True
    previous = [e for e in atlas.all_evidence() if e["kind"] == "observation" and
                e["revision"] > working.get("summary_upto_revision", 0)]
    return sum(len(e["text"]) for e in previous[:-1]) >= working.get("compression_min_chars", 1800)


def management_reason(atlas, working, goal, phase):
    if working.get("decision_recovery") and phase == "state" and working.get("decision_targets"):
        from .decision_evidence import review
        if any(r["status"] != "supported" for r in review(atlas, working, goal)):
            return "missing_evidence"
    if phase == "state" and working.get("recovery_slots"):
        return "decision_recovery"
    if working.get("interaction_feedback"):
        return "stalled_or_invalid_action"
    rows = shopping_checklist(atlas, goal, working.get("current_entity", "task"), working.get("candidate_since", 0))
    claims = core_claims(atlas, working)
    if any(c.get("conflicted") for c in claims) or any(r["status"] == "conflict" for r in rows):
        return "conflicting_evidence"
    if phase == "state" and (working.get("confirmation") or {}).get("status") in {"pending", "conflict"}:
        return "public_confirmation_gap"
    if phase == "state" and historical_candidate_evidence(atlas, working):
        return "historical_candidate_evidence"
    if any(c["support_status"] == "pending" and c["predicate"] != "current_candidate" for c in claims) or any(
            r["status"] == "pending" and r["constraint"] != "remaining_user_requirements" for r in rows):
        return "missing_evidence"
    if phase == "context" and compression_due(atlas, working):
        return "context_threshold"
    return None


def action_digest(atlas, entity):
    attempts = [e for e in atlas.all_evidence() if e["kind"] == "action_receipt" and e["entity"] in {"task", entity}]
    counts = Counter(e["text"] for e in attempts)
    return {"repeated": [{"command": command, "count": count} for command, count in counts.most_common(4) if count > 1],
            "recent": [{"command": e["text"], "id": e["id"]} for e in attempts[-3:]]}
