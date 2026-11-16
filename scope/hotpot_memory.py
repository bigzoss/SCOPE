"""Public-only Hotpot memory candidates; mentions are not inferred world facts.

No answer/support labels or future query are accepted by formation/compression.
All selected content is an exact, sentence-aligned source span.
"""
import re

from .io import digest
from .retrieval import terms
from .state_graph import PublicFact
from .validation import Validator


def contains(text, title):
    return len(title) >= 3 and re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)", text, re.I) is not None


def sentences(record, within=None):
    lo, hi = (within["start"], within["end"]) if within else (0, len(record["text"]))
    return [{"id": record["id"], "start": a, "end": b}
            for a, b in record["sentence_offsets"] if lo <= a < b <= hi and b-a <= 768]


def quote(archive, ref):
    return archive[ref["id"]]["text"][ref["start"]:ref["end"]]


class MentionAdapter:
    """An observation mentions a public title, not necessarily a factual relation.

    Per-target predicates avoid treating mentions of two objects as a state
    conflict. Only entities already observed in this prefix may be referenced.
    """
    def __init__(self, archive):
        self.archive = archive

    @property
    def state_predicates(self):
        return frozenset("mentions:" + r["entity"] for r in self.archive.values())

    def public_facts(self, record):
        source = self.archive.get(record["id"])
        if not source:
            return ()
        result = []
        for ref in sentences(source):
            for target in self.archive.values():
                if target["entity"] != source["entity"] and contains(quote(self.archive, ref), target["title"]):
                    result.append(PublicFact(source["entity"], "mentions:" + target["entity"], True,
                                             ref["start"], ref["end"]))
        return result


class MentionValidator(Validator):
    def check(self, atlas, proposal):
        checked = super().check(atlas, proposal)
        if checked["predicate"].startswith("mentions:"):
            target = checked["predicate"][len("mentions:"):]
            node = atlas.node(target)
            if node["kind"] != "Entity" or not node["active"]:
                from .contracts import ContractError
                raise ContractError("Mention target must be an observed public entity")
            checked["mention_target"] = target
            # Include the checked target in the content identity.
            checked["id"] = "claim_" + digest([checked["id"], target])[:24]
        return checked


def state_options(atlas, archive, tools, visible_ids, visible_refs=None):
    active = {(c["entity"], c["predicate"], c["value"]) for c in atlas.nodes("Claim")}
    result = []
    for record in archive.values():
        if record["id"] not in visible_ids:
            continue
        for fact in tools.validator.public_facts(record):
            if visible_refs is not None and not any(r["id"] == record["id"] and
                    r["start"] <= fact.start < fact.end <= r["end"] for r in visible_refs):
                continue
            if (fact.entity, fact.predicate, fact.value) in active:
                continue
            proposal = fact.proposal(record, atlas.revision)
            tools.validator.check(atlas, proposal)
            ref = proposal["evidence_refs"][0]
            result.append({"operation": "UPDATE_STATE", "proposal": proposal, "evidence": ref,
                           "semantics": "source_mentions_public_entity_not_world_relation"})
    return list({digest(x): x for x in result}.values())


def bridge_options(archive, kept, question):
    """One observed mention -> another public document; select its query-relevant sentence."""
    wanted = set(terms(question))
    result = []
    for source_ref in kept:
        source = archive[source_ref["id"]]
        for target in archive.values():
            if source["entity"] == target["entity"] or not contains(quote(archive, source_ref), target["title"]):
                continue
            spans = sentences(target)
            if not spans:
                continue
            ref = max(spans, key=lambda r: (len(wanted & set(terms(quote(archive, r)))), -r["start"]))
            if any(old["id"] == ref["id"] and old["start"] <= ref["start"] and old["end"] >= ref["end"] for old in kept):
                continue
            result.append({"operation": "RETRIEVE", "evidence": ref,
                           "bridge_ref": source_ref, "query": question + " " + target["title"],
                           "strategy": "public_entity_bridge"})
    return list({digest(x["evidence"]): x for x in result}.values())


def relation_compressions(archive, kept):
    """Conservative extractive compression before the question is released.

    Keep observed entity bridges, numbers and explicit negation/qualification.
    These are protected public anchors, not gold supporting-fact labels.
    """
    selected, protected = [], []
    for ref in kept:
        record = archive[ref["id"]]
        spans = sentences(record, ref)
        if not spans:
            selected.append(dict(ref))
            protected.append(dict(ref))
            continue
        must = [span for span in spans if re.search(
            r"\b(?:not|never|except|only|before|after|until|without|at least|at most)\b|\d", quote(archive, span), re.I)
            or any(other["id"] != ref["id"] and contains(quote(archive, span), other["title"])
                   for other in archive.values())]
        protected.extend(must)
        selected.extend(must or spans[:1])
    selected = list({digest(x): x for x in selected}.values())
    if len(selected) > 12 or sum(r["end"]-r["start"] for r in selected) >= sum(r["end"]-r["start"] for r in kept):
        return []
    return [{"operation": "COMPRESS", "strategy": "preserve_public_relations",
             "refs": selected, "protected_refs": protected}]


def expanded_options(stage, options, atlas, archive, kept, tools):
    """At most eight choices, preserving noop and reserving room for new types."""
    visible = {r["id"] for r in kept}
    visible_refs = list(kept)
    if stage.startswith("write"):
        visible.update(x["evidence"]["id"] for x in options if x.get("evidence"))
        # Newly arriving documents are public at formation time. At retrieval
        # time a state update may only cite the working view: it is not a free
        # alternative to the budgeted retrieval operation.
        visible_refs.extend({"id":x["evidence"]["id"],"start":0,"end":len(archive[x["evidence"]["id"]]["text"])}
                            for x in options if x.get("evidence"))
    updates = state_options(atlas, archive, tools, visible, visible_refs)
    if stage == "compress":
        extra = relation_compressions(archive, kept)
        ordered = [options[0], *extra, *options[1:]]
    else:
        # Existing retrieval options may already have bridge-first ranking.
        ordered = [options[0], *options[1:5], *updates[:2], *options[5:]]
    return list({digest(x): x for x in ordered}.values())[:8]
