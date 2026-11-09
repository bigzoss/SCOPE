"""Public evidence support checks; a quote existing is not enough for arbitrary predicates."""
from __future__ import annotations
import re
import math
from .contracts import ContractError
from .io import digest

PRICE = re.compile(r"(?:(?:价格|售价|现价|单价)\s*[:：]?\s*[¥￥]?\s*)(\d+(?:\.\d+)?)(?:\s*元)?|[¥￥]\s*(\d+(?:\.\d+)?)(?:\s*元)?")
_SELECTED=re.compile(r"(?:^|\n|\[SEP\])\s*(?P<field>(?:已选颜色|已选择颜色|selected colou?r)\s*[:：]\s*(?P<value>[^\n，,;；。\[]+))", re.I)
PREDICATES = {"quoted_text", "requirement", "price", "available_option", "selected_color", "attempted_action"}


def public_prices(text):
    for match in PRICE.finditer(text):
        prefix = text[max(0, match.start()-8):match.start()]
        suffix = text[match.end():match.end()+8].lstrip()
        if re.search(r"(?:不是|并非|不含|不高于|不低于|高于|低于|大于|小于|至少|至多|未确认)\s*$", prefix):
            continue
        if re.match(r"(?:起|以上|以下|以内|左右|到|至|[-~～])", suffix):
            continue
        yield match


class Validator:
    def __init__(self,semantic_validation=True, adapter=None):
        self.semantic_validation=semantic_validation
        self.adapter = adapter
        self.domain_predicates = getattr(adapter, "state_predicates", frozenset())

    def public_facts(self, record):
        if self.adapter and record["kind"] == "observation" and not record.get("terminal"):
            return tuple(self.adapter.public_facts(record))
        return ()

    def check(self, atlas, proposal):
        if proposal.get("expected_revision") != atlas.revision:
            raise ContractError("STALE_REVISION")
        predicate = proposal.get("predicate")
        domain = predicate in self.domain_predicates
        if predicate not in PREDICATES and not domain:
            raise ContractError("UNREGISTERED_PREDICATE")
        status = proposal.get("support_status", "pending")
        if status not in {"supported", "pending"}:
            raise ContractError("Unsupported support status")
        value = proposal.get("value")
        if value is None or isinstance(value, (list, dict)):
            raise ContractError("Scalar non-null value required")
        if predicate=="price":
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ContractError("Price requires a finite nonnegative number, not a boolean")
        elif not (domain and type(value) is bool) and (not isinstance(value,str) or not value.strip()):
            raise ContractError("Text predicates require a nonempty string")
        entity = proposal.get("entity", "task")
        if not domain:
            entity_node=atlas.node(entity)
            if entity_node["kind"]!="Entity" or not entity_node["active"]:raise ContractError("Claim subject must be a public active entity")
        refs = proposal.get("evidence_refs", [])
        if not isinstance(refs, list) or not refs:
            raise ContractError("Evidence references required, including for pending claims")
        spans = []
        records = []
        for ref in refs:
            if not isinstance(ref, dict) or not {"id", "start", "end"} <= ref.keys():
                raise ContractError("Malformed evidence reference")
            record = atlas.evidence(ref["id"])
            start, end = ref["start"], ref["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(record["text"]):
                raise ContractError("Invalid evidence span")
            if not domain and entity != "task" and record["entity"] != entity:
                raise ContractError("WRONG_ENTITY_EVIDENCE")
            if record["scope"] != vars(atlas.scope):
                raise ContractError("CROSS_SCOPE_EVIDENCE")
            spans.append(record["text"][start:end]); records.append(record)
        quote = "\n".join(spans)
        supported = False
        domain_matches = [(f, record, ref) for record, ref in zip(records, refs) for f in self.public_facts(record)
                          if f.entity == entity and f.predicate == predicate and f.value == value
                          and (type(f.value) is type(value) or predicate == "price")
                          and ref["start"] <= f.start < f.end <= ref["end"]]
        matching_prices=[]
        if domain:
            if not domain_matches:
                raise ContractError("EVIDENCE_DOES_NOT_BIND_DOMAIN_SLOT")
            supported = any(f.status == "supported" for f, _, _ in domain_matches)
        elif predicate == "requirement":
            supported = any(r["kind"]=="requirement" and value==r["text"] and ref["start"]==0 and ref["end"]==len(r["text"])
                            for r,ref in zip(records,refs))
        elif predicate == "quoted_text":
            # This predicate says only 'the source contains this quote'. No semantic upgrade.
            supported = any(str(value) in span for span in spans)
        elif predicate == "price":
            for record,ref in zip(records,refs):
                if record["kind"]!="observation" or entity=="task":continue
                for match in public_prices(record["text"]):
                    if not ref["start"]<=match.start()<match.end()<=ref["end"]:continue
                    if float(match.group(1) or match.group(2))==float(value):matching_prices.append(match.group(0))
            supported=bool(matching_prices)
        elif predicate == "available_option":
            supported = str(value) in quote and any(str(value) in r.get("clickables", []) for r in records)
        elif predicate == "selected_color":
            # Current upstream text does not usually expose this confirmation: keep pending.
            supported = entity!="task" and any(r["kind"]=="observation" and any(
                ref["start"]<=match.start('field')<match.end('field')<=ref["end"] and match.group('value').strip()==value
                for match in _SELECTED.finditer(r["text"])) for r,ref in zip(records,refs))
        elif predicate == "attempted_action":
            supported = any(r["kind"]=="action_receipt" and value==r["text"] and ref["start"]==0 and ref["end"]==len(r["text"])
                            for r,ref in zip(records,refs))
        if status == "supported" and not supported and (self.semantic_validation or domain):
            raise ContractError("EVIDENCE_DOES_NOT_SUPPORT_PREDICATE")
        unit=proposal.get("unit")
        if predicate=="price":
            inferred="CNY" if any(any(c in field for c in ("元","人民币","￥")) for field in matching_prices) else None
            normalized="CNY" if unit in ("元","人民币","RMB","CNY") else unit
            if normalized is not None and normalized!=inferred:
                raise ContractError("PRICE_UNIT_NOT_SUPPORTED_BY_EVIDENCE")
            unit=inferred
        node = {"kind": "Claim", "entity": entity, "predicate": predicate, "value": value,
                "support_status": status, "evidence_refs": refs,
                "observed_at_revision":max(r["revision"] for r in records),"validity":"as_observed_not_automatic_current_truth",
                "validation_mode":"semantic_checked" if self.semantic_validation else "semantic_check_ablated",
                "unit": unit, "polarity": proposal.get("polarity", "positive"),
                "protected": predicate == "requirement"}
        if domain_matches:
            matching = [(record, ref) for f, record, ref in domain_matches if status == "pending" or f.status == "supported"]
            if matching:
                node["observed_at_revision"] = max(record["revision"] for record, _ in matching)
                node["evidence_refs"] = list({(ref["id"], ref["start"], ref["end"]): ref for _, ref in matching}.values())
            node["state_slot"] = predicate + (":" + value.split(":", 1)[0] if predicate == "selected_spec" else "")
            node["validity"] = "current_until_superseded" if status == "supported" else "pending_public_confirmation"
        if node["polarity"] != "positive":
            raise ContractError("Negative semantics require a registered predicate; use a literal quote")
        node["id"] = "claim_" + digest([atlas.scope.key, node, atlas.revision])[:24]
        return node

    def summary(self, atlas, proposal, goal):
        """Extractive structured summaries first; arbitrary free paraphrase stays unconfirmed."""
        items = proposal.get("items")
        if not isinstance(items, list) or len(items) > 12:
            raise ContractError("Summary requires at most 12 source-mapped items")
        checked = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("evidence_ref"), dict):
                raise ContractError("Malformed summary item")
            ref = item.get("evidence_ref", {})
            record = atlas.evidence(ref.get("id"))
            a, b = ref.get("start"), ref.get("end")
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(record["text"]):
                raise ContractError("Invalid summary span")
            quote = record["text"][a:b]
            if item.get("text") != quote:
                raise ContractError("Summary text must preserve the exact source span")
            checked.append({"text": quote, "evidence_ref": ref, "entity": record["entity"],
                            "support_status": "source_quote"})
        # Goal is rendered separately and unconditionally. Summary cannot rewrite/drop it.
        return {"goal_hash": digest(goal), "items": checked, "atlas_revision": atlas.revision,
                "source_hashes":{item["evidence_ref"]["id"]:atlas.evidence(item["evidence_ref"]["id"])["content_hash"] for item in checked},
                "protected_claim_hashes":{c["id"]:digest(c) for c in atlas.nodes("Claim")
                    if c.get("protected") or c.get("conflicted") or c.get("support_status")=="pending"}}

    @staticmethod
    def summary_valid(atlas,summary,goal):
        if not summary or summary.get("goal_hash")!=digest(goal) or "source_hashes" not in summary:return False
        try:
            expected=Validator().summary(atlas,{"items":summary.get("items")},goal)
            if expected["source_hashes"]!=summary["source_hashes"]:return False
            for eid,sha in summary["source_hashes"].items():
                if atlas.evidence(eid)["content_hash"]!=sha:return False
            for nid,sha in summary.get("protected_claim_hashes",{}).items():
                current=atlas.node(nid)
                if not current.pop("active") or digest(current)!=sha:return False
        except ContractError:return False
        return True

