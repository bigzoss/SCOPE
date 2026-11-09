"""Scope is enforced before lexical scoring or bounded relation traversal."""
import math
import re
from collections import Counter
from .contracts import ContractError


def terms(text):
    return re.findall(r"[a-z0-9_.]+|[\u3400-\u9fff]", str(text).lower())


class Retriever:
    def search_raw(self, atlas, query, entity, top_k=2, as_of_revision=None, min_revision=0):
        """Recall public observations without requiring a previously extracted Claim.

        Returned snippets are hypotheses to inspect, not confirmations of the predicate.
        Exact numeric entity boundaries prevent apple 1 from matching apple 10.
        """
        revision = atlas.revision if as_of_revision is None else as_of_revision
        if type(revision) is not int or not 0 <= revision <= atlas.revision:
            raise ContractError("Invalid evidence revision")
        if type(min_revision) is not int or min_revision < 0 or not 1 <= top_k <= 8:
            raise ContractError("Invalid raw retrieval limits")
        if min_revision > revision:
            return []
        name = entity.split(":", 1)[-1] if entity and entity != "task" else None
        match_entity = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I) if name else None
        wanted = set(terms(query))
        ranked = []
        for record in atlas.all_evidence():
            if record["kind"] != "observation" or record.get("terminal") or not min_revision <= record["revision"] <= revision:
                continue
            text = re.split(r"(?im)^\s*Your task is to\b", record["text"])[0]
            hit = match_entity.search(text) if match_entity else None
            if match_entity and not hit and record["entity"] != entity:
                continue
            overlap = wanted & set(terms(text))
            if not overlap and not hit and record["id"] != query:
                continue
            center = hit.start() if hit else next((m.start() for m in re.finditer(r"\w+", text)
                                                  if m.group().lower() in wanted), 0)
            start = max(0, center - 64)
            end = min(len(text), start + 256)
            if not start < end:
                continue
            ranked.append((len(overlap), record["revision"], {**record, "matched_spans": [{
                "id": record["id"], "start": start, "end": end}], "retrieval_path": [record["id"]],
                "support_status": "unverified_source"}))
        ranked.sort(key=lambda x: (-x[0], -x[1], x[2]["id"]))
        return [r for _, _, r in ranked[:top_k]]

    def search(self, atlas, query, entity=None, top_k=8, max_hops=0,relation_types=None,
               predicate=None, as_of_revision=None):
        if not 1 <= top_k <= 8 or not 0 <= max_hops <= 2:
            raise ContractError("Retrieval limits exceeded")
        revision = atlas.revision if as_of_revision is None else as_of_revision
        if type(revision) is not int or not 0 <= revision <= atlas.revision:
            raise ContractError("Invalid evidence revision")
        eligible = None
        if predicate is not None:
            eligible = {}
            for claim in atlas.nodes("Claim", active=as_of_revision is None):
                if claim["predicate"] == predicate and (entity is None or claim["entity"] == entity) and (
                        claim["revision"] <= revision < claim.get("valid_until_revision", atlas.revision + 1)):
                    for ref in claim["evidence_refs"]:
                        eligible.setdefault(ref["id"], []).append(ref)
        documents = [e for e in atlas.all_evidence() if e["revision"] <= revision and (
            e["id"] in eligible if eligible is not None else entity is None or e["entity"] == entity)]
        if eligible is not None:
            documents = [{**record, "matched_spans": eligible[record["id"]]} for record in documents]
        by_id = {e["id"]: e for e in documents}
        tokens = [terms(e["text"]) for e in documents]
        avg = sum(map(len, tokens)) / max(1, len(tokens))
        wanted = terms(query)
        frequencies = Counter(w for words in tokens for w in set(words))
        scored = []
        for record, words in zip(documents, tokens):
            counts = Counter(words)
            score = sum(math.log(1 + (len(tokens) - frequencies[w] + .5) / (frequencies[w] + .5))
                        * counts[w] * 2.2 / (counts[w] + 1.2 * (.25 + .75 * len(words) / max(1, avg)))
                        for w in set(wanted) if counts[w])
            if record["id"] == query:
                score += 100
            if predicate is not None:
                score += 1  # Slot provenance can identify relevant spans without a lexical hit.
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], -pair[1]["revision"], pair[1]["id"]))
        # Reserve result capacity for graph expansion; previously k lexical seeds filled
        # every slot, making max_hops a no-op whenever at least k documents matched.
        seeds=max(1,(top_k+1)//2) if max_hops else top_k
        selected = {e["id"]: {**e,"retrieval_path":[e["id"]]} for _, e in scored[:seeds]}
        frontier = set(selected)
        paths={eid:[eid] for eid in frontier};visited=set(frontier)
        for _ in range(max_hops):
            new = set()
            for row in atlas.db.execute("SELECT src,relation,dst FROM edges ORDER BY src,relation,dst"):
                if relation_types is not None and row[1] not in relation_types:continue
                if row[0] in frontier or row[2] in frontier:
                    origin=row[0] if row[0] in frontier else row[2]
                    for nid in (row[0], row[2]):
                        if nid in visited:continue
                        node=atlas.node(nid)
                        if as_of_revision is None and not node["active"]:continue
                        if node.get("revision", 0) > revision or node.get("valid_until_revision", atlas.revision + 1) <= revision:continue
                        if len(visited)+len(new)>=top_k*8:continue
                        paths.setdefault(nid,paths[origin]+[row[1],nid])
                        if nid in by_id and nid not in selected:
                            record = by_id[nid]
                            if len(selected) < top_k:
                                selected[nid] = {**record,"retrieval_path":paths[nid]}
                        new.add(nid)
            visited.update(new)
            frontier = new
        for _,record in scored:
            if len(selected)>=top_k:break
            selected.setdefault(record["id"],{**record,"retrieval_path":[record["id"]]})
        values=list(selected.values())
        lexical=[r for r in values if len(r["retrieval_path"])==1]
        expanded=[r for r in values if len(r["retrieval_path"])>1]
        mixed=[]
        for i in range(max(len(lexical),len(expanded))):
            if i<len(lexical):mixed.append(lexical[i])
            if i<len(expanded):mixed.append(expanded[i])
        return mixed[:top_k]

