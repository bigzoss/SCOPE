"""Scope is enforced before lexical scoring or bounded relation traversal."""
import math
import re
from collections import Counter
from .contracts import ContractError


def terms(text):
    return re.findall(r"[a-z0-9_.]+|[\u3400-\u9fff]", str(text).lower())


class Retriever:
    def search(self, atlas, query, entity=None, top_k=8, max_hops=0,relation_types=None):
        if not 1 <= top_k <= 8 or not 0 <= max_hops <= 2:
            raise ContractError("Retrieval limits exceeded")
        documents = [e for e in atlas.all_evidence() if entity is None or e["entity"] == entity]
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
                        if not node["active"]:continue
                        if len(visited)+len(new)>=top_k*8:continue
                        paths.setdefault(nid,paths[origin]+[row[1],nid])
                        if nid.startswith("obs_") and nid not in selected:
                            record = atlas.evidence(nid)
                            if entity is None or record["entity"] == entity:
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

