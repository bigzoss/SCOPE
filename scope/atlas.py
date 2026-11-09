"""Versioned property-state atlas. SQL is storage, not a learned truth oracle."""
from __future__ import annotations
import json
import sqlite3
from .contracts import ContractError, Scope
from .io import canonical, digest, plain
from .public_data import assert_public_text

ONTOLOGY_VERSION = "scope-state-atlas-v2"
RELATIONS = {"ABOUT", "SUPPORTED_BY", "CONTRADICTS", "SUPERSEDES", "SUMMARIZES",
             "CONSTRAINED_BY", "USED_IN", "DERIVED_FROM", "TARGETS", "DIFFERS_FROM", "CITES"}


class StateAtlas:
    def __init__(self, scope: Scope, path=":memory:"):
        self.scope = scope
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, active INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS edges(src TEXT, relation TEXT, dst TEXT, UNIQUE(src,relation,dst));
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, proposal_id TEXT UNIQUE, revision INTEGER, body TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS node_kind ON nodes(kind,active);
        CREATE INDEX IF NOT EXISTS edge_dst ON edges(dst);
        CREATE TABLE IF NOT EXISTS views(id TEXT PRIMARY KEY, revision INTEGER, body TEXT NOT NULL, stale INTEGER NOT NULL);
        """)
        old = self.db.execute("SELECT value FROM meta WHERE key='scope'").fetchone()
        if old and json.loads(old[0]) != plain(scope):
            raise ContractError("Atlas belongs to another scope")
        old_ontology=self.db.execute("SELECT value FROM meta WHERE key='ontology'").fetchone()
        if old_ontology and json.loads(old_ontology[0])!=ONTOLOGY_VERSION:raise ContractError("Atlas ontology migration required")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('scope',?)", (canonical(scope),))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('revision','0')")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('ontology',?)", (canonical(ONTOLOGY_VERSION),))

    @property
    def revision(self):
        return int(self.db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])

    def _advance(self):
        rev = self.revision + 1
        self.db.execute("UPDATE meta SET value=? WHERE key='revision'", (str(rev),))
        self.db.execute("UPDATE views SET stale=1")
        self.db.execute("UPDATE nodes SET active=0 WHERE kind='MemoryView'")
        return rev

    def append_public(self, text, kind="observation", entity="task", **public_fields):
        if kind not in {"observation", "requirement", "action_receipt"}:
            raise ContractError("Only public observation kinds are admitted")
        # Callers pass strings and explicit public fields, never raw simulator result objects.
        if set(public_fields)-{"clickables","terminal"}:
            raise ContractError("Hidden or reserved metadata cannot enter public evidence")
        assert_public_text(text)
        if not isinstance(entity,str) or not entity:raise ContractError("Entity ID must be a nonempty string")
        if "terminal" in public_fields and type(public_fields["terminal"]) is not bool:raise ContractError("Terminal flag must be boolean")
        if "clickables" in public_fields and (not isinstance(public_fields["clickables"],list) or any(not isinstance(x,str) for x in public_fields["clickables"])):
            raise ContractError("Public clickables must be a list of strings")
        with self.db:
            rev = self._advance()
            eid = f"obs_{rev:06d}_{digest([self.scope.key,text,kind,entity])[:10]}"
            record = {"id": eid, "text": str(text), "kind": kind, "entity": entity,
                      "scope": plain(self.scope), "revision": rev, "content_hash": digest(str(text)), **public_fields}
            self.db.execute("INSERT INTO evidence VALUES(?,?)", (eid, canonical(record)))
            self.db.execute("INSERT INTO nodes VALUES(?,?,?,1)", (eid, "Observation", canonical(record)))
            entity_node = {"id": entity, "scope": plain(self.scope)}
            self.db.execute("INSERT OR IGNORE INTO nodes VALUES(?,?,?,1)", (entity, "Entity", canonical(entity_node)))
            self.db.execute("INSERT INTO edges VALUES(?,?,?)", (eid, "ABOUT", entity))
            self._event("observation", {"evidence_id": eid}, rev)
        return record

    def evidence(self, eid):
        if not isinstance(eid,str):raise ContractError("Evidence ID must be text")
        row = self.db.execute("SELECT body FROM evidence WHERE id=?", (eid,)).fetchone()
        if not row:
            raise ContractError("Unknown or out-of-scope evidence: " + str(eid))
        return json.loads(row[0])

    def all_evidence(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT body FROM evidence ORDER BY id")]

    def nodes(self, kind=None, active=True):
        sql = "SELECT body FROM nodes WHERE 1=1"
        params = []
        if kind:
            sql += " AND kind=?"; params.append(kind)
        if active:
            sql += " AND active=1"
        return [json.loads(r[0]) for r in self.db.execute(sql + " ORDER BY id", params)]

    def node(self, node_id):
        if not isinstance(node_id,str):raise ContractError("Node ID must be text")
        row = self.db.execute("SELECT body,active,kind FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            raise ContractError("Unknown atlas node")
        return {**json.loads(row[0]), "active": bool(row[1]), "kind": row[2]}

    def _event(self, kind, payload, revision, proposal_id=None):
        event = {"kind": kind, "payload": payload, "revision": revision}
        event_id = "event_" + digest([event, proposal_id])[:24]
        self.db.execute("INSERT INTO events VALUES(?,?,?,?)", (event_id, proposal_id, revision, canonical(event)))
        return {"id": event_id, **event}

    def commit(self, checked, expected_revision, proposal_id,proposal_hash=None):
        """Validator output only. Idempotence precedes CAS for a genuine retry."""
        with self.db:
            self.db.execute("UPDATE meta SET value=value WHERE key='revision'")
            existing = self.db.execute("SELECT body FROM events WHERE proposal_id=?", (proposal_id,)).fetchone()
            if existing:
                saved = json.loads(existing[0])
                if saved["payload"].get("checked_hash") != digest(checked):
                    raise ContractError("Proposal ID reused with different content")
                return {"idempotent": True, **saved}
            if self.revision != expected_revision:
                raise ContractError("STALE_REVISION")
            rev = self._advance()
            node = {**checked, "scope": plain(self.scope), "revision": rev}
            node_id = node["id"]
            superseded=node.get("supersedes")
            if superseded:
                old=self.node(superseded)
                if old["kind"]!="Claim" or not old["active"] or old.get("protected"):raise ContractError("Invalid supersession")
                self.db.execute("UPDATE nodes SET active=0 WHERE id=?",(superseded,))
                self.db.execute("INSERT INTO edges VALUES(?,?,?)",(node_id,"SUPERSEDES",superseded))
            previous = self.nodes("Claim") if node["kind"] == "Claim" else []
            for old in previous:
                if node["predicate"] in {"price","selected_color","requirement"} and (old["entity"], old["predicate"]) == (node["entity"], node["predicate"]) and old["value"] != node["value"]:
                    # A value changing over time is not itself a logical contradiction.
                    relation="DIFFERS_FROM"
                    if old.get("observed_at_revision")==node.get("observed_at_revision"):
                        relation="CONTRADICTS";node["conflicted"]=True;old["conflicted"]=True
                        self.db.execute("UPDATE nodes SET body=? WHERE id=?",(canonical(old),old["id"]))
                    self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (node_id, relation, old["id"]))
            self.db.execute("INSERT INTO nodes VALUES(?,?,?,1)", (node_id, node["kind"], canonical(node)))
            for ref in node.get("evidence_refs", []):
                self.evidence(ref["id"])
                relation="SUPPORTED_BY" if node.get("support_status")=="supported" and node.get("validation_mode")!="semantic_check_ablated" else "CITES"
                self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (node_id, relation, ref["id"]))
            self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (node_id, "ABOUT", node.get("entity", "task")))
            return self._event("commit", {"node_id": node_id, "checked_hash": digest(checked),"proposal_hash":proposal_hash}, rev, proposal_id)

    def retire(self, node_id, expected_revision, proposal_id,proposal_hash=None):
        node = self.node(node_id)
        if node["kind"] != "Claim" or node.get("protected"):
            raise ContractError("Protected evidence/requirements cannot be retired")
        with self.db:
            old=self.db.execute("SELECT body FROM events WHERE proposal_id=?",(proposal_id,)).fetchone()
            if old:
                saved=json.loads(old[0])
                if saved["kind"]!="retire" or saved["payload"]["node_id"]!=node_id:
                    raise ContractError("Proposal ID reused with different content")
                return {"idempotent":True,**saved}
            if self.revision != expected_revision:
                raise ContractError("STALE_REVISION")
            rev = self._advance()
            self.db.execute("UPDATE nodes SET active=0 WHERE id=?", (node_id,))
            return self._event("retire", {"node_id": node_id,"proposal_hash":proposal_hash}, rev, proposal_id)

    def save_view(self, selected_ids, summary=None):
        for nid in selected_ids:
            if not self.node(nid)["active"]:
                raise ContractError("Cannot use inactive node")
        body = {"atlas_revision": self.revision, "selected_ids": selected_ids, "summary": summary}
        vid = "view_" + digest(body)[:24]
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO views VALUES(?,?,?,0)", (vid, self.revision, canonical(body)))
            self.db.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,1)",(vid,"MemoryView",canonical({"id":vid,**body})))
            for nid in selected_ids:
                self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)",(vid,"SUMMARIZES",nid))
        return {"id": vid, **body}

    def record_use(self,view_id,command,policy_version):
        self.view(view_id)
        event={"id":"action_"+digest([view_id,command,self.revision,policy_version])[:24],"command":command,
               "status":"proposed_action","policy_version":policy_version,"scope":plain(self.scope)}
        with self.db:
            self.db.execute("INSERT INTO nodes VALUES(?,?,?,1)",(event["id"],"ActionEvent",canonical(event)))
            self.db.execute("INSERT INTO edges VALUES(?,?,?)",(view_id,"USED_IN",event["id"]))
            self._event("use_view",{"view_id":view_id,"action_id":event["id"]},self._advance())
        return event

    def view(self, vid):
        row = self.db.execute("SELECT body,stale FROM views WHERE id=?", (vid,)).fetchone()
        if not row or row[1]:
            raise ContractError("STALE_OR_UNKNOWN_VIEW")
        return json.loads(row[0])

    def export(self):
        return {"scope": plain(self.scope), "ontology": ONTOLOGY_VERSION,
                "tables": {name: [list(r) for r in self.db.execute(f"SELECT * FROM {name} ORDER BY "+("src,relation,dst" if name=="edges" else "1"))]
                           for name in ("meta", "evidence", "nodes", "edges", "events", "views")}}

    @classmethod
    def restore(cls, snapshot, path=":memory:"):
        if snapshot["ontology"] != ONTOLOGY_VERSION:
            raise ContractError("Ontology mismatch")
        atlas = cls(Scope(**snapshot["scope"]), path)
        if set(snapshot["tables"])!={"meta","evidence","nodes","edges","events","views"}:
            raise ContractError("Incomplete snapshot tables")
        with atlas.db:
            for name, rows in snapshot["tables"].items():
                if name not in {"meta", "evidence", "nodes", "edges", "events", "views"}:
                    raise ContractError("Invalid snapshot table")
                atlas.db.execute(f"DELETE FROM {name}")
                for row in rows:
                    atlas.db.execute(f"INSERT INTO {name} VALUES({','.join('?' for _ in row)})", row)
            meta={r[0]:r[1] for r in atlas.db.execute("SELECT * FROM meta")}
            if json.loads(meta["scope"])!=plain(atlas.scope) or json.loads(meta["ontology"])!=ONTOLOGY_VERSION:
                raise ContractError("Snapshot metadata/scope mismatch")
            if atlas.revision<0:raise ContractError("Invalid atlas revision")
            for row in atlas.db.execute("SELECT id,body FROM evidence"):
                record=json.loads(row[1])
                if record["id"]!=row[0] or record["scope"]!=plain(atlas.scope) or record["content_hash"]!=digest(record["text"]):
                    raise ContractError("Snapshot evidence identity mismatch")
                assert_public_text(record["text"])
        return atlas

    def fingerprint(self):
        return digest(self.export())

    def close(self):
        self.db.close()
