"""Offline teacher recommendations select probes, never candidate actions or rewards."""
from .contracts import ContractError
from .io import digest, file_sha, read_jsonl


def prefix_signature(atlas, working, goal, observation, phase):
    claims = sorted((c["entity"], c["predicate"], str(c["value"]), c["support_status"],
                     bool(c.get("conflicted")), sorted(atlas.evidence(r["id"])["content_hash"] for r in c["evidence_refs"]))
                    for c in atlas.nodes("Claim"))
    return digest([atlas.scope.task, phase, goal, observation, working.get("public_state", {}), claims])


class TeacherProbeSelector:
    def __init__(self, specification, allowed_tasks):
        path = specification["path"]
        if file_sha(path) != specification["sha256"]:
            raise ContractError("Teacher probe manifest checksum mismatch")
        allowed = set(map(str, allowed_tasks))
        self.keys = set()
        self.matched = 0
        for row in read_jsonl(path):
            if str(row["task_id"]) not in allowed or row.get("source") != "teacher_public_probe":
                raise ContractError("Teacher probe manifest crosses training partition or provenance")
            if row.get("phase") not in {"state", "context"} or not isinstance(row.get("prefix_hash"), str):
                raise ContractError("Invalid teacher probe entry")
            self.keys.add(row["prefix_hash"])

    def __call__(self, atlas, working, goal, observation, phase):
        match = prefix_signature(atlas, working, goal, observation, phase) in self.keys
        self.matched += int(match)
        return match
