"""Frozen, provenance-checked training experience; never transfer task answers."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from .contracts import ContractError
from .io import atomic_jsonl,digest,read_jsonl,file_sha
from .retrieval import terms
from .public_data import assert_public_text


class ExperienceBank:
    def __init__(self,path=None,allowed_training_tasks=()):
        self.records=read_jsonl(path) if path else []
        self.path=path
        self.file_hash=file_sha(path) if path else None
        allowed=set(map(str,allowed_training_tasks))
        forbidden={"target_asin","asin","goal_options","reward_detail","answer","options"}
        for record in self.records:
            def check(value):
                if isinstance(value,dict):
                    if forbidden & value.keys():raise ContractError("Task-answer fields in experience bank")
                    for v in value.values():check(v)
                elif isinstance(value,list):
                    for v in value:check(v)
            check(record)
            tasks=set(map(str,record.get("support_tasks",[])))
            if len(tasks)<2 or not tasks<=allowed:raise ContractError("Experience needs >=2 in-split training supports")
            if not record.get("text") or not record.get("preconditions"):
                raise ContractError("Experience missing content/preconditions")
            supports = record.get("support_records")
            if supports is not None and (not isinstance(supports, list) or
                    any(not isinstance(row, dict) or not row.get("observation_id") or not row.get("text_hash") for row in supports)
                    or {str(row.get("task_id")) for row in supports} != tasks):
                raise ContractError("Experience provenance does not match its training tasks")
            assert_public_text(record["text"])
            assert_public_text(str(record["preconditions"]))
            if not record.get("id"):record["id"]="experience_"+digest(record)[:20]
        if len({r["id"] for r in self.records})!=len(self.records):raise ContractError("Duplicate experience IDs")
        self.initial_hash=digest(self.records)

    def search(self,query,top_k=2):
        wanted=set(terms(query));scored=[]
        for record in self.records:
            score=len(wanted & set(terms(record["text"]+" "+str(record["preconditions"]))))
            if score:scored.append((score,record))
        scored.sort(key=lambda p:(-p[0],p[1]["id"]))
        return [{"id":r["id"],"text":r["text"],"preconditions":r["preconditions"],
                 "status":"training_experience_not_current_fact"} for _,r in scored[:top_k]]

    def assert_unchanged(self):
        if digest(self.records)!=self.initial_hash:raise ContractError("Frozen experience bank changed")
        if self.path and file_sha(self.path)!=self.file_hash:raise ContractError("Experience source file changed")


def build_experiences(public_file, output, allowed_training_tasks, adapter):
    """Distill only recurring public action/observation patterns into task-free guidance."""
    from .warmup import public_evidence

    if Path(output).exists():
        raise ContractError("Refusing to overwrite a frozen experience bank")
    allowed = set(map(str, allowed_training_tasks))
    rules = {
        "selected_color": ("After selecting a color, check the current product's explicit selection field.",
                           "product option clicked and selection feedback is available"),
        "selected_spec": ("Verify the selected specification against the user requirements before submitting.",
                          "product option clicked and selection feedback is available"),
        "held": ("Bind the observed object and container identifiers, then confirm pickup from public feedback.",
                 "an object is being taken from a known container"),
        "container_open": ("Verify the container's open or closed state from the public response before acting inside it.",
                            "a known container has just been opened or closed"),
    }
    supports = defaultdict(dict)
    for row in read_jsonl(public_file):
        task = str(row["task_id"])
        if task not in allowed:
            raise ContractError("Experience extraction requires training-only tasks")
        last_action = None
        for record in public_evidence(row):
            assert_public_text(record["text"])
            if record.get("scope", {}).get("task", task) != task:
                raise ContractError("Experience evidence task scope mismatch")
            if record.get("terminal"):
                break
            if record["kind"] == "action_receipt":
                last_action = record["text"]
            elif record["kind"] == "observation" and last_action:
                for fact in adapter.public_facts(record):
                    matches = ((fact.predicate in {"selected_color", "selected_spec"} and last_action.startswith("click[")) or
                               (fact.predicate == "held" and fact.value is True and last_action.startswith("take ")) or
                               (fact.predicate == "container_open" and last_action.startswith(("open ", "close "))))
                    if matches and fact.status == "supported":
                        supports[fact.predicate].setdefault(task, {"task_id": task, "observation_id": record["id"],
                                                                    "text_hash": digest(record["text"])})
                last_action = None
    records = [{"id":"experience_" + digest([rule, *rules[rule]])[:20], "text":rules[rule][0],
                "preconditions":rules[rule][1], "support_tasks":sorted(rows),
                "support_records":[rows[t] for t in sorted(rows)]}
               for rule, rows in sorted(supports.items()) if len(rows) >= 2]
    atomic_jsonl(output, records)
    return {"experiences": len(records), "sha256": file_sha(output)}
