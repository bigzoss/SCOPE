"""Frozen, provenance-checked training experience; never transfer task answers."""
from __future__ import annotations
from .contracts import ContractError
from .io import digest,read_jsonl,file_sha
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
