"""Verified teacher answers on the student's own public answer context."""
import json
from pathlib import Path

from scope.contracts import ContractError
from scope.hotpot import answer_messages, score
from scope.io import atomic_json, atomic_jsonl, digest, file_sha


def prepare_examples(parent, tasks, dev_ids, destination):
    diagnostic = parent / "reader_diagnostic"
    summary = json.loads((diagnostic / "summary.json").read_text())
    source = diagnostic / "results.jsonl"
    if not summary["complete"] or file_sha(source) != summary["results_sha256"]:
        raise ContractError("Teacher diagnostic results failed verification")
    examples, rejected = [], []
    for row in map(json.loads, source.read_text().splitlines()):
        task_id = row["task_id"]
        if task_id not in tasks or task_id in dev_ids:
            raise ContractError("Teacher answer lies outside the training split")
        public, gold = tasks[task_id]
        probe = json.loads((parent / "probes" / row["probe"]).read_text())
        label = probe["proposal"]["proposal"]["choice"]
        branch = probe["forced"][label]
        view = branch["trace"]["answer_view"]
        measured = score({"answer":row["teacher_answer"],"answer_view":view},gold)
        if digest(view) != row["public_view_sha256"] or measured != row["teacher_score"]:
            raise ContractError("Teacher response or its public evidence was altered")
        if measured["em"] != 1 or measured["support_visible_recall"] != 1 or row["teacher_finish_reason"] == "length":
            rejected.append({"task_id":task_id,"reason":"strict_answer_or_complete_support_or_termination"})
            continue
        paragraphs = dict(public["paragraphs"])
        citations = []
        for title, index in gold["supporting_facts"]:
            ref = next((item for item in view if item["title"] == title and index in item["complete_sentences"]), None)
            if ref is None:
                raise ContractError("A labeled supporting sentence is outside the student view")
            sentence = paragraphs[title][index]
            if sentence not in ref["text"]:
                raise ContractError("Supporting quotation does not match public text")
            citations.append({"evidence_id":ref["id"],"title":title,"sentence_index":index,"quote":sentence})
        examples.append({
            "task_id":task_id,"phase":"execute","source":"teacher_actor_public_prefix",
            "messages":answer_messages(public["question"],view),"completion":row["teacher_answer"],
            "environment_accepted":True,"task_success":1,
            "verification":{"strict_em":1,"support_visible_recall":1,
                            "citations":citations,"citation_origin":"offline official support annotation audit",
                            "public_view_sha256":digest(view),"teacher_model":"qwen3.8-flash-next",
                            "scorer_fields_excluded_from_messages":True},
        })
    if not examples or len({(x["task_id"],digest(x["messages"])) for x in examples}) != len(examples):
        raise ContractError("Empty or duplicated teacher answer examples")
    atomic_jsonl(destination / "actor_examples.jsonl", examples)
    report = {"accepted":len(examples),"rejected":rejected,"examples_sha256":file_sha(destination/"actor_examples.jsonl"),
              "source_results_sha256":file_sha(source),"task_ids":[x["task_id"] for x in examples]}
    atomic_json(destination/"data_validation.json",report)
    return report
