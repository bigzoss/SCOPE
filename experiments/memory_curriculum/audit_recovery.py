"""Replay public prefixes from existing results; no model/environment calls or success estimates.

python -m experiments.memory_curriculum.audit_recovery --results FILE --output FILE
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scope.alfworld import ALFWorldActions
from scope.atlas import StateAtlas
from scope.contracts import Scope
from scope.io import atomic_json, file_sha
from scope.recovery import prepare_recovery, record_recovery


def audit(path):
    counts = Counter()
    examples = []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        adapter = ALFWorldActions()
        evidence = {e["id"]:e for e in row["public_evidence"]}
        ordered = sorted(evidence.values(), key=lambda e:e["revision"])
        goal = next(e["text"] for e in ordered if e["kind"] == "requirement")
        with_atlas = StateAtlas(Scope("diagnostic", str(row["task_id"]), "public-prefix"))
        working = {"public_state":{}, "current_entity":"task", "decision_recovery":True}
        index = 0
        had_recovery = False
        try:
            for event in row["history"]:
                if event["phase"] != "execute":
                    continue
                before = evidence[event["observation_before_id"]]
                while index < len(ordered) and ordered[index]["revision"] <= before["revision"]:
                    item = ordered[index]
                    if item["kind"] in {"observation","requirement"}:
                        with_atlas.append_public(item["text"],item["kind"],item["entity"],
                                                 clickables=item.get("clickables",[]))
                    index += 1
                if not working["public_state"]:
                    working["public_state"] = adapter.observe(before["text"], {}, None, before.get("clickables",[]))
                plan = prepare_recovery(adapter, with_atlas, working, goal)
                counts["turns"] += 1
                counts["turns_with_public_candidates"] += bool(plan["candidates"])
                counts["logged_command_would_be_blocked"] += event["normalized_command"] in plan["blocked_commands"]
                if event.get("loop") and plan["candidates"]:
                    counts["loop_turns_with_alternatives"] += 1
                    had_recovery = True
                    if len(examples)<3:
                        examples.append({"task_id":row["task_id"],"command":event["normalized_command"],
                                         "candidates":plan["candidates"][:3]})
                from scope.io import digest
                action = {**event,"public_state_before":digest(adapter.progress_key(working["public_state"]))}
                record_recovery(working,action)
                if event["action_status"] == "accepted":
                    after = evidence[event["observation_after_id"]]
                    working["public_state"] = adapter.observe(after["text"],working["public_state"],
                                                               event["normalized_command"],after.get("clickables",[]))
            counts["tasks"] += 1
            counts["tasks_with_loop_alternative"] += had_recovery
        finally:
            with_atlas.close()
    return {"kind":"read_only_public_prefix_audit", "results_sha256":file_sha(path), "counts":dict(counts),
            "examples":examples, "interpretation":"Candidate availability only; no new rollout or success-rate measurement."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.results)
    atomic_json(args.output,result)
    print(json.dumps(result["counts"],ensure_ascii=False))
