"""Ground teacher diagnosis in public spans, with the menu and projection held apart."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import re

from scope.contracts import ContractError
from scope.data import SplitRegistry
from scope.io import atomic_json, atomic_jsonl, canonical, digest, file_sha, read_jsonl
from scope.memory_gaps import diagnose
from scope.teacher import LocalTeacher

from experiments.memory_curriculum.gap_study import evaluate, labeled, teacher_contrast


def teacher_choice(teacher, instruction, payload, choices):
    """Decode one allowed answer at the next-token boundary."""
    tokenizer, torch = teacher.tokenizer, teacher.torch
    messages = [{"role":"system", "content":instruction + " Respond with exactly one option letter."},
                {"role":"user", "content":canonical({**payload, "options":choices})}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
    options = [tokenizer.encode(letter, add_special_tokens=False) for letter in choices]
    if any(len(tokens) != 1 for tokens in options) or len({tokens[0] for tokens in options}) != len(choices):
        raise ContractError("Teacher choice labels must have distinct single-token IDs")
    inputs = torch.tensor([ids], device=teacher.device)
    with torch.inference_mode():
        logits = teacher.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
                               logits_to_keep=1).logits[0, -1]
        chosen = int(logits[[tokens[0] for tokens in options]].argmax())
    return list(choices.values())[chosen]


def audit(teacher, row):
    """Check the label and, for positive claims, independently select a public witness."""
    prefix = row["prefix"]
    options = json.loads(row["messages"][-1]["content"])["verified_options"]
    kind = teacher_choice(
        teacher,
        "Diagnose why a memory operation is absent. First check whether retrieval is blocked by exhausted budget "
        "or already used in the state phase. Otherwise check the named target against public observations, then "
        "projected fields, then the menu. A stated goal or available option does not confirm selection. A fact "
        "visible but not projected is extractor_gap. A projected fact with a feasible but omitted action is menu_gap. "
        "Use undetermined if no target is named or the relevant evidence is incomplete.",
        prefix, {k:v["diagnosis"] for k,v in options.items()})
    result = {"teacher_class":kind, "accepted":kind == row["gold"], "citation_valid":False}
    if not result["accepted"] or kind not in {"extractor_gap", "menu_gap"}:
        return result
    spans = list(dict.fromkeys((record["id"], record["text"]) for record in prefix["public_prefix"]))
    choice_items = [None, *[{"id":eid,"quote":quote} for eid,quote in spans]]
    choice_items.sort(key=lambda v:digest([row["task_id"], row["revision"], v]))
    evidence = teacher_choice(
        teacher,
        "Select the public quotation that supports the target for the same entity. Null means no quotation supports it. "
        "An available option or a bare click is not proof of a selection.",
        {"target":prefix["target"], "entity":prefix.get("target_entity")},
        {chr(65+i):item for i,item in enumerate(choice_items)})
    exact = bool(evidence and (evidence["id"],evidence["quote"]) in spans and
                 evidence["id"] == row["witness_id"] and row.get("witness_token") and
                 row["witness_token"] in evidence["quote"])
    if not exact:
        return {**result,"accepted":False}
    return {**result,"citation_valid":True,"evidence_id":evidence["id"],"quote":evidence["quote"]}


def native_pairs(path, train_ids):
    """Use real training observations with explicit, controlled projection/menu mutations."""
    rows, seen = [], set()
    for trajectory in read_jsonl(path):
        task = trajectory["task_id"]
        if task not in train_ids:
            raise ContractError("Teacher received a prefix outside rl_train")
        if task in seen:
            continue
        records = [e for e in trajectory["evidence"] if e["kind"] == "observation" and not e.get("terminal")]
        for i, product in enumerate(records):
            entity = product["entity"]
            if not entity.startswith("product:"):
                continue
            pid = entity.split(":", 1)[1]
            pattern = re.compile(r"(?:^|\[SEP\])\s*(" + re.escape(pid) + r"\s*\[SEP\]\s*[^\[\n]+)")
            listing = next(((prev,m.group(1).strip()) for prev in reversed(records[:i])
                            if prev["entity"] == "task" and pid in prev.get("clickables", [])
                            and (m := pattern.search(prev["text"]))), None)
            if listing is None:
                continue
            prev, quote = listing
            target = "search result listing for " + entity
            start = prev["text"].index(quote)
            distractor = product["text"][:220]
            prefix = {"target":target,"target_entity":entity,"phase":"context","remaining_retrievals":4,
                      "state_lookup_used":False,"audit_projection_complete":True,
                      "public_prefix":[{"id":prev["id"],"entity":"task","kind":"observation","start":start,"text":quote},
                                       {"id":product["id"],"entity":entity,"kind":"observation","start":0,"text":distractor}],
                      "projected_fields":[],"menu_operations":["FINISH"]}
            row = {"task_id":task,"revision":product["revision"],"prefix":prefix,
                   "gold":diagnose(target=target,public_witness=True,extracted=False,menu_covers=False,allowed=True),
                   "witness_id":prev["id"],"witness_token":quote,"label_origin":"native_public_controlled_projection"}
            rows.append(labeled(row))
            seen.add(task)
            break
    return rows


def build(args, config):
    out, previous = args.output, args.previous
    source_receipt = json.loads((previous/"study_receipt.json").read_text())
    registry = SplitRegistry(config["data_root"],config.get("split_seed",233))
    train = list(read_jsonl(previous/"warmup.jsonl"))
    native = native_pairs(args.native_prefixes,set(registry.ids("rl_train")))
    real = list(read_jsonl(previous/"historical_audit.jsonl"))
    historical_ids = {row["task_id"] for row in real}
    dev_ids = set(source_receipt["dev_tasks"])
    if (historical_ids | dev_ids) & {row["task_id"] for row in [*train,*native]}:
        raise ContractError("Teacher training data overlaps an evaluation task")
    teacher = LocalTeacher(args.teacher_model,"cuda")
    # Verify real-source seeds and a balanced, fixed subset of positive controlled examples.
    controlled = sorted((row for row in train if row["gold"] in {"extractor_gap","menu_gap"}),
                        key=lambda row:digest(row["prefix"]))
    seeds = [*native, *[r for r in controlled if r["gold"]=="extractor_gap"][:8],
                     *[r for r in controlled if r["gold"]=="menu_gap"][:8]]
    contrasts = []
    for row in seeds:
        row["teacher_audit"] = audit(teacher,row)
        twin = teacher_contrast(row)
        if twin:
            contrasts.append(twin)
            if row in native:
                contrasts.append(row)
    for row in real:
        row["teacher_audit_v2"] = audit(teacher,row)
    # Keep the rule-only run's sample and class counts when comparing the two warmups.
    replaced = Counter(row["gold"] for row in contrasts)
    audited_seeds = {id(row) for row in seeds}
    removed = set()
    for row in sorted(train, key=lambda row:id(row) in audited_seeds):
        if replaced[row["gold"]]:
            replaced[row["gold"]] -= 1
            removed.add(id(row))
    if any(replaced.values()):
        raise ContractError("Not enough rule rows for a class-matched teacher substitution")
    train = [row for row in train if id(row) not in removed] + contrasts
    random.Random(233).shuffle(train)
    atomic_jsonl(out/"warmup.jsonl",train)
    atomic_jsonl(out/"historical_audit.jsonl",real)
    atomic_jsonl(out/"controlled_dev.jsonl",list(read_jsonl(previous/"controlled_dev.jsonl")))
    atomic_jsonl(out/"teacher_audits.jsonl",seeds)
    receipt = {"rule_rows":len(train)-len(contrasts),"teacher_augmentation_rows":len(contrasts),
               "teacher_verified_train":sum(r["teacher_audit"]["accepted"] for r in seeds),
               "teacher_train_reviewed":len(seeds),"native_public_seeds":len(native),
               "native_public_accepted":sum(r["teacher_audit"]["accepted"] for r in native),
               "teacher_historical_correct":sum(r["teacher_audit_v2"]["teacher_class"]==r["gold"] for r in real),
               "teacher_historical_accepted":sum(r["teacher_audit_v2"]["accepted"] for r in real),
               "historical_causes":dict(Counter(r["gold"] for r in real)),
               "train_tasks":sorted({r["task_id"] for r in train}),"controlled_dev_tasks":sorted(dev_ids),
               "historical_tasks":sorted(historical_ids),"native_source_sha256":file_sha(args.native_prefixes),
               "warmup_sha256":file_sha(out/"warmup.jsonl"),"teacher_identity":teacher.identity,
               "historical_dev_used_for_training":False}
    atomic_json(out/"study_receipt.json",receipt)
    print(canonical(receipt),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=("build","evaluate"))
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--native-prefixes",type=Path)
    parser.add_argument("--teacher-model")
    parser.add_argument("--before",type=Path)
    parser.add_argument("--after",type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    (build if args.command=="build" else evaluate)(args,config)
