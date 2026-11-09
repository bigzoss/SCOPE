"""Train a memory selector and compare it on disjoint controlled and live development tasks."""
import argparse
import json
from pathlib import Path
import random

from scope.atlas import StateAtlas
from scope.candidate_actions import candidate_actions, choice_messages
from scope.contracts import Budget, ContractError
from scope.data import SplitRegistry
from scope.environments import ShopActions
from scope.io import atomic_json, atomic_jsonl, canonical, digest, file_sha, read_jsonl
from scope.tools import ToolExecutor
from .study import CASES, build_case, live, score


def fixture(task, split, case, variant):
    row = build_case(task, split, case, variant)
    atlas = StateAtlas.restore(row["atlas"])
    try:
        menu = candidate_actions(atlas, row["working"], row["goal"], row["phase"], Budget(**row["budget"]),
                                 ToolExecutor(adapter=ShopActions()))
    finally:
        atlas.close()
    gold = [label for label, proposal in menu.items() if score(row, canonical(proposal))["effective"]]
    if not gold or len(menu) < 2:
        raise ContractError(f"Curriculum case lacks a meaningful correct candidate: {case}")
    return row, menu, gold


def build(config, output, previous, original):
    registry = SplitRegistry(config["data_root"], config.get("split_seed", 233))
    earlier = json.loads(previous.read_text())
    excluded = set(map(str, json.loads(original.read_text())["source_diagnostic_tasks"])) | set(earlier["dev_tasks"])
    training = sorted(registry.ids("rl_train"), key=lambda task:digest(["menu_train", task]))[:24]
    dev = sorted(set(registry.ids("internal_dev")) - excluded, key=lambda task:digest(["menu_dev", task]))[:6]
    if len(dev) != 6 or len(training) != 24 or set(training) & set(dev):
        raise ContractError("Choice study requires disjoint training and development task IDs")
    examples = []
    for index, task in enumerate(training):
        for case in CASES:
            row, menu, gold = fixture(task, "rl_train", case, index % 3)
            examples.append({"task_id":task,"phase":row["phase"],"case":case,"messages":choice_messages(row["messages"], menu),
                             "completion":gold[0],"source":"state_graph_candidate_choice"})
    random.Random(233).shuffle(examples)
    dev_rows = []
    for index, task in enumerate(dev):
        for case in CASES:
            row, menu, gold = fixture(task, "internal_dev", case, 3 + index % 2)
            dev_rows.append({**row, "menu":menu, "gold_labels":gold, "choice_messages":choice_messages(row["messages"], menu)})
    atomic_jsonl(output/"warmup.jsonl",examples)
    atomic_jsonl(output/"controlled_dev.jsonl",dev_rows)
    atomic_json(output/"study_receipt.json",{"train_tasks":training,"dev_tasks":dev,"excluded_previous_dev":sorted(excluded),
                "warmup_sha256":file_sha(output/"warmup.jsonl"),"controlled_dev_sha256":file_sha(output/"controlled_dev.jsonl"),
                "controlled_train":len(examples),"controlled_dev":len(dev_rows),
                "gate":earlier["gate"],"gate_live_min_non_finish":4})


def probe(config, output, before, after):
    import torch
    from scope.checkpoint import load_checkpoint
    from scope.policy import HFPolicy

    path = output/"controlled_dev.jsonl"
    receipt = json.loads((output/"study_receipt.json").read_text())
    if file_sha(path) != receipt["controlled_dev_sha256"]:
        raise ContractError("Registered dev data changed")
    rows = read_jsonl(path)
    policy = HFPolicy(config["model"]["path"])
    results, report = [], {}
    for name, checkpoint in (("before", before), ("after", after)):
        load_checkpoint(checkpoint, policy, restore_random=False)
        initial = policy.fingerprint()
        actor = policy.fingerprint("actor")
        group = []
        for index, row in enumerate(rows):
            for repetition in range(2):
                seed = 700 + 2*index + repetition
                torch.manual_seed(seed)
                sample = policy.sample_choice(row["choice_messages"],row["phase"],digest(row["task_id"]),list(row["menu"]),
                                              config["runtime"]["manager_max_input"])
                result = score(row,canonical(row["menu"][sample.text]))
                group.append({"condition":name,"task_id":row["task_id"],"case":row["case"],"seed":seed,
                              "actor_sha256":actor,"option":sample.text,"candidate_count":len(row["menu"]),**result})
        if policy.fingerprint() != initial:
            raise ContractError("Policy changed during dev evaluation")
        results.extend(group)
        report[name] = {"n":len(group),"legal":sum(r["legal"] for r in group),"effective":sum(r["effective"] for r in group),
            "cases":{case:{"n":sum(r["case"]==case for r in group),"legal":sum(r["case"]==case and r["legal"] for r in group),
                           "effective":sum(r["case"]==case and r["effective"] for r in group)} for case in CASES}}
        print(name, canonical(report[name]), flush=True)
    if len({r["actor_sha256"] for r in results}) != 1:
        raise ContractError("Before/after actors differ")
    atomic_jsonl(output/"controlled_results.jsonl",results)
    atomic_json(output/"controlled_summary.json",report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("build","probe","live"))
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--original",type=Path)
    parser.add_argument("--before",type=Path)
    parser.add_argument("--after",type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.command=="build":
        build(config,args.output,args.previous,args.original)
    elif args.command=="probe":
        probe(config,args.output,args.before,args.after)
    else:
        live(config,args.output,args.before,args.after)


if __name__=="__main__":
    main()
