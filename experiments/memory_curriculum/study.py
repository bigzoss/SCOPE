"""Build and audit memory-tool demonstrations and independent paired probes."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random

from scope.atlas import StateAtlas
from scope.contracts import Budget, ContractError, Scope
from scope.data import SplitRegistry
from scope.environments import ShopActions
from scope.io import atomic_json, atomic_jsonl, canonical, digest, file_sha, read_jsonl
from scope.rollout import manager_messages
from scope.state_graph import apply_observation
from scope.teacher import LocalTeacher, TeacherDistillation
from scope.tools import ToolExecutor, context_view, parse_tool


CASES = ("wrong_entity", "stale_state", "pending", "preserve_constraints", "sufficient_state")


def reference(record):
    return {"id": record["id"], "start": 0, "end": len(record["text"])}


def build_case(task, split, case, variant=0):
    identity = digest([task, case, variant])
    atlas = StateAtlas(Scope(split, task, identity[:16]))
    tools = ToolExecutor(adapter=ShopActions())
    budget = Budget()
    colors = ["黑色", "蓝色", "白色", "红色", "灰色", "绿色"]
    wanted = colors[int(identity[12:14], 16) % len(colors)]
    old = colors[(colors.index(wanted) + 1 + variant) % len(colors)]
    cap = 200 + int(identity[16:18], 16)
    first, current = ("product:" + str(int(identity[i:i+9], 16) + 10**10) for i in (0, 20))
    goal = f"颜色是{wanted}，预算不超过{cap}元。"
    working = {"state_graph":True, "current_entity":current, "domain_predicates":sorted(tools.validator.domain_predicates)}

    def add(text, entity="task", clickables=()):
        return atlas.append_public(text, "observation", entity, clickables=list(clickables))

    def selected(color, entity=current):
        return add(f"已选颜色：{color}", entity)

    def update(record, value, status="supported", entity=current):
        return {"operation":"PROPOSE_STATE_UPDATE", "expected_revision":atlas.revision, "entity":entity,
            "predicate":"selected_color", "value":value, "support_status":status,"evidence_refs":[reference(record)]}

    req = atlas.append_public(goal, "requirement")
    tools.execute(atlas, working, {"operation":"PROPOSE_STATE_UPDATE", "expected_revision":atlas.revision,
        "entity":"task", "predicate":"requirement", "value":goal, "support_status":"supported",
        "evidence_refs":[reference(req)]}, "state", budget, goal)

    phase = "state"
    if case == "wrong_entity":
        other_page = selected(wanted, first)
        apply_observation(tools, atlas, working, other_page, budget, goal)
        observed = selected(old)
        good = update(observed, old)
        wrong = update(other_page, wanted)
    elif case == "stale_state":
        earlier = selected(old)
        if variant % 2 == 0:
            apply_observation(tools, atlas, working, earlier, budget, goal)
        observed = selected(wanted)
        if variant % 2:
            apply_observation(tools, atlas, working, observed, budget, goal)
        good = update(observed, wanted)
        wrong = update(earlier, old)
    elif case == "pending":
        atlas.append_public(f"click[{wanted}]", "action_receipt", current)
        observed = add(f"可选颜色：{wanted}、{old}。", current, [wanted, old])
        good = update(observed, wanted, "pending")
        wrong = update(observed, wanted)
    elif case == "preserve_constraints":
        phase = "context"
        old_page = add(f"价格：{cap-25}元\nBuy Now", current, ["Buy Now"])
        apply_observation(tools, atlas, working, old_page, budget, goal)
        observed = add(f"{wanted}可点击，页面没有显示已选颜色。", current, [wanted])
        working["observations_since_summary"] = 7
        good = {"operation":"SUMMARIZE", "expected_revision":atlas.revision,
                "items":[{"text":r["text"], "evidence_ref":reference(r)} for r in (old_page, observed)]}
        wrong = {**good,"items":good["items"][:1]}
    else:
        add(f"价格：{cap-25}元\nBuy Now", current, ["Buy Now"])
        observed = selected(wanted)
        for record in atlas.all_evidence()[1:]:
            apply_observation(tools, atlas, working, record, budget, goal)
        good = {"operation":"FINISH","expected_revision":atlas.revision}
        wrong = {"operation":"RETRIEVE","expected_revision":atlas.revision,"query":wanted,"top_k":1}

    messages = manager_messages(atlas, observed["text"], working, goal, phase, budget)
    snapshot = atlas.export()
    atlas.close()
    return {"task_id":task,"case":case,"variant":variant,"phase":phase,"messages":messages,
        "atlas":snapshot,"working":working,"goal":goal,"budget":vars(budget),"gold":good,"contrast":wrong}


def score(case, completion):
    """Score tool acceptance independently of accomplishing the intended repair."""
    tools = ToolExecutor(adapter=ShopActions())
    atlas = StateAtlas.restore(case["atlas"])
    working, budget, gold = copy.deepcopy(case["working"]), Budget(**case["budget"]), case["gold"]
    try:
        proposal = parse_tool(completion)
        TeacherDistillation.verify(atlas, working, case["goal"], case["phase"], budget, case["messages"], tools, proposal)
        result = tools.execute(atlas, working, proposal, case["phase"], budget, case["goal"])
        operation = proposal["operation"]
        effective = operation == gold["operation"]
        if effective and operation == "PROPOSE_STATE_UPDATE":
            node = atlas.node(result["commit"]["payload"]["node_id"])
            effective = all(node[k] == gold[k] for k in ("entity", "predicate", "value", "support_status")) and (
                {r["id"] for r in node["evidence_refs"]} == {r["id"] for r in gold["evidence_refs"]})
        if effective and operation == "SUMMARIZE":
            view = context_view(atlas, working, case["goal"])
            summary = view["summary"]
            effective = bool(summary and all(any(
                item["evidence_ref"]["id"] == required["evidence_ref"]["id"] and
                item["evidence_ref"]["start"] <= required["evidence_ref"]["start"] and
                item["evidence_ref"]["end"] >= required["evidence_ref"]["end"]
                for item in summary["items"]) for required in gold["items"]) and
                any(row["constraint"] == "color" and row["status"] == "pending" for row in view["constraint_checklist"]) and
                any(row["constraint"] == "maximum_price_cny" and row["status"] == "confirmed" for row in view["constraint_checklist"]))
        return {"legal":True,"effective":bool(effective),"operation":operation,"reason":None}
    except (ContractError, KeyError, TypeError, ValueError, AttributeError) as exc:
        return {"legal":False,"effective":False,"operation":None,"reason":str(exc)}
    finally:
        atlas.close()


def teacher_review(teacher, case):
    options = [case["gold"], case["contrast"]]
    if int(digest(case["messages"])[0], 16) % 2:
        options.reverse()
    request = [{"role":"system","content": "Choose the memory operation that safely addresses the public state. "
        "Cite the current entity and revision. Preserve unresolved hard constraints in summaries. "
        'Output only {"choice":1} or {"choice":2}.'},
        {"role":"user","content":canonical({"student_messages":case["messages"], "options":options})}]
    try:
        answer = parse_tool(teacher(request))
        if set(answer) == {"choice"} and type(answer["choice"]) is int and answer["choice"] in (1, 2):
            return score(case, canonical(options[answer["choice"] - 1]))["effective"]
    except (TypeError, ValueError):
        pass
    return False


def build(config, output, old_teacher, old_protocol, teacher_model=None):
    output.mkdir(parents=True, exist_ok=True)
    registry = SplitRegistry(config["data_root"], config.get("split_seed", 233))
    old = json.loads(old_protocol.read_text())
    excluded = set(map(str, old["source_diagnostic_tasks"]))
    train_ids = sorted(registry.ids("rl_train"), key=lambda t:digest(["train", t]))[:16]
    dev_ids = sorted(set(registry.ids("internal_dev")) - excluded, key=lambda t:digest(["dev", t]))[:6]
    assert len(train_ids) == 16 and len(dev_ids) == 6 and not set(train_ids) & set(dev_ids)
    teacher = LocalTeacher(teacher_model, "cuda") if teacher_model else None
    records, audit = [], Counter()
    for index, task in enumerate(train_ids):
        for case_name in CASES:
            case = build_case(task, "rl_train", case_name, index % 3)
            if not score(case, canonical(case["gold"]))["effective"] or score(case, canonical(case["contrast"]))["effective"]:
                raise ContractError("Invalid curriculum contrast: " + case_name)
            reviewed = teacher_review(teacher, case) if teacher and index < 3 else None
            if reviewed is not None:
                audit["reviewed:" + case_name] += 1
                if reviewed:
                    audit["agreed:" + case_name] += 1
            records.append({"task_id":task,"phase":case["phase"],"case":case_name,"messages":case["messages"],
                            "completion":canonical(case["gold"]),"teacher_agreement":reviewed,
                            "source":"state_graph_public_prefix","label_origin":"controlled_verified"})
    real = [r for r in read_jsonl(old_teacher) if r["source"] == "teacher_state_graph_public_prefix"]
    if any(r["task_id"] not in set(registry.ids("rl_train")) or r["task_id"] in dev_ids for r in real):
        raise ContractError("Teacher corpus outside training partition")
    records.extend(real)
    random.Random(233).shuffle(records)
    atomic_jsonl(output / "warmup.jsonl", records)
    probes = [build_case(task, "internal_dev", name, 3 + index % 2) for index, task in enumerate(dev_ids) for name in CASES]
    atomic_jsonl(output / "controlled_dev.jsonl", probes)
    manifest = {"train_tasks":sorted({r["task_id"] for r in records}),"dev_tasks":dev_ids,
        "legacy_dev_excluded":sorted(excluded),"controlled_train":len(records)-len(real),"actual_teacher_train":len(real),
        "teacher_audit":dict(audit),"teacher_identity":teacher.identity if teacher else None,
        "controlled_dev":len(probes),"warmup_sha256":file_sha(output/"warmup.jsonl"),
        "controlled_dev_sha256":file_sha(output/"controlled_dev.jsonl"),"seed":233,
        "gate":{"controlled_legal":.8,"controlled_effective":.6,"each_case_effective":.5,
                "live_legal":.75,"live_useful_actions_min":4,"allow_strict_success_drop":False}}
    atomic_json(output / "study_receipt.json", manifest)
    print(canonical(manifest), flush=True)


def probe(config, output, before, after):
    import torch
    from scope.checkpoint import load_checkpoint
    from scope.policy import HFPolicy

    receipt = json.loads((output/"study_receipt.json").read_text())
    path = output/"controlled_dev.jsonl"
    if file_sha(path) != receipt["controlled_dev_sha256"]:
        raise ContractError("Controlled dev data changed after registration")
    rows = read_jsonl(path)
    policy = HFPolicy(config["model"]["path"])
    all_results = []
    for name, checkpoint in (("before", before), ("after", after)):
        load_checkpoint(checkpoint, policy, restore_random=False)
        actor, manager = policy.fingerprint("actor"), policy.fingerprint("manager")
        for index, row in enumerate(rows):
            for repetition in range(2):
                seed = 700 + 2 * index + repetition
                torch.manual_seed(seed)
                sample = policy.sample(row["messages"], "manager", row["phase"], digest([row["task_id"], row["case"]]),
                    config["runtime"]["manager_max_input"], config["runtime"]["manager_max_new"], greedy=False)
                all_results.append({"condition":name,"task_id":row["task_id"],"case":row["case"],"seed":seed,
                    "actor_sha256":actor,"manager_sha256":manager,"tokens":len(sample.completion_ids),
                    "completion":sample.text, **score(row, sample.text)})
        if policy.fingerprint("actor") != actor or policy.fingerprint("manager") != manager:
            raise ContractError("Model changed during frozen probe")
        atomic_jsonl(output/"controlled_results.jsonl", all_results)
    if len({row["actor_sha256"] for row in all_results}) != 1:
        raise ContractError("Actor checkpoint differs across conditions")
    report = {}
    for name in ("before", "after"):
        group = [r for r in all_results if r["condition"] == name]
        report[name] = {"n":len(group),"legal":sum(r["legal"] for r in group),"effective":sum(r["effective"] for r in group),
            "cases":{case:{"n":sum(r["case"]==case for r in group),
                           "legal":sum(r["case"]==case and r["legal"] for r in group),
                           "effective":sum(r["case"]==case and r["effective"] for r in group)} for case in CASES}}
    atomic_json(output/"controlled_summary.json", report)
    print(canonical(report), flush=True)


def live(config, output, before, after):
    from scope.checkpoint import load_checkpoint
    from scope.environments import ShopEnvironment
    from scope.evaluation import evaluate, paired_comparison
    from scope.launcher import own_service
    from scope.policy import HFPolicy

    receipt = json.loads((output/"study_receipt.json").read_text())
    tasks = receipt["dev_tasks"]
    registry = SplitRegistry(config["data_root"], config.get("split_seed",233))
    if set(tasks) & set(receipt["train_tasks"]) or not set(tasks) <= set(registry.ids("internal_dev")):
        raise ContractError("Live dev split is no longer independent")

    class FixedDev:
        def ids(self, split):
            return tasks

        def evaluation_sequences(self, split, length, seed):
            return [[task] for task in tasks]

    policy = HFPolicy(config["model"]["path"])
    actors, report = {}, {}
    with own_service(config, output/"live_service", True):
        env = ShopEnvironment(config["environment"]["base_url"], tasks, file_sha(config["environment"]["manifest"]))
        env.health()
        for name, checkpoint in (("before", before), ("after", after)):
            load_checkpoint(checkpoint, policy, restore_random=False)
            actors[name] = policy.fingerprint("actor")
            folder = output/("live_"+name)
            result = evaluate(policy, env, FixedDev(), config, folder, split="internal_dev", limit=len(tasks))
            counts = Counter()
            for row in read_jsonl(folder/"results.jsonl"):
                visible = {eid for step in row["history"] if step["phase"]=="execute"
                           for eid in step.get("context_evidence_ids", [])}
                for event in row["manager_operations"]:
                    if not event["sampled"]:
                        continue
                    tool = event["result"]
                    counts["sampled"] += 1
                    if not tool["accepted"]:
                        counts["rejected"] += 1
                        continue
                    op = tool.get("operation")
                    counts["legal"] += 1
                    counts["operation:"+str(op)] += 1
                    if op in {"RETRIEVE", "FETCH_EVIDENCE"}:
                        records = tool.get("retrieved", [tool["evidence"]] if "evidence" in tool else [])
                        counts["useful"] += bool({r["id"] for r in records} & visible)
                    elif op in {"PROPOSE_STATE_UPDATE", "ADD_MEMORY", "UPDATE_MEMORY"}:
                        counts["useful"] += tool.get("commit", {}).get("kind") == "commit"
                    elif op == "SUMMARIZE":
                        counts["useful"] += bool(set(tool.get("summary", {}).get("source_hashes", {})) & visible)
            report[name] = {"n":result["n"],"success":result["success_count"],"wrong_purchase_rate":result["wrong_purchase_rate"],
                            "mean_total_tokens":result["mean_total_tokens"],"manager":dict(counts)}
            print(name, canonical(report[name]), flush=True)
    if actors["before"] != actors["after"]:
        raise ContractError("Live actor is not shared across conditions")
    report["actor_sha256"] = actors["before"]
    report["paired"] = paired_comparison(output/"live_before", output/"live_after")
    atomic_json(output/"live_summary.json", report)
    controlled = json.loads((output/"controlled_summary.json").read_text())["after"]
    thresholds = receipt["gate"]
    managed = report["after"]["manager"]
    checks = {"controlled_legal":controlled["legal"]/controlled["n"] >= thresholds["controlled_legal"],
              "controlled_effective":controlled["effective"]/controlled["n"] >= thresholds["controlled_effective"],
              "each_case_effective":all(c["effective"]/c["n"] >= thresholds["each_case_effective"] for c in controlled["cases"].values()),
              "live_legal":bool(managed.get("sampled")) and managed.get("legal",0)/managed["sampled"] >= thresholds["live_legal"],
              "live_useful_actions":managed.get("useful",0) >= thresholds["live_useful_actions_min"],
              "strict_success_not_lower":report["after"]["success"] >= report["before"]["success"]}
    atomic_json(output/"grpo_gate.json", {"passed":all(checks.values()),"checks":checks})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("build","probe","live"))
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--old-teacher",type=Path)
    p.add_argument("--old-protocol",type=Path)
    p.add_argument("--teacher-model")
    p.add_argument("--before",type=Path)
    p.add_argument("--after",type=Path)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    if args.command=="build":
        build(config,args.output,args.old_teacher,args.old_protocol,args.teacher_model)
    elif args.command=="probe":
        probe(config,args.output,args.before,args.after)
    else:
        live(config,args.output,args.before,args.after)


if __name__=="__main__":
    main()
