"""USTC teacher demonstrations with measured student-side counterfactual rewards."""
import argparse
from collections import Counter
from email.utils import parsedate_to_datetime
import fcntl
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import requests
import torch

from scope.checkpoint import load_checkpoint
from scope.cli import model_identity, source_hashes, verify_checkpoint_base
from scope.contracts import ContractError
from scope.hotpot import NOOP, STAGES, rollout, score, split_row
from scope.hotpot_intervention import continuation_tokens, coverage, factorial
from scope.io import atomic_json, atomic_jsonl, canonical, digest, file_sha
from scope.policy import HFPolicy
from scope.warmup import warmup


TEACHER_MODEL = "qwen3.8-flash-next"
SEED = 233
TASKS = 8
PROBE_COUNT = 3


class USTCTeacher:
    def __init__(self, config_path, cache):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        url = urlsplit(config["base_url"])
        if (url.scheme,url.hostname,url.path.rstrip("/")) != ("https","api.llm.ustc.edu.cn","/v1") or url.username:
            raise ContractError("Teacher must use the configured USTC inference service")
        self.url = config["base_url"].rstrip("/")+"/chat/completions"
        self.http = requests.Session()
        self.http.headers["Authorization"] = "Bearer " + config["api_key"]
        self.last_request = 0.
        self.counts = Counter()
        self.cache = cache

    def close(self):
        self.http.close()

    def __call__(self, messages, menu, public_options, working_sources):
        payload = json.loads(messages[-1]["content"])
        payload.update(option_sources=public_options,working_sources=working_sources)
        instruction = ("You are the memory controller for this phase. Choose one legal option. "
                       "During writing preserve an informative source for a future question. During compression preserve useful facts "
                       "and remove repetition. During retrieval prefer a missing hop, never invent a source. "
                       "Return only its single option letter.")
        dialogue = [{"role":"system","content":instruction},{"role":"user","content":canonical(payload)}]
        cache = self.cache/(digest([TEACHER_MODEL,dialogue])+".json")
        if cache.exists():
            previous = json.loads(cache.read_text())
            if (previous.get("model")==TEACHER_MODEL and previous.get("choice") in menu
                    and previous.get("alternative") in menu and previous["choice"]!=previous["alternative"]
                    and previous["evidence_id"]==public_options.get(previous["choice"],{}).get("source_id")):
                self.counts["cached_decisions"]+=1
                return previous
            raise ContractError("Cached USTC choice does not match its evidence menu")
        for attempt in range(5):
            time.sleep(max(0.,6.5-(time.monotonic()-self.last_request)))
            self.last_request = time.monotonic()
            self.counts["requests"]+=1
            try:
                response = self.http.post(self.url,json={"model":TEACHER_MODEL,"messages":dialogue,
                    "temperature":0,"max_tokens":48,"chat_template_kwargs":{"enable_thinking":False}},
                    timeout=(12,50),allow_redirects=False)
            except requests.RequestException:
                self.counts["transport_retries"]+=1
                time.sleep(2**attempt)
                continue
            if response.status_code in {429,500,502,503,504}:
                self.counts[f"http_{response.status_code}_retries"]+=1
                delay = 62. if response.status_code==429 else 2.**(attempt+1)
                header = response.headers.get("Retry-After")
                if header:
                    try:
                        try:
                            delay = float(header)
                        except ValueError:
                            delay = parsedate_to_datetime(header).timestamp()-time.time()
                    except (TypeError, ValueError, OverflowError):
                        pass
                if attempt < 4:
                    time.sleep(max(2.,min(120.,delay)))
                continue
            if response.status_code!=200:
                raise ContractError(f"USTC teacher returned HTTP {response.status_code}")
            body = response.json()
            if body.get("model")!=TEACHER_MODEL:
                raise ContractError("USTC response model does not match the requested teacher")
            try:
                completion = body["choices"][0]
                text = completion["message"]["content"].strip()
                match = re.fullmatch(r"(?:Option\s+)?([A-H])\.?",text,re.I)
                choice = match[1].upper() if match else None
                if choice not in menu:
                    raise ValueError("Teacher choice does not match the verified menu")
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                self.counts["invalid_decision_retries"]+=1
                dialogue[0]["content"] = instruction + " Valid letters: " + ", ".join(menu)
                continue
            self.counts["verified_decisions"]+=1
            usage = body.get("usage") or {}
            for key in ("prompt_tokens","completion_tokens"):
                self.counts[key]+=int(usage.get(key,0) or 0)
            decision = {"choice":choice,"alternative":next(k for k in menu if k!=choice),
                    "evidence_id":public_options.get(choice,{}).get("source_id"),"source_binding":"verified_menu",
                    "model":TEACHER_MODEL,"usage":{k:usage.get(k) for k in ("prompt_tokens","completion_tokens")}}
            atomic_json(cache,decision)
            return decision
        raise ContractError(f"USTC teacher decision failed after five attempts; counters={dict(self.counts)}")


def select_tasks(data, excluded, validation, count):
    train = pq.read_table(data).to_pylist()
    dev = pq.read_table(validation,columns=["id","question"]).to_pylist()
    blocked_questions = {x["question"] for x in dev}
    blocked_ids = set(excluded) | {x["id"] for x in dev}
    pool = [r for r in train if r["id"] not in blocked_ids and r["question"] not in blocked_questions]
    pool.sort(key=lambda r:digest([SEED,"teacher-rewards",r["id"]]))
    selected = pool[:count]
    if len({r["id"] for r in selected})!=count or len({r["question"] for r in selected})!=count:
        raise ContractError("Not enough isolated HotpotQA training questions")
    return [split_row(row) for row in selected]


def run_task(policy,teacher,task,index,evidence_intervention=False,shared_interface=False):
    public,gold = task
    seed = int(digest([SEED,public["id"]])[:8],16)
    common = {"sentence_candidates":True,"record_prompts":True,"greedy":True,
              "bridge_candidates":evidence_intervention,"shared_interface":shared_interface}
    expert,_ = rollout(policy,public,"rl_train",seed,chooser=teacher,**common)
    baseline,_ = rollout(policy,public,"rl_train",seed,**common)
    opportunities = [(index+offset)%len(STAGES) for offset in range(len(STAGES))]
    pairs, labels = [], []
    for stage in opportunities:
        a = expert["actions"][stage]
        if "teacher_decision" not in a:
            continue
        choice = a["choice"]
        noop = next(k for k,v in a["menu"].items() if v["operation"]==NOOP[a["stage"]])
        alternative = noop if choice!=noop else a["teacher_decision"]["alternative"]
        branches = []
        for letter in (choice,alternative):
            trace,_ = rollout(policy,public,"rl_train",seed,factual=expert,intervention=stage,
                              intervention_label=letter,**common)
            action = trace["actions"][stage]
            if action["boundary"]!=a["boundary"] or action["choice"]!=letter or action["sample"] is not None:
                raise ContractError("Paired intervention does not match the verified decision")
            branches.append({"choice":letter,"operation":action["operation"],"score":score(trace,gold),
                             "continuation_tokens":sum(continuation_tokens(trace,stage).values()),"trace":trace})
        x,y = branches
        delta = x["score"]["reward"]-y["score"]["reward"]
        evidence_delta = x["score"]["support_visible_recall"]-y["score"]["support_visible_recall"]
        pair = {"stage":a["stage"],"stage_index":stage,"boundary":a["boundary"],"seed":seed,
                "terminal_delta":delta,"evidence_delta":evidence_delta,"teacher":x,"alternative":y}
        if evidence_intervention:
            pair["factorial"] = factorial(policy,public,gold,x["trace"],y["trace"],stage)
        pairs.append(pair)
        if abs(delta)>1e-9:
            winner,loser = (x,y) if delta>0 else (y,x)
            labels.append({"task_id":public["id"],"phase":"state" if a["stage"].startswith("write") else "context",
                 "hotpot_stage":a["stage"],"source":"state_graph_candidate_choice","messages":a["student_messages"],
                 "completion":winner["choice"],"rejected":loser["choice"],"boundary":a["boundary"],
                 "preferred_operation":winner["operation"],"preferred_reward":winner["score"]["reward"],
                 "rejected_reward":loser["score"]["reward"],"teacher_preferred":delta>0,
                 "reward_margin":abs(delta),"training_use":"offline_sft_or_preference"})
            if evidence_intervention:
                labels[-1]["evidence_credit"] = abs(pair["factorial"]["evidence_credit"])
        if len(pairs)==PROBE_COUNT:
            break
    result = {"task_id":public["id"],"teacher":{"trace":expert,"score":score(expert,gold)},
              "student":{"trace":baseline,"score":score(baseline,gold)},"pairs":pairs,"labels":labels}
    if evidence_intervention:
        for role in ("teacher","student"):
            result[role]["coverage"] = coverage(result[role]["trace"],public,gold)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("train","validation","model","checkpoint","output","identity-cache","ustc-config"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--warmup",action="store_true")
    parser.add_argument("--reuse-run",type=Path)
    parser.add_argument("--evidence-intervention",action="store_true")
    parser.add_argument("--shared-interface",action="store_true",
                        help="Distill the same memory tool interface used by target environments and new GRPO runs.")
    parser.add_argument("--tasks",type=int,choices=range(1,TASKS+1),default=TASKS)
    parser.add_argument("--exclude-receipt",type=Path,action="append",default=[])
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    with (out/".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,26 * 2**30/total))
        prior = json.loads((args.checkpoint.parent.parent/"receipt.json").read_text())
        excluded = set(prior["train"]["ids"])
        for path in args.exclude_receipt:
            previous = json.loads(path.read_text())
            excluded.update(previous.get("train_ids",previous.get("train",{}).get("ids",[])))
        tasks = select_tasks(args.train,excluded,args.validation,args.tasks)
        ids = [item["id"] for item,_ in tasks]
        identity = model_identity(args.model,args.identity_cache)
        verify_checkpoint_base(args.checkpoint,identity)
        receipt = {"teacher":TEACHER_MODEL,"provider":"USTC","seed":SEED,"train_ids":ids,
                   "train_sha256":file_sha(args.train),"excluded_validation_sha256":file_sha(args.validation),
                   "prior_train_ids_excluded":sorted(excluded),"student_checkpoint":str(args.checkpoint),
                   "student_sha256":file_sha(args.checkpoint/"state.pt"),"model":identity,
                   "source_sha256":source_hashes(),"runner_sha256":file_sha(Path(__file__)),
                   "probes_per_task":PROBE_COUNT,"teacher_visible":"public prefix only; question after compression",
                   "rewards":"frozen student continuation; terminal (EM+F1)/2; supporting recall diagnostic",
                   "evidence_intervention":args.evidence_intervention,"shared_interface":args.shared_interface,"warmup":args.warmup}
        if args.reuse_run:
            donor_path = args.reuse_run/"receipt.json"
            donor = json.loads(donor_path.read_text())
            for field in ("teacher","train_ids","train_sha256","student_sha256","source_sha256","rewards"):
                if donor[field]!=receipt[field]:
                    raise ContractError("Reusable teacher run differs in " + field)
            if donor.get("evidence_intervention",False)!=args.evidence_intervention:
                raise ContractError("Reusable teacher run differs in evidence intervention protocol")
            if donor.get("shared_interface",False)!=args.shared_interface:
                raise ContractError("Reusable teacher run differs in manager interface")
            receipt["reused_run"] = {"path":str(args.reuse_run),"receipt_sha256":file_sha(donor_path)}
        receipt_path = out/"receipt.json"
        if receipt_path.exists() and json.loads(receipt_path.read_text())!=receipt:
            raise ContractError("Teacher trajectory receipt changed")
        atomic_json(receipt_path,receipt)
        policy = HFPolicy(args.model)
        load_checkpoint(args.checkpoint,policy,restore_random=False)
        initial = policy.fingerprint()
        teacher = USTCTeacher(args.ustc_config,out/"teacher_choices")
        units = []
        try:
            for i,task in enumerate(tasks):
                unit = out/"units"/f"{i:02d}.json"
                atomic_json(out/"status.json",{"phase":"collecting","completed":i,"expected":args.tasks})
                if unit.exists():
                    row = json.loads(unit.read_text())
                    if row["task_id"]!=ids[i]:
                        raise ContractError("Cached teacher unit does not match the declared task")
                else:
                    donor_unit = args.reuse_run/"units"/unit.name if args.reuse_run else None
                    if donor_unit and donor_unit.exists():
                        row = json.loads(donor_unit.read_text())
                        if row["task_id"]!=ids[i] or any(a.get("teacher_decision",{}).get("model") not in {None,TEACHER_MODEL}
                                                       for a in row["teacher"]["trace"]["actions"]):
                            raise ContractError("Reusable teacher trajectory does not match the task or model")
                        row["reused_unit_sha256"] = file_sha(donor_unit)
                    else:
                        row = run_task(policy,teacher,task,i,args.evidence_intervention,args.shared_interface)
                    atomic_json(unit,row)
                units.append(row)
        finally:
            teacher.close()
        if policy.fingerprint()!=initial:
            raise ContractError("Reward collection modified the frozen student")
        labels = [label for row in units for label in row["labels"]]
        pairs = [pair for row in units for pair in row["pairs"]]
        examples = out/"rewarded_preferences.jsonl"
        atomic_jsonl(examples,labels)
        atomic_jsonl(out/"complete_trajectories.jsonl",units)
        report = {"complete":True,"tasks":len(units),"pairs":len(pairs),"examples":len(labels),
                  "teacher_wins":sum(p["terminal_delta"]>1e-9 for p in pairs),
                  "teacher_losses":sum(p["terminal_delta"]< -1e-9 for p in pairs),
                  "terminal_ties":sum(abs(p["terminal_delta"])<=1e-9 for p in pairs),
                  "teacher_em":sum(row["teacher"]["score"]["em"] for row in units),
                  "student_em":sum(row["student"]["score"]["em"] for row in units),
                  "teacher_mean_support_visible_recall":sum(row["teacher"]["score"]["support_visible_recall"] for row in units)/len(units),
                  "student_mean_support_visible_recall":sum(row["student"]["score"]["support_visible_recall"] for row in units)/len(units),
                  "teacher_operations":dict(Counter(a["operation"] for row in units for a in row["teacher"]["trace"]["actions"])),
                  "api_calls_this_invocation":dict(teacher.counts),"student_frozen_during_collection":True,
                  "rewarded_examples_sha256":file_sha(examples),"complete_trajectories_sha256":file_sha(out/"complete_trajectories.jsonl")}
        warm_labels = [label for label in labels if not args.evidence_intervention or label["evidence_credit"]]
        if args.evidence_intervention:
            interventions = [pair["factorial"] for pair in pairs]
            report.update(eligible_pairs=sum(p["eligible"] for p in interventions),evidence_aligned_examples=len(warm_labels),
                          abstentions=dict(Counter(p["reason"] for p in interventions if not p["eligible"])),
                          reader_reruns=sum(not cell["reused_visible"] for p in interventions if p["eligible"]
                                            for key,cell in p["cells"].items() if key.endswith("_masked")))
        if args.warmup and warm_labels:
            warm = out/"student_warmup"
            warm_examples = out/"warmup_examples.jsonl"
            atomic_jsonl(warm_examples,warm_labels)
            warm_receipt = {**receipt,"stage":"rewarded_teacher_sft","examples_sha256":file_sha(warm_examples)}
            warm_receipt_path = warm/"receipt.json"
            if warm_receipt_path.exists() and json.loads(warm_receipt_path.read_text())!=warm_receipt:
                raise ContractError("Warmup receipt changed")
            atomic_json(warm_receipt_path,warm_receipt)
            summary_path = warm/"summary.json"
            if summary_path.exists() and (warmed:=json.loads(summary_path.read_text())).get("complete"):
                verify_checkpoint_base(Path(warmed["checkpoint"]),identity)
            else:
                latest = warm/"checkpoints/latest.json"
                resume = latest.parent/json.loads(latest.read_text())["path"] if latest.exists() else None
                optimizers = {role:torch.optim.AdamW(policy.parameters(role),lr=1e-5,weight_decay=0.)
                              for role in ("manager","actor")}
                atomic_json(out/"status.json",{"phase":"student_warmup","completed":args.tasks})
                warmed = warmup(policy,warm_examples,optimizers,warm,digest(warm_receipt),epochs=1,batch_size=2,
                                allowed_tasks=ids,resume=resume,graph_token_budget=(2048,1),memory_choice=True)
            report["warmup_checkpoint"] = warmed["checkpoint"]
        report["warmup_examples"] = len(warm_labels) if args.warmup else 0
        atomic_json(out/"summary.json",report)
        atomic_json(out/"status.json",{"phase":"complete","completed":args.tasks})
        print(canonical(report),flush=True)


if __name__=="__main__":
    main()
