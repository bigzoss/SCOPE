"""Keep privileged catalog annotations in split auditing, outside model prompts."""
from __future__ import annotations
import random
from pathlib import Path
from .contracts import ContractError
from .io import digest, file_sha, read_jsonl


class SplitRegistry:
    def __init__(self,root,seed=233):
        self.root=Path(root)
        paths={"train":"grpo/train.jsonl","dev":"grpo/dev.jsonl","test":"test/unseen.jsonl",
               "sft_train":"sft/train_tasks.jsonl","sft_dev":"sft/validation_tasks.jsonl"}
        self.rows={key:read_jsonl(self.root/value) for key,value in paths.items()}
        self.hashes={key:file_sha(self.root/value) for key,value in paths.items()}
        for key,rows in self.rows.items():
            ids=[str(r["task_id"]) for r in rows]
            if len(ids)!=len(set(ids)):raise ContractError("Duplicate task IDs in "+key)
        for left,right in (("train","dev"),("train","test"),("dev","test")):
            for field in ("task_id","isolation_group"):
                if {str(r[field]) for r in self.rows[left]} & {str(r[field]) for r in self.rows[right]}:
                    raise ContractError("Split overlap: "+left+"/"+right+"/"+field)
        sft={str(r["task_id"]) for key in ("sft_train","sft_dev") for r in self.rows[key]}
        sft_groups={str(r["isolation_group"]) for key in ("sft_train","sft_dev") for r in self.rows[key] if r.get("isolation_group")}
        for key in ("train","dev","test"):
            for r in self.rows[key]:
                if str(r["task_id"]) in sft or str(r["isolation_group"]) in sft_groups or r.get("sft_gradient_seen") or r.get("sft_validation_seen"):
                    raise ContractError("Recorded SFT exposure in "+key)
        groups={}
        for row in self.rows["train"]:groups.setdefault(row["isolation_group"],[]).append(row)
        ordered=sorted(groups,key=lambda g:digest([seed,g]))
        n=max(1,round(len(ordered)*.1))
        held=set(ordered[:n])
        self.rows["internal_dev"]=[r for r in self.rows["train"] if r["isolation_group"] in held]
        self.rows["rl_train"]=[r for r in self.rows["train"] if r["isolation_group"] not in held]

    def ids(self,split):return [str(row["task_id"]) for row in self.rows[split]]

    def training_sequence(self,length):
        domains={}
        for row in self.rows["rl_train"]:domains.setdefault(row["domain"],[]).append(str(row["task_id"]))
        eligible=[ids for ids in domains.values() if len(ids)>=length]
        if not eligible:raise ContractError("No domain has enough distinct training tasks")
        return random.sample(random.choice(eligible),length)

    def evaluation_sequences(self,split,length=2,seed=233):
        groups={}
        for row in self.rows[split]:groups.setdefault(row["domain"],[]).append(str(row["task_id"]))
        result=[]
        for domain,ids in sorted(groups.items()):
            ids.sort(key=lambda t:digest([seed,t]))
            for i in range(0,len(ids),length):result.append(ids[i:i+length])
        return result

