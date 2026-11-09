"""Public client and opaque, server-side simulator snapshots."""
from __future__ import annotations
import copy
import re
import math
import requests
from .contracts import ContractError, InfrastructureError
from .interaction import ActionAdapter, Binding
from .state_graph import SHOP_PREDICATES, shop_facts
from .io import digest

ACTION = re.compile(r"^\s*Action\s*:\s*(search|click|buy)\s*\[([^\]\n]+)\]\s*$",re.I|re.M)
RAW_ACTION = re.compile(r"\s*(search|click|buy)\s*\[([^\]\n]+)\]\s*",re.I)


def parse_action(text):
    if "</think>" in text:
        text = text.split("</think>",1)[1]
    found = ACTION.findall(text)
    if not found:
        raw=RAW_ACTION.fullmatch(text)
        if raw:found=[raw.groups()]
    if len(found)!=1:return None
    kind, value = found[0][0].lower(), found[0][1].strip()
    if not value:return None
    if kind == "buy" or value.lower() in {"buy now","购买","立即购买"}:
        return "click[Buy Now]"
    return f"{kind}[{value}]"


class ShopActions(ActionAdapter):
    parse = staticmethod(parse_action)
    state_predicates = SHOP_PREDICATES
    public_facts = staticmethod(shop_facts)

    def recovery_candidates(self, state, goal):
        labels = state.get("clickables", [])
        for label in sorted(labels, key=lambda value:(value not in goal, value)):
            if label.casefold() not in {"buy now", "购买", "立即购买"}:
                yield {"command": f"click[{label}]", "source_labels":[label],
                       "reason":"inspect_public_option_or_other_candidate"}

    def observe(self, observation, previous, command, clickables):
        return {**super().observe(observation, previous, command, clickables),
                "clickables": sorted(set(clickables))}

    def bind(self, command, state):
        clickables = state.get("clickables", [])
        if not command.startswith("click[") or not clickables:
            return Binding()
        if command[6:-1].casefold() not in {item.casefold() for item in clickables}:
            return Binding("unobserved", "the button is absent from the current public page")
        return Binding("bound")

    def confirmation_target(self, command, entity):
        match = re.fullmatch(r"click\[(黑色|白色|红色|蓝色|灰色|绿色|银色|金色|透明)\]", command)
        if match and entity.startswith("product:"):
            return entity, "selected_color", match[1]
        return None

    def evidence_entity(self, command, previous):
        value = command.split("[", 1)[1][:-1]
        if command.startswith("search[") or value.lower() == "back to search":
            return "task"
        if re.fullmatch(r"(?:\d{8,}|[Bb][A-Za-z0-9]{9})", value):
            return "product:" + value.upper()
        return previous


class ShopEnvironment:
    action_adapter = ShopActions()
    def __init__(self, base_url, allowed_tasks, manifest_sha=None, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.allowed = set(map(str,allowed_tasks))
        self.manifest_sha = manifest_sha
        self.session = requests.Session()
        self.timeout = timeout
        self.lease = None
        self.task = None
        self.observation = ""
        self.metrics = {"success":0.,"r_strict":0.,"r_loose":0.,"purchased":0.}

    def _call(self, route, payload=None):
        try:
            result = self.session.get(self.base_url+route,timeout=self.timeout) if payload is None else self.session.post(self.base_url+route,json=payload,timeout=self.timeout)
            result.raise_for_status()
            data = result.json()
        except (requests.RequestException,ValueError) as exc:
            raise InfrastructureError(f"ShopSimulator {route}: {exc}") from exc
        if not isinstance(data,dict):raise InfrastructureError("ShopSimulator returned a non-object response")
        if data.get("error"):
            raise InfrastructureError(data["error"])
        return data

    def health(self):
        data = self._call("/healthz")
        if not data.get("ready") or not data.get("manifest_unchanged"):
            raise InfrastructureError("Service is not ready or its manifest changed")
        if self.manifest_sha and data.get("manifest_sha256") != self.manifest_sha:
            raise ContractError("Environment manifest hash mismatch")
        return data

    def reset(self, task):
        self.close()
        if str(task) not in self.allowed:
            raise ContractError("Task outside client split")
        data = self._call("/reset", {"task_id":str(task)})
        self.lease,self.task,self.observation = data["lease"],str(task),data["observation"]
        self.metrics = {"success":0.,"r_strict":0.,"r_loose":0.,"purchased":0.}
        return self.observation

    def step(self, response):
        command = parse_action(response)
        if command is None:
            return self.observation, False, {"is_error":True,"command":None,"clickables":[]}
        return self.step_command(command)

    def step_command(self, command):
        data = self._call("/step", {"lease":self.lease,"command":command})
        self.observation = data["observation"]
        if data["done"]:
            metrics=data.get("metrics")
            if not isinstance(metrics,dict) or any(k not in metrics or not isinstance(metrics[k],(int,float)) or not math.isfinite(metrics[k]) or not 0<=metrics[k]<=1 for k in ("success","r_strict","r_loose","purchased")):
                raise InfrastructureError("Terminal task score is missing or invalid")
            if metrics["success"] not in (0,1):raise InfrastructureError("Terminal success is not binary")
            self.metrics = metrics  # Never pass this object to the manager or atlas.
        return self.observation,data["done"],{"is_error":data["is_error"],"command":command,
                                             "clickables":data.get("clickables",[])}

    def snapshot(self):
        data = self._call("/snapshot", {"lease":self.lease})
        return {"handle":data["handle"],"state_hash":data["state_hash"],"task":self.task,
                "observation_hash":digest(self.observation),"restore_level":"native_private_state"}

    def restore(self, snapshot):
        self.close()
        if snapshot["task"] not in self.allowed:
            raise ContractError("Snapshot task outside split")
        data = self._call("/restore", {"handle":snapshot["handle"]})
        if data["state_hash"] != snapshot["state_hash"] or digest(data["observation"]) != snapshot["observation_hash"]:
            raise InfrastructureError("Native simulator snapshot verification failed")
        self.lease,self.task,self.observation = data["lease"],snapshot["task"],data["observation"]
        self.metrics = {"success":0.,"r_strict":0.,"r_loose":0.,"purchased":0.}
        return self.observation

    def drop_snapshot(self, snapshot):
        self._call("/drop_snapshot", {"handle":snapshot["handle"]})

    def close(self):
        if self.lease is not None:
            lease,self.lease = self.lease,None
            self._call("/release", {"lease":lease})


class DeterministicTestEnvironment:
    """Test fixture only: never used as an automatic fallback for real experiments."""
    action_adapter = ShopActions()
    def __init__(self, task="1"):
        self.task = task
        self.reset(task)

    def reset(self, task):
        self.task=str(task);self.position=0
        self.observation="用户需要黑色商品，预算不超过400元。"
        self.metrics={"success":0.,"r_strict":0.,"r_loose":0.,"purchased":0.}
        return self.observation

    def step(self, response):
        return self.step_command(parse_action(response))

    def step_command(self, command):
        self.position+=1
        done=command=="click[Buy Now]"
        if done:self.metrics={"success":1.,"r_strict":1.,"r_loose":1.,"purchased":1.}
        self.observation="商品 123 价格：358元，可点击的按钮: [\"黑色\", \"Buy Now\"]"
        return self.observation,done,{"command":command,"is_error":command is None,"clickables":["黑色","Buy Now"]}

    def snapshot(self):
        state=copy.deepcopy(self.__dict__)
        return {"handle":state,"state_hash":digest(state),"restore_level":"test_only"}

    def restore(self,snapshot):
        self.__dict__.update(copy.deepcopy(snapshot["handle"]))
        return self.observation

    def drop_snapshot(self,snapshot):pass
    def close(self):pass

