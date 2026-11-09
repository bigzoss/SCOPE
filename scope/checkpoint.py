"""Atomic, completed-update checkpoints, never weights-only resumptions."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import uuid
import json
from enum import Enum
from .contracts import ContractError
from .io import atomic_json, file_sha,digest,fsync_directory
from .rollout import capture_rng,restore_rng

ROLES={"manager","actor","manager_ref","actor_ref"}
_CONFIG_KEYS=("r","lora_alpha","lora_dropout","target_modules","bias","modules_to_save",
              "rank_pattern","alpha_pattern","use_rslora","use_dora","fan_in_fan_out","task_type",
              "layers_to_transform","layers_pattern")


def _normal(value):
    if isinstance(value,Enum):return value.value
    if isinstance(value,dict):return {k:_normal(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):return sorted(_normal(v) for v in value)
    if isinstance(value,(list,tuple)):return [_normal(v) for v in value]
    return value


def adapter_configs(policy):
    return {role:{key:_normal(policy.model.peft_config[role].to_dict().get(key)) for key in _CONFIG_KEYS} for role in sorted(ROLES)}


def _verify_legacy_config(path,meta,policy):
    origin_path=path.parent.parent/"receipt.json"
    if not origin_path.exists():raise ContractError("Legacy checkpoint lacks a verified originating receipt")
    origin=json.loads(origin_path.read_text())
    # This is the exact audited v0.1 HF constructor, with q/k/v/o LoRA and zero dropout.
    known="fd7f8cee57229407a632f664172a9072b0fa90ed34387f9b9b34dfa27d7ef419"
    if digest(origin)!=meta["receipt_hash"] or origin["source_sha256"].get("scope/policy.py")!=known:
        raise ContractError("Legacy adapter configuration cannot be established from its source receipt")
    model=origin["config"]["model"]
    expected={"r":model.get("rank",16),"lora_alpha":model.get("alpha",32),"lora_dropout":0.,
              "target_modules":["k_proj","o_proj","q_proj","v_proj"],"bias":"none",
              "use_rslora":False,"use_dora":False,"fan_in_fan_out":False}
    for cfg in adapter_configs(policy).values():
        if any(cfg.get(key)!=value for key,value in expected.items()):raise ContractError("Legacy LoRA configuration mismatch")


def save_checkpoint(root,step,policy,optimizers,receipt_hash,extra=None):
    import torch
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/f"step_{step:05d}"
    if target.exists():raise ContractError("Refusing to overwrite an existing checkpoint")
    stage=root/(".saving_"+uuid.uuid4().hex);stage.mkdir()
    try:
        state={"format_version":2,"step":step,"versions":dict(policy.versions),"receipt_hash":receipt_hash,
               "adapter_configs":adapter_configs(policy),
               "adapters":{role:policy.adapter_state(role) for role in ("manager","actor","manager_ref","actor_ref")},
               "optimizers":{role:opt.state_dict() for role,opt in optimizers.items()},"rng":capture_rng(),
               "extra":extra or {},"resume_boundary":"completed_update"}
        with (stage/"state.pt").open("wb") as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        atomic_json(stage/"metadata.json",{"step":step,"receipt_hash":receipt_hash,"state_sha256":file_sha(stage/"state.pt"),
                    "versions":policy.versions,"resume_boundary":"completed_update"})
        os.replace(stage,target)
        fsync_directory(root)
        atomic_json(root/"latest.json",{"path":target.name,"step":step})
    finally:
        if stage.exists():shutil.rmtree(stage)
    return target


def load_checkpoint(path,policy,optimizers=None,receipt_hash=None,restore_random=True,roles=None):
    import torch
    path=Path(path)
    meta=json.loads((path/"metadata.json").read_text())
    if file_sha(path/"state.pt")!=meta["state_sha256"]:raise ContractError("Checkpoint digest mismatch")
    if receipt_hash is not None and meta["receipt_hash"]!=receipt_hash:raise ContractError("Checkpoint receipt mismatch")
    # Only load trusted, locally-created checkpoints; optimizer/RNG contain Python objects.
    data=torch.load(path/"state.pt",map_location="cpu",weights_only=False)
    if data.get("format_version",1) not in {1,2}:raise ContractError("Unsupported checkpoint format")
    if data["resume_boundary"]!="completed_update":raise ContractError("Unsupported resume boundary")
    if data["step"]!=meta["step"] or data["receipt_hash"]!=meta["receipt_hash"] or data["versions"]!=meta["versions"]:
        raise ContractError("Checkpoint metadata/payload mismatch")
    if set(data["adapters"])!=ROLES or set(data["versions"])!={"manager","actor"}:
        raise ContractError("Incomplete checkpoint roles")
    if "adapter_configs" in data:
        if data["adapter_configs"]!=adapter_configs(policy):raise ContractError("LoRA configuration mismatch")
    else:_verify_legacy_config(path,meta,policy)
    names=dict(policy.model.named_parameters())
    # Validate every role before mutating any weight. Same tensor names/shapes do not
    # establish the same LoRA policy when alpha or rank-dependent scaling changed.
    for role,state in data["adapters"].items():
        if set(state)!=set(policy.adapter_state(role)):raise ContractError("Adapter parameter identity mismatch")
        for key,tensor in state.items():
            if tensor.shape!=names[key].shape or tensor.dtype!=names[key].dtype or not torch.isfinite(tensor).all():
                raise ContractError("Invalid adapter tensor")
    if optimizers is not None and not set(optimizers)<=set(data["optimizers"]):raise ContractError("Missing optimizer state")
    chosen=set(roles) if roles is not None else {"manager","actor"}
    if not chosen or not chosen<={"manager","actor"}:raise ContractError("Invalid checkpoint role selection")
    selected=chosen|{r+"_ref" for r in chosen}
    with torch.no_grad():
        for role,state in data["adapters"].items():
            if role not in selected:continue
            expected=set(policy.adapter_state(role))
            if set(state)!=expected:raise ContractError("Adapter parameter identity mismatch")
            for key,tensor in state.items():names[key].copy_(tensor.to(names[key].device))
    policy.versions.update({role:data["versions"][role] for role in chosen})
    if optimizers is not None:
        for role,opt in optimizers.items():
            opt.load_state_dict(data["optimizers"][role])
            # Optimizer.load_state_dict performs parameter-aware placement. In particular
            # Adam's non-capturable step counter must remain on CPU; moving it to CUDA
            # changes the foreach numerical path and can make the resumed update differ.
    if restore_random:restore_rng(data["rng"])
    return data

