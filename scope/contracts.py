from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .io import digest


class ContractError(ValueError):
    pass


class InfrastructureError(RuntimeError):
    """Never convert an unavailable environment into a model failure reward."""


@dataclass(frozen=True)
class Scope:
    split: str
    task: str
    episode: str
    user: str = "u0000"

    @property
    def key(self):
        return digest(self)


@dataclass
class Sample:
    role: str
    phase: str
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]
    text: str
    version: int
    boundary: str
    termination: str = "eos"
    advantage: float = 0.0
    credit_scope: str = "global"


@dataclass
class Budget:
    turns: int = 12
    manager_tokens: int = 1536
    retrievals: int = 6
    total_tokens: int = 100000

    def charge(self, role, input_tokens, output_tokens):
        if role not in {"manager","actor"} or any(type(x) is not int or x<0 for x in (input_tokens,output_tokens)):
            raise ContractError("Invalid token charge")
        amount = int(input_tokens) + int(output_tokens)
        if amount > self.total_tokens:
            raise ContractError("Total token budget exhausted")
        if role == "manager":
            if output_tokens > self.manager_tokens:
                raise ContractError("Manager generation budget exhausted")
        self.total_tokens -= amount
        if role=="manager":self.manager_tokens -= output_tokens


@dataclass
class Outcome:
    task_id: str
    samples: list[Sample]
    metrics: dict
    reward: float
    events: list[dict] = field(default_factory=list)
    boundaries: list[Any] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    protocol: str = "full_instruction"
    termination_reason: str | None = None


@dataclass
class Boundary:
    scope: Scope
    atlas: dict
    environment: dict
    budget: Budget
    goal: str
    observation: str
    history: list[dict]
    phase: str
    role_versions: dict
    rng: Any
    costs: dict
    invalid: int
    working: dict = field(default_factory=dict)
    task_index: int = 0
    continuation_tasks: list[str] = field(default_factory=list)

    def fingerprint(self):
        return digest({"scope": self.scope, "atlas": self.atlas, "environment": self.environment,
                       "budget": self.budget, "goal": self.goal, "observation": self.observation,
                       "history": self.history, "working":self.working,"costs":self.costs,"invalid":self.invalid,
                       "continuation_tasks":self.continuation_tasks,"phase": self.phase, "versions": self.role_versions})
