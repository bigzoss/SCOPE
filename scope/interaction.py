"""Benchmark-independent action dispatch and public-state loop diagnostics."""
from dataclasses import dataclass
import re

from .contracts import ContractError
from .io import digest
from .public_data import assert_public_text


ACTION_MODES = ("legacy", "normalize", "grounded")
LOOP_POLICIES = ("legacy", "state")
TRACE_VERSION = "public-interaction-v1"
WINDOW_SIZE = 8


def action_text(response):
    """Record the proposed command, excluding model reasoning."""
    text = response.split("</think>", 1)[-1].strip()
    lines = re.findall(r"^\s*Action\s*:[ \t]*(.*)$", text, re.I | re.M)
    if lines:
        return lines[-1].strip() or None
    return text if text and "\n" not in text and not text.lower().startswith(("thought:", "<think>")) else None


@dataclass(frozen=True)
class Binding:
    status: str = "unchecked"
    reason: str | None = None
    candidates: tuple[str, ...] = ()


class ActionAdapter:
    """Domain hooks over public observations. Projections must be JSON-serializable."""

    def parse(self, response):
        raise NotImplementedError

    def normalize(self, command):
        return command

    def bind(self, command, state):
        return Binding()

    def observe(self, observation, previous, command, clickables):
        return {"page": digest(" ".join(observation.split()))}

    def progress_key(self, state):
        """Override to omit volatile UI fields from progress and loop detection."""
        return state

    def evidence_entity(self, command, previous):
        return previous


def execute_action(env, adapter, response, observation, state, mode):
    raw = action_text(response)
    binding = Binding()
    if adapter is None:
        next_observation, done, public = env.step(response)
        command = public.get("command")
        sent = None  # An older adapter does not expose dispatch semantics.
        status = "accepted" if not public["is_error"] else (
            "format_error" if command is None else "environment_invalid")
    else:
        command = adapter.parse(response)
        if command is not None and mode != "legacy":
            command = adapter.normalize(command)
        if command and mode == "grounded":
            binding = adapter.bind(command, state)
        status = "accepted" if command else "format_error"
        if binding.status in {"missing", "ambiguous", "unobserved"}:
            status = "binding_failed"
        elif binding.status == "invalid_syntax":
            status = "format_error"
        sent = status == "accepted"
        if sent:
            next_observation, done, public = env.step_command(command)
            if public["is_error"]:
                status = "environment_invalid"
        else:
            next_observation, done, public = observation, False, {"clickables": []}
    assert_public_text(next_observation)
    public = {"command": command, "is_error": status != "accepted", "clickables": public.get("clickables", [])}
    trace = {"raw_command": raw, "normalized_command": command, "sent": sent, "action_status": status,
             "binding_status": binding.status, "binding_reason": binding.reason,
             "binding_candidates": list(binding.candidates)}
    return next_observation, done, public, trace


def record_progress(window, before, after, command, evidence_id, accepted, done):
    """Detect a repeated transition after returning to a public state within a bounded window."""
    if not accepted or command is None or done:
        window.clear()
        return ("terminal" if accepted and done else None), None
    edge = [before, command, after]
    repeat = next((i for i in range(len(window) - 1, -1, -1) if window[i]["edge"] == edge), None)
    cycle = None
    if repeat is not None:
        earlier = window[repeat:]
        if earlier[-1]["edge"][2] != before:
            raise ContractError("Discontinuous public interaction history")
        cycle = {"actions": [item["edge"][1] for item in earlier] + [command],
                 "evidence_ids": list(dict.fromkeys(item["evidence_id"] for item in earlier))}
    window.append({"edge": edge, "evidence_id": evidence_id})
    del window[:-WINDOW_SIZE]
    return ("legal_no_progress" if before == after or cycle else "state_changed"), cycle


def interaction_guidance(trace):
    if trace["sent"] is False:
        reason = trace["binding_reason"] or "provide one valid action in the requested format"
        matches = trace["binding_candidates"]
        suffix = f" Public matches: {', '.join(matches)}." if matches else ""
        return f"Action was not sent: {reason}.{suffix} Choose a new action using public evidence."
    if trace["action_status"] == "environment_invalid":
        return "The environment rejected the action. Check the current public state before choosing again."
    if trace["loop"]:
        actions = " → ".join(trace["loop"]["actions"])
        return f"Public state loop: {actions}. Recheck relevant evidence or choose a different action."
    return ""
