"""Validate offline teacher action proposals against a fixed public decision view."""
import json

from .io import canonical
from .teacher_execution import TeacherActionRejected


def teacher_messages(packet):
    subgoal = (", and expected_subgoal (a short, publicly checkable effect)" if packet.get("protocol") == 2 else "")
    return [{"role": "system", "content":
             "Choose one next action from the supplied labels using only the public evidence. "
             "Return JSON with exactly action_label (a label or null), evidence (a list of "
             "{id, quote} copied verbatim from the public records), and missing_preconditions "
             "(a list of short strings)" + subgoal + ". Select a safe check or exploration action when useful. "
             "If an action can end the task, choose it only when the public record supports "
             "the required object and prerequisites. Do not invent state or attempt to solve "
             "from hidden labels."},
            {"role": "user", "content": canonical(packet)}]


def validate_proposal(response, packet):
    try:
        proposal = json.loads(response)
    except (TypeError, ValueError) as exc:
        raise TeacherActionRejected("Teacher proposal is not JSON") from exc
    expected = {"action_label", "evidence", "missing_preconditions"}
    if packet.get("protocol") == 2:
        expected.add("expected_subgoal")
    if not isinstance(proposal, dict) or set(proposal) != expected:
        raise TeacherActionRejected("Invalid public action repair schema")
    label = proposal["action_label"]
    missing = proposal["missing_preconditions"]
    if not isinstance(missing, list) or len(missing) > 6 or any(
            not isinstance(item, str) or not item.strip() or len(item) > 200 for item in missing):
        raise TeacherActionRejected("Invalid missing preconditions")
    menu = {record["label"]: record["command"] for record in packet["menu"]}
    if label is not None and (not isinstance(label, str) or label not in menu):
        raise TeacherActionRejected("Teacher action is outside the exact public menu")
    if packet.get("protocol") == 2 and (not isinstance(proposal["expected_subgoal"], str) or
            not proposal["expected_subgoal"].strip() or len(proposal["expected_subgoal"]) > 240):
        raise TeacherActionRejected("Expected subgoal must be a short public effect")
    references = {row["id"]: row["text"] for row in packet["records"]}
    evidence = proposal["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 4 or (label is not None and not evidence):
        raise TeacherActionRejected("An executable teacher action must cite public evidence")
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {"id", "quote"}
                or not isinstance(item["id"], str) or item["id"] not in references
                or not isinstance(item["quote"], str) or not item["quote"].strip()
                or len(item["quote"]) > 240 or item["quote"] not in references[item["id"]]):
            raise TeacherActionRejected("Teacher citation does not match a visible public span")
    if label is not None and missing and menu[label].startswith("focus on "):
        raise TeacherActionRejected("Teacher cannot finalize while reporting missing prerequisites")
    return {**proposal, "command": menu.get(label)}


def verified_gain(control, branch):
    """Success dominates progress; a rejected or unsuccessfully terminal action cannot earn credit."""
    if branch["action_status"] != "accepted" or branch["failed_on_intervention"]:
        return False
    left, right = control["metrics"], branch["metrics"]
    if right["success"] != left["success"]:
        return right["success"] > left["success"]
    return right["r_loose"] > left["r_loose"]
