"""Teacher action demonstrations over the student's exact public decision view.

Grammar and citations are checked here. Task utility is established only by the
environment rollout; an accepted proposal alone is never a positive training label.
"""
import json
import re

from .contracts import ContractError
from .io import canonical


class TeacherActionRejected(ContractError):
    """The teacher response cannot be executed under the public action contract."""


class ExecutionTeacher:
    def __init__(self, generate, intervention="recovery"):
        self.generate = generate
        if intervention not in {"all", "recovery"}:
            raise ContractError("Unknown teacher intervention mode")
        self.intervention = intervention

    def should_intervene(self, working):
        return self.intervention == "all" or bool(working.get("warning") or
            working.get("recovery_plan", {}).get("repeated_commands") or
            (working.get("confirmation") or {}).get("status") == "conflict")

    def propose(self, messages, state, adapter):
        if adapter is None:
            raise ContractError("Execution teacher requires a public action adapter")
        request = [{"role":"system", "content":
            "Teach a small executor using the supplied exact public decision view. Return JSON with "
            "command (one executable command), evidence_ids (visible public IDs), and subgoal (brief text). "
            "Use the action grammar in the student's system prompt. A natural-language task goal is not a command. "
            "Check missing prerequisites and rejected actions. Seeing an object does not prove holding it. "
            "Do not invent confirmations. Select a useful action, including public exploration when necessary."},
            {"role":"user", "content":canonical({"student_messages":messages})}]
        response = self.generate(request)
        try:
            answer = json.loads(response)
        except (ValueError, TypeError) as exc:
            raise TeacherActionRejected("Invalid execution teacher JSON") from exc
        if not isinstance(answer, dict) or set(answer) != {"command", "evidence_ids", "subgoal"}:
            raise TeacherActionRejected("Invalid execution teacher schema")
        command, refs, subgoal = answer["command"], answer["evidence_ids"], answer["subgoal"]
        if not isinstance(command, str) or "\n" in command or not isinstance(subgoal, str) or len(subgoal) > 300:
            raise TeacherActionRejected("Invalid teacher command/subgoal")
        visible = set(re.findall(r"obs_\d+_[a-f0-9]+", canonical(messages)))
        if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in visible for x in refs):
            raise TeacherActionRejected("Teacher action must cite visible evidence")
        parsed = adapter.parse("Action: " + command)
        if parsed is None:
            raise TeacherActionRejected("Teacher action does not parse")
        command = adapter.normalize(parsed)
        if adapter.bind(command, state).status not in {"bound", "not_required", "unchecked"}:
            raise TeacherActionRejected("Teacher action does not bind to public state")
        return {"text":"Evidence: " + ", ".join(refs) + "\nAction: " + command,
                "subgoal":subgoal, "evidence_ids":refs, "grammar_verified":True,
                "utility_verified":False}
