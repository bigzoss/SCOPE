import json

import pytest

from scope.public_action_repair import validate_proposal, verified_gain
from scope.teacher_execution import TeacherActionRejected


def test_teacher_label_and_quote_are_checked_against_public_packet():
    packet = {"menu": [{"label": "A", "command": "focus on thermometer"}],
              "records": [{"id": "obs1", "text": "You see a thermometer on the shelf."}]}
    proposal = {"action_label": "A", "evidence": [{"id": "obs1", "quote": "a thermometer"}],
                "missing_preconditions": []}
    assert validate_proposal(json.dumps(proposal), packet)["command"] == "focus on thermometer"
    for corrupt in (dict(proposal, action_label="B"),
                    dict(proposal, evidence=[{"id": "obs1", "quote": "You are holding a thermometer"}]),
                    dict(proposal, missing_preconditions=["need to measure the substance"])):
        with pytest.raises(TeacherActionRejected):
            validate_proposal(json.dumps(corrupt), packet)


def test_terminal_failure_and_strict_regression_cannot_become_demonstrations():
    baseline = {"metrics": {"success": 0, "r_loose": 0.1}}
    branch = {"action_status": "accepted", "failed_on_intervention": True,
              "metrics": {"success": 0, "r_loose": 0.25}}
    assert not verified_gain(baseline, branch)
    branch["failed_on_intervention"] = False
    assert verified_gain(baseline, branch)
    assert not verified_gain({"metrics": {"success": 1, "r_loose": 0.1}}, branch)
