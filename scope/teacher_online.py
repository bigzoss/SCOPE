"""Bounded online recovery for frozen benchmark evaluation."""
from collections import Counter
from email.utils import parsedate_to_datetime
import time

from .teacher_api import APITeacher
from .teacher_execution import ExecutionTeacher, TeacherActionRejected


TEACHER = {"provider": "USTC", "model": "qwen3.8-flash-next",
           "trigger": "rejection_loop_or_public_conflict", "max_proposals_per_task": 2,
           "request_spacing_seconds": 6.5, "invalid_proposal": "student_fallback",
           "infrastructure_failure": "abort_and_resume"}


class OnlineRecovery:
    def __init__(self, credentials, cache, max_requests):
        self.identity = dict(TEACHER)
        self.api = APITeacher(credentials, cache, max_requests=max_requests, model=self.identity["model"],
                              allow_unusable=True)
        self.executor = ExecutionTeacher(self.api, "recovery")
        self.next_request = 0.0
        post = self.api.http.post

        def paced_post(*args, **kwargs):
            time.sleep(max(0.0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.identity["request_spacing_seconds"]
            response = post(*args, **kwargs)
            if response.status_code == 429:
                value = response.headers.get("Retry-After")
                try:
                    delay = float(value)
                except (TypeError, ValueError):
                    try:
                        delay = parsedate_to_datetime(value).timestamp() - time.time()
                    except (TypeError, ValueError, OverflowError):
                        delay = 60.0
                self.next_request = max(self.next_request, time.monotonic() + min(180.0, delay))
            return response

        self.api.http.post = paced_post
        self.begin_task()

    def begin_task(self):
        self.counts = Counter()
        self.start = self.api.counts.copy()
        self.blocked = set()

    def should_intervene(self, working):
        self.blocked = set(working.get("recovery_plan", {}).get("blocked_commands", []))
        return (self.counts["proposals"] < self.identity["max_proposals_per_task"]
                and self.executor.should_intervene(working))

    def record_rejection(self, reason):
        self.counts["rejected_proposals"] += 1
        self.counts["rejected_" + reason] += 1

    def propose(self, messages, state, adapter):
        self.counts["proposals"] += 1
        try:
            label = self.executor.propose(messages, state, adapter)
        except TeacherActionRejected:
            self.record_rejection("syntax_or_evidence")
            return None
        if adapter.parse(label["text"]) in self.blocked:
            self.record_rejection("blocked_repeat")
            return None
        return label

    def task_usage(self):
        return {**self.counts, **(self.api.counts - self.start)}

    def close(self):
        self.api.close()
