"""Bounded 429 backoff without changing prompts, cache keys, or loss ownership."""
from email.utils import parsedate_to_datetime
import math
import time

from .contracts import InfrastructureError, TeacherBudgetExhausted
from .teacher_api import APITeacher


def retry_after_seconds(value, now=None):
    if value is None:
        return 0.
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            return 0.
    return max(0., seconds) if math.isfinite(seconds) else 0.


class RateLimitedSession:
    """The base transport accounts for the first attempt; count every retry here."""
    def __init__(self, inner, teacher, *, attempts=12, base_delay=30., cap_delay=120.,
                 max_wait=1200., min_interval=5., on_wait=None):
        if attempts < 1 or min(base_delay, cap_delay, max_wait, min_interval) < 0:
            raise ValueError('Invalid retry policy')
        self.inner, self.teacher = inner, teacher
        self.attempts, self.base_delay, self.cap_delay = attempts, base_delay, cap_delay
        self.max_wait, self.min_interval, self.on_wait = max_wait, min_interval, on_wait
        self.last_start = None

    def close(self):
        self.inner.close()

    def post(self, *args, **kwargs):
        started = time.monotonic()
        for index in range(self.attempts):
            if self.last_start is not None:
                delay = self.min_interval - (time.monotonic() - self.last_start)
                if delay > 0:
                    time.sleep(delay)
                    self.teacher.counts['pacing_wait_seconds'] += delay
            self.last_start = time.monotonic()
            response = self.inner.post(*args, **kwargs)
            if response.status_code != 429:
                return response
            self.teacher.counts['rate_limited_attempts'] += 1
            delay = max(min(self.cap_delay, self.base_delay * 2**index),
                        retry_after_seconds(response.headers.get('Retry-After')))
            response.close()
            if index+1 == self.attempts or time.monotonic()-started+delay > self.max_wait:
                raise InfrastructureError('Teacher HTTP 429 persisted beyond bounded cooldown; checkpoint remains recoverable')
            if self.teacher.counts['requests'] >= self.teacher.limit:
                raise TeacherBudgetExhausted('Teacher API request budget exhausted during rate-limit retry')
            self.teacher.counts['retries'] += 1
            if self.on_wait is not None:
                self.on_wait({'event':'teacher_rate_limit_wait','attempt':index+1,'wait_seconds':delay,
                              'http_requests_used':self.teacher.counts['requests']})
            time.sleep(delay)
            self.teacher.counts['rate_limit_wait_seconds'] += delay
            if self.teacher.counts['requests'] >= self.teacher.limit:
                raise TeacherBudgetExhausted('Teacher API request budget exhausted during rate-limit retry')
            self.teacher.counts['requests'] += 1
        raise InfrastructureError('Unreachable teacher retry state')


class RateLimitedTeacher(APITeacher):
    def __init__(self, *args, on_wait=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.http = RateLimitedSession(self.http, self, on_wait=on_wait)
