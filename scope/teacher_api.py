"""Bounded USTC teacher transport for offline public-prefix supervision.

The config is read locally. Credentials and HTTP response bodies never enter logs.
Cached responses are proposals only; callers must run the normal tool validator.
"""
from collections import Counter
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests

from .contracts import ContractError, InfrastructureError
from .io import atomic_json, canonical, digest


class APITeacher:
    def __init__(self, config_path, cache, max_requests=24):
        config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
        url = urlsplit(config["base_url"])
        if (url.scheme, url.hostname, url.path.rstrip("/")) != ("https", "api.llm.ustc.edu.cn", "/v1") or (
                url.username or url.password or url.query or url.fragment or url.port not in (None, 443)):
            raise ContractError("Teacher endpoint must be the USTC HTTPS service")
        if type(max_requests) is not int or max_requests < 1:
            raise ContractError("Teacher request budget must be positive")
        self.model = config.get("model", "qwen3.8-flash-next")
        self.identity = {"provider":"USTC", "model":self.model, "protocol":"public-prefix-json-v1"}
        self.url = config["base_url"].rstrip("/") + "/chat/completions"
        self.http = requests.Session()
        self.http.headers["Authorization"] = "Bearer " + config["api_key"]
        self.cache, self.limit, self.counts = Path(cache), max_requests, Counter()

    def close(self):
        self.http.close()

    def __call__(self, messages):
        body = {"model":self.model, "messages":messages, "temperature":0, "max_tokens":768,
                "chat_template_kwargs":{"enable_thinking":False}}
        key = digest(body)
        path = self.cache / (key + ".json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("request_hash") != key or row.get("response_hash") != digest(row.get("text")):
                raise ContractError("Teacher cache identity mismatch")
            self.counts["cache_hits"] += 1
            return row["text"]
        for attempt in range(3):
            if self.counts["requests"] >= self.limit:
                raise InfrastructureError("Teacher API request budget exhausted")
            self.counts["requests"] += 1
            try:
                response = self.http.post(self.url, json=body, timeout=(15, 50), allow_redirects=False)
            except requests.RequestException:
                if attempt == 2:
                    raise InfrastructureError("Teacher API transport failed") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code in {429, 500, 502, 503, 504}:
                self.counts["retries"] += 1
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
            if response.status_code != 200:
                raise InfrastructureError(f"Teacher API HTTP {response.status_code}")
            try:
                result = response.json()
                if result["model"] != self.model or result["choices"][0]["finish_reason"] != "stop":
                    raise ValueError("Model mismatch or truncated completion")
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not isinstance(json.loads(text), dict):
                    raise ValueError("Expected a JSON object")
            except (ValueError, KeyError, TypeError, IndexError):
                raise InfrastructureError("Teacher API returned an invalid structured completion") from None
            usage = result.get("usage") or {}
            for field in ("prompt_tokens", "completion_tokens"):
                self.counts[field] += int(usage.get(field, 0) or 0)
            atomic_json(path, {"request_hash":key, "response_hash":digest(text), "text":text,
                               "identity":self.identity, "usage":usage})
            return text
        raise InfrastructureError("Teacher API exhausted retries")

    def select_witness(self, target, spans):
        result = json.loads(self([{"role":"system", "content":
            "Choose a public span supporting the target entity, predicate and value. Seeing an object does not "
            "prove holding it. Return {\"index\": integer} or {\"index\": null} if no span supports it."},
            {"role":"user", "content":canonical({"target":target, "spans":spans})}]))
        index = result.get("index")
        return spans[index] if type(index) is int and 0 <= index < len(spans) else None
