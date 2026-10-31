"""Optional Jev decisions with deterministic local rules."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DecisionRequest:
    task: str
    options: tuple[str, ...]
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResponse:
    selected: str | None
    approved: bool
    rationale: str
    provider: str
    model: str
    latency_ms: float
    used_fallback: bool = False
    error: str | None = None


class LocalRuleDecisionProvider:
    provider_name = "local-rules"
    model_name = "deterministic"

    def health_check(self) -> dict[str, Any]:
        return {"available": True, "provider": self.provider_name, "model": self.model_name, "offline": True}

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        started = time.perf_counter()
        selected = request.options[0] if request.options else None
        return DecisionResponse(
            selected, True, "The first explicitly supplied option is the deterministic local choice.",
            self.provider_name, self.model_name, (time.perf_counter() - started) * 1000, used_fallback=True
        )


class JevDecisionProvider:
    """Structured decision adapter for Jev-compatible or OpenRouter endpoints."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout_seconds: float = 15.0):
        self.provider_name = os.getenv("JEV_PROVIDER", "jev").strip().lower() or "jev"
        default_endpoint = (
            "https://openrouter.ai/api/v1/chat/completions"
            if self.provider_name == "openrouter" else ""
        )
        self.endpoint = (endpoint or os.getenv("JEV_ENDPOINT", "") or default_endpoint).strip()
        self.api_key = api_key or os.getenv("JEV_API_KEY", "")
        self.model_name = model or os.getenv("JEV_MODEL", "")
        self.timeout_seconds = timeout_seconds

    def health_check(self) -> dict[str, Any]:
        return {
            "available": bool(self.endpoint and self.api_key),
            "provider": self.provider_name,
            "model": self.model_name,
            "configured": bool(self.endpoint),
        }

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        if not self.endpoint or not self.api_key or not self.model_name:
            raise RuntimeError("Jev is not configured; set JEV_PROVIDER, JEV_MODEL, and JEV_API_KEY.")
        started = time.perf_counter()
        if self.provider_name == "openrouter":
            instruction = (
                "Return JSON only with exactly these keys: selected, approved, rationale. "
                "selected must be one of the supplied options or null; approved must be boolean.\n"
                f"Task: {request.task}\nOptions: {list(request.options)}\n"
                f"Context: {json.dumps(dict(request.context), default=str)}"
            )
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a constrained narrative decision evaluator."},
                    {"role": "user", "content": instruction},
                ],
                "temperature": 0,
                "max_tokens": 200,
            }
        else:
            payload = {
                "model": self.model_name,
                "task": request.task,
                "options": list(request.options),
                "context": dict(request.context),
            }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if os.getenv("OPENROUTER_SITE_URL"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
        if self.provider_name == "openrouter":
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "AetherScribe")
        http_request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            if self.provider_name == "openrouter":
                content = data["choices"][0]["message"]["content"]
                match = re.search(r"\{.*\}", content, re.DOTALL)
                data = json.loads(match.group(0) if match else content)
            selected = data.get("selected")
            if selected not in request.options and selected is not None:
                raise ValueError("Jev returned an option that was not supplied.")
            if not isinstance(data.get("approved", True), bool):
                raise ValueError("Jev returned a non-boolean approval.")
            return DecisionResponse(
                selected, data.get("approved", True), str(data.get("rationale", "")),
                self.provider_name, self.model_name, (time.perf_counter() - started) * 1000,
            )
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Jev response rejected: {exc}") from exc


class DecisionRouter:
    def __init__(self):
        self.local = LocalRuleDecisionProvider()
        self.jev = JevDecisionProvider()
        self.provider = self.jev if os.getenv("JEV_ENABLED", "false").lower() == "true" else self.local

    def health_check(self) -> dict[str, Any]:
        return self.provider.health_check()

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        try:
            return self.provider.evaluate(request)
        except RuntimeError as exc:
            fallback = self.local.evaluate(request)
            return DecisionResponse(
                fallback.selected, fallback.approved, fallback.rationale, fallback.provider, fallback.model,
                fallback.latency_ms, used_fallback=True, error=str(exc)
            )
