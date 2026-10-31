"""Decision-first narrative renderer with local and offline providers."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from llm.prompt_templates import build_compact_local_prompt, build_prompt
from llm.providers import GenerationConfig, ProviderRouter

class StoryRenderer:
    """Render prose without allowing a language provider to change locked actions."""

    def __init__(self, api_key: str | None = None):
        self.logger = logging.getLogger("AetherScribeLab.StoryRenderer")
        # Retain the argument for callers from older integrations; credentials are not used.
        self.api_key = None
        self.router = ProviderRouter()
        self.provider = self.router.provider.provider_name
        self.model = self.router.provider.model_name
        self.last_error: str | None = None
        self.last_latency_ms: float | None = None
        self.last_used_fallback = self.provider == "deterministic-local"
        self.last_input_tokens: int | None = None
        self.last_output_tokens: int | None = None
        self.last_estimated_cost_usd: float | None = None
        self.last_provider_metadata: dict[str, Any] | None = None
        self.scene_latency_ms = 0.0
        self.scene_input_tokens = 0
        self.scene_output_tokens = 0
        self.scene_estimated_cost_usd = 0.0

    def status(self) -> dict[str, Any]:
        health = self.router.health_check()
        return {
            "provider": self.provider,
            "model": health.get("model") or self.model,
            "endpoint": health.get("endpoint"),
            "state": health.get("state", "provider_error"),
            "provider_error": self.last_error or health.get("provider_error"),
            "requested_provider": health.get("requested_provider"),
            "has_api_key": bool(self.api_key),
            "local_available": bool(health.get("available")),
            "health": health,
            "fallback": self.provider == "deterministic-local",
            "last_latency_ms": self.last_latency_ms,
            "last_used_fallback": self.last_used_fallback,
            "input_tokens": self.last_input_tokens,
            "output_tokens": self.last_output_tokens,
            "estimated_cost_usd": self.last_estimated_cost_usd,
            "finish_reason": (self.last_provider_metadata or {}).get("finish_reason"),
        }

    def _generate_text(self, prompt: str, temperature: float = 0.8,
                       max_output_tokens: int = 180, default_text: str = "",
                       generation_seed: int | None = None) -> str:
        local_provider = getattr(self.router.provider, "provider_name", "") in {"local-endpoint", "llama.cpp"}
        configured_output_tokens = int(
            os.getenv("NARRATIVE_MAX_OUTPUT_TOKENS", str(max_output_tokens))
        )
        if local_provider:
            configured_output_tokens = min(configured_output_tokens, max_output_tokens)
        if os.getenv("NARRATIVE_FAST_MODE", "true").lower() == "true":
            configured_output_tokens = min(
                configured_output_tokens,
                int(os.getenv("NARRATIVE_FAST_MAX_OUTPUT_TOKENS", "144")),
            )
        retry_match = re.search(r"retry\s+(\d+)", prompt, re.IGNORECASE)
        if retry_match:
            retry_number = int(retry_match.group(1))
            configured_output_tokens = min(
                int(os.getenv("NARRATIVE_RETRY_MAX_OUTPUT_TOKENS", "320")),
                max(configured_output_tokens, 128) + max(0, retry_number - 1) * 64,
            )
        configured_temperature = float(
            os.getenv(
                "NARRATIVE_LOCAL_TEMPERATURE" if local_provider else "NARRATIVE_TEMPERATURE",
                str(temperature),
            )
        )
        try:
            result = self.router.generate(
                prompt,
                GenerationConfig(
                    temperature=configured_temperature,
                    top_p=float(os.getenv("NARRATIVE_TOP_P", "0.8")),
                    repeat_penalty=float(os.getenv("NARRATIVE_REPEAT_PENALTY", "1.05")),
                    max_output_tokens=configured_output_tokens,
                    timeout_seconds=float(os.getenv("NARRATIVE_TIMEOUT_SECONDS", os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180"))),
                    context_size=int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "2048")),
                    seed=generation_seed,
                ),
            )
        except RuntimeError as exc:
            self.last_error = str(exc)
            self.last_used_fallback = False
            self.last_provider_metadata = {"error": str(exc)}
            return default_text
        self.provider = result.provider
        self.model = result.model
        self.last_error = result.error
        self.last_latency_ms = result.latency_ms
        self.last_used_fallback = result.used_fallback
        self.last_input_tokens = result.input_tokens
        self.last_output_tokens = result.output_tokens
        self.last_estimated_cost_usd = result.estimated_cost_usd
        self.last_provider_metadata = dict(result.provider_metadata or {})
        self.scene_latency_ms += result.latency_ms
        self.scene_input_tokens += result.input_tokens or 0
        self.scene_output_tokens += result.output_tokens or 0
        self.scene_estimated_cost_usd += result.estimated_cost_usd or 0.0
        return result.text.strip() or default_text

    @staticmethod
    def _normalize_scene(text: str, actions: dict[str, str], scene_index: int = 0) -> str | None:
        """Accept one complete scene and discard duplicated or placeholder sections."""
        narrative = re.search(
            r"\[SCENE\s+\d+\]\s*NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        body = narrative.group(1).strip() if narrative else text.strip()
        if not body:
            return None
        if "<" in body or ">" in body:
            return None
        trace_match = re.search(
            r"\[QUANTUM_TRACE\]\s*(.*?)(?=\[COHERENCE\]|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        trace = trace_match.group(1).strip() if trace_match else ""
        if "<" in trace or ">" in trace:
            trace = ""
        return (
            f"[SCENE {scene_index + 1}]\nNARRATIVE:\n"
            f"{body}\n\n[QUANTUM_TRACE]\n{trace}"
        )

    def analyze_rejections(self, state, chosen_action, rejected_actions, rationale, language="English"):
        genre = state["genre"]
        characters = state.get("director_brief", {}).get("characters", {})
        name_map = {role.capitalize(): name for role, name in characters.items()}
        results = []
        for rejection in rejected_actions[:2]:
            rejected = ", ".join(
                f"{name_map.get(role, role)}: {action.replace('_', ' ')}"
                for role, action in rejection["action"].items()
            )
            chosen = ", ".join(
                f"{name_map.get(role, role)}: {action.replace('_', ' ')}"
                for role, action in chosen_action.items()
            )
            prompt = (
                f"[SIMPLE STORY AUDIT]\nDOMAIN: {genre}\nPATH WE USED: {chosen}\n"
                f"PATH WE SKIPPED: {rejected}\nREASON: {rationale}\n"
                "Explain in one simple sentence why the skipped path was less suitable. "
                f"Use character names only. LANGUAGE: {language}"
            )
            results.append(self._generate_text(
                prompt, temperature=0.6, max_output_tokens=80,
                default_text="This choice was not as interesting for the story.",
            ))
        return results

    def render_scene(self, state, actions, rationale, previous_summary=None, language="English",
                     temperature=0.8, mode="Short Story", lite_mode=False,
                     memory_context=None, planning_guidance=None, secret_guidance=None,
                     story_continuity=None, repetition_warning=None, generation_seed=None):
        self.scene_latency_ms = 0.0
        self.scene_input_tokens = 0
        self.scene_output_tokens = 0
        self.scene_estimated_cost_usd = 0.0
        scene_index = state.get("step", 0)
        action_summary = ", ".join(
            f"{role} {action.replace('_', ' ')}" for role, action in actions.items()
        )
        if getattr(self.router.provider, "provider_name", "") in {"local-endpoint", "llama.cpp"}:
            prompt = build_compact_local_prompt(
                state,
                actions,
                scene_index,
                language,
                story_continuity,
                planning_guidance,
                repetition_warning,
            )
        else:
            prompt = build_prompt(
                state, actions, rationale, scene_index, previous_summary, language, mode,
                memory_context=memory_context,
                planning_guidance=planning_guidance,
                secret_guidance=secret_guidance,
                story_continuity=story_continuity,
                repetition_warning=repetition_warning,
            )
        scene_token_budget = int(os.getenv("NARRATIVE_SCENE_MAX_OUTPUT_TOKENS", "256"))
        story_text = self._generate_text(prompt, temperature, scene_token_budget, "", generation_seed)
        normalized = self._normalize_scene(story_text, actions, scene_index)
        if normalized is None:
            repair_prompt = (
                f"{prompt}\n\nREPAIR: Return only complete reader-facing narrative prose for scene {scene_index + 1}. "
                "Use 3-5 sentences, include concrete action and consequence, and do not echo the prompt, use angle brackets "
                "or duplicate section headers. Preserve these locked actions exactly: "
                f"{action_summary}."
            )
            normalized = self._normalize_scene(
                self._generate_text(
                    repair_prompt,
                    temperature=0.2,
                    max_output_tokens=scene_token_budget,
                    default_text="",
                    generation_seed=(generation_seed + 1) if generation_seed is not None else None,
                ),
                actions, scene_index
            )
        if normalized is None:
            return "", "", prompt, 0.0
        story_text = normalized
        narrative = re.search(
            r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)",
            story_text,
            re.IGNORECASE | re.DOTALL,
        )
        sentences = re.split(r"(?<=[.!?])\s+", narrative.group(1).strip()) if narrative else []
        summary = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip())[:320]
        return story_text, summary, prompt, None
