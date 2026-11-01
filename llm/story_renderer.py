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
        startup_health = self.router.health_check()
        if startup_health.get("state") == "provider_error":
            self.provider = f"{self.router.requested_provider} (unavailable)"
            self.model = startup_health.get("configured_model") or startup_health.get("model") or "configured model"
        else:
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
        configured_top_p = float(os.getenv("NARRATIVE_TOP_P", "0.8"))
        if retry_match:
            # Bounded sampling diversity helps a repair escape the rejected
            # wording while retaining the same scene contract and locked action.
            configured_temperature = min(1.15, configured_temperature + 0.12 * retry_number)
            configured_top_p = min(0.95, configured_top_p + 0.04 * retry_number)
        try:
            result = self.router.generate(
                prompt,
                GenerationConfig(
                    temperature=configured_temperature,
                    top_p=configured_top_p,
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
            health = self.router.health_check()
            if health.get("state") == "provider_error":
                self.provider = f"{self.router.requested_provider} (unavailable)"
                self.model = health.get("configured_model") or health.get("model") or self.model
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

    @staticmethod
    def _join_continuation(prefix: str, continuation: str) -> str:
        """Join generated prose while removing an exact token overlap at the seam."""
        left = prefix.rstrip()
        right = continuation.strip()
        right = re.split(r"\[QUANTUM_TRACE\]|\[COHERENCE\]", right, maxsplit=1, flags=re.I)[0].strip()
        right = re.sub(r"^\[?(?:SCENE|NARRATIVE)\b[^\n]*\]?\s*", "", right, flags=re.I)
        right = re.sub(r"^NARRATIVE:\s*", "", right, flags=re.I)
        left_words, right_words = left.split(), right.split()
        max_overlap = min(len(left_words), len(right_words), 100)
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if [word.casefold() for word in left_words[-size:]] == [word.casefold() for word in right_words[:size]]:
                overlap = size
                break
        return left + (" " if left and right_words[overlap:] else "") + " ".join(right_words[overlap:])

    @staticmethod
    def _opening_sentence(text: str) -> str:
        match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)", text, re.I | re.S)
        body = match.group(1).strip() if match else text.strip()
        body = re.sub(r"^\[SCENE\s+\d+\]\s*", "", body, flags=re.I)
        return re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0].strip()

    @staticmethod
    def _continuation_word_budget(prefix: str, maximum_words: int) -> int:
        """Return remaining prose capacity while reserving words for a clean ending."""
        used = len(re.findall(r"\b[\w'-]+\b", prefix))
        return max(0, int(maximum_words) - used - 8)

    @staticmethod
    def _remove_exact_duplicate_sentences(text: str) -> tuple[str, int]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
        seen = set()
        kept = []
        removed = 0
        for sentence in sentences:
            key = re.sub(r"\W+", " ", sentence.casefold()).strip()
            if key and key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(sentence)
        return " ".join(kept), removed

    @staticmethod
    def _trim_to_complete_sentences(text: str, word_limit: int) -> tuple[str, bool]:
        """Fit an overlong draft by dropping only trailing whole sentences."""
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
        kept = []
        for sentence in sentences:
            candidate = " ".join((*kept, sentence))
            if len(re.findall(r"\b[\w'-]+\b", candidate)) > int(word_limit):
                break
            kept.append(sentence)
        result = " ".join(kept)
        return result, bool(kept) and len(kept) < len(sentences)

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
        initial_metadata = dict(self.last_provider_metadata or {})
        initial_finish_reason = initial_metadata.get("finish_reason")
        initial_error = self.last_error
        initial_fallback = self.last_used_fallback
        narrative_match = re.search(
            r"(?:NARRATIVE:\s*)?(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
            story_text, re.I | re.S,
        )
        prefix = narrative_match.group(1).strip() if narrative_match else ""
        prefix = re.sub(r"^\[SCENE\s+\d+\]\s*", "", prefix, flags=re.I)
        prefix = re.sub(r"^NARRATIVE:\s*", "", prefix, flags=re.I)
        configured_word_max = int(os.getenv("NARRATIVE_MAX_SCENE_WORDS", "130"))
        remaining_words = self._continuation_word_budget(prefix, configured_word_max)
        # Continue only a non-empty, recognizably prose prefix that the provider
        # explicitly reports as token-truncated. Other malformed output uses repair.
        if (initial_finish_reason == "length" and len(prefix.split()) >= 8 and remaining_words >= 12
                and not re.search(r"<[^>]*>|\[SCENE\s+\d+\].*\[SCENE\s+\d+\]", prefix, re.I | re.S)):
            continuation_prompt = (
                f"{prompt}\n\nSAFE CONTINUATION: Continue this exact scene from the final unfinished words. "
                "Keep the original scene contract, locked actions, and story state. Do not restart, repeat a paragraph, "
                f"add a new scene header, or add a trace. Use at most {remaining_words} additional words. "
                "Finish only the planned event and consequence, then stop. Return continuation prose only.\n"
                f"GENERATED PREFIX (preserve exactly; continue after it):\n{prefix}"
            )
            continuation_tokens = min(scene_token_budget, max(48, remaining_words * 2))
            continuation = self._generate_text(
                continuation_prompt, temperature, continuation_tokens, "",
                (generation_seed + 1) if generation_seed is not None else None,
            )
            if continuation.strip() and not self.last_used_fallback:
                continuation_metadata = dict(self.last_provider_metadata or {})
                continued = self._join_continuation(prefix, continuation)
                story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{continued}"
                self.last_provider_metadata = {
                    **continuation_metadata,
                    "continuation_used": True,
                    "initial_finish_reason": initial_finish_reason,
                    "initial_output_tokens": initial_metadata.get("completion_tokens"),
                    "continuation_finish_reason": continuation_metadata.get("finish_reason"),
                }
                self.last_error = initial_error or self.last_error
                self.last_used_fallback = initial_fallback or self.last_used_fallback
            else:
                story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{prefix}"
                self.last_provider_metadata = {**initial_metadata, "continuation_used": False}
        elif initial_finish_reason == "length":
            story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{prefix}"
            self.last_provider_metadata = {
                **initial_metadata, "continuation_used": False,
                "continuation_skipped_reason": "insufficient_scene_word_budget",
            }
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
        removed_duplicate_sentences = 0
        if repetition_warning and re.search(r"duplicate|repeated sentence", repetition_warning, re.I):
            body_match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)", story_text, re.I | re.S)
            current_body = body_match.group(1).strip() if body_match else ""
            repaired_body, removed_duplicate_sentences = self._remove_exact_duplicate_sentences(current_body)
            if removed_duplicate_sentences:
                story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{repaired_body}"
                self.last_provider_metadata = {
                    **(self.last_provider_metadata or {}),
                    "exact_duplicate_sentences_removed": removed_duplicate_sentences,
                }
        length_target = re.search(r"Cut the passage to\s+(\d+)\s+words maximum", repetition_warning or "", re.I)
        if length_target:
            target_words = int(length_target.group(1))
            body_match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)", story_text, re.I | re.S)
            current_body = body_match.group(1).strip() if body_match else ""
            current_word_count = len(re.findall(r"\b[\w'-]+\b", current_body))
            if current_word_count > target_words:
                original_scene_metadata = dict(self.last_provider_metadata or {})
                compression_prompt = (
                    f"Rewrite the complete scene below in no more than {target_words} words. Return prose only. "
                    "Preserve the same named characters, established facts, action, causal order, and consequences. "
                    "Remove repetition and decorative setup. Do not add events or alternatives. If this is the ending, "
                    "retain the concrete resolution and lasting consequence. Use complete sentences and stop at the word limit.\n"
                    f"SCENE TO COMPRESS:\n{current_body}"
                )
                compressed = self._generate_text(
                    compression_prompt, temperature=0.45,
                    max_output_tokens=min(scene_token_budget, max(128, target_words * 2)),
                    default_text="",
                    generation_seed=(generation_seed + 11) if generation_seed is not None else None,
                )
                compressed_match = re.search(
                    r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
                    compressed, re.I | re.S,
                )
                compressed_body = (compressed_match.group(1) if compressed_match else compressed).strip()
                compressed_body = re.sub(r"^\[SCENE\s+\d+\]\s*", "", compressed_body, flags=re.I)
                compressed_words = len(re.findall(r"\b[\w'-]+\b", compressed_body))
                if (compressed_body and compressed_words <= target_words
                        and re.search(r"[.!?][\"'’)]*$", compressed_body)):
                    story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{compressed_body}"
                    self.last_provider_metadata = {
                        **original_scene_metadata,
                        "focused_length_compression": True,
                        "focused_compression_finish_reason": (self.last_provider_metadata or {}).get("finish_reason"),
                    }
                else:
                    self.last_provider_metadata = {
                        **original_scene_metadata,
                        "focused_length_compression": False,
                        "focused_compression_finish_reason": (self.last_provider_metadata or {}).get("finish_reason"),
                    }
        required_opening = re.search(
            r"FIRST WORD[^\n]{0,100}?MUST be\s+([\w'-]+)", repetition_warning or "", re.I,
        )
        if required_opening:
            required_name = required_opening.group(1)
            scene_match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)", story_text, re.I | re.S)
            body = scene_match.group(1).strip() if scene_match else ""
            original_opening = self._opening_sentence(body)
            if not original_opening.casefold().startswith(required_name.casefold()):
                scene_generation_metadata = dict(self.last_provider_metadata or {})
                rejected_match = re.search(r"rejected opening was:\s*['\"](.+?)['\"]", repetition_warning, re.I)
                rejected = rejected_match.group(1) if rejected_match else "the rejected opening"
                repair_prompt = (
                    f"REWRITE ONLY ONE SENTENCE. Return one complete opening sentence, no labels or explanation. "
                    f"Its first word MUST be {required_name}. It must preserve the scene facts and lead naturally into this scene. "
                    f"Do not use the same subject, action, image, or rhythm as this rejected sentence: {rejected!r}. "
                    f"Scene context: {body[:900]}"
                )
                rewritten = self._generate_text(
                    repair_prompt, temperature=0.65, max_output_tokens=128, default_text="",
                    generation_seed=(generation_seed + 7) if generation_seed is not None else None,
                )
                opening_repair_metadata = dict(self.last_provider_metadata or {})
                rewritten_opening = self._opening_sentence(rewritten)
                if (rewritten_opening.casefold().startswith(required_name.casefold())
                        and re.search(r"[.!?][\"'’)]*$", rewritten_opening)):
                    remainder = body[len(original_opening):].lstrip()
                    story_text = (
                        f"[SCENE {scene_index + 1}]\nNARRATIVE:\n"
                        f"{rewritten_opening}{(' ' + remainder) if remainder else ''}"
                    )
                    self.last_provider_metadata = {
                        **scene_generation_metadata,
                        "focused_opening_rewrite": True,
                        "focused_opening_finish_reason": opening_repair_metadata.get("finish_reason"),
                        "required_opening_character": required_name,
                    }
                else:
                    self.last_provider_metadata = {
                        **scene_generation_metadata,
                        "focused_opening_rewrite": False,
                        "focused_opening_finish_reason": opening_repair_metadata.get("finish_reason"),
                        "required_opening_character": required_name,
                    }
        if repetition_warning and re.search(r"duplicate|repeated sentence", repetition_warning, re.I):
            body_match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)", story_text, re.I | re.S)
            body = body_match.group(1).strip() if body_match else ""
            repaired_body, removed = self._remove_exact_duplicate_sentences(body)
            if removed:
                story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{repaired_body}"
                metadata = dict(self.last_provider_metadata or {})
                metadata["exact_duplicate_sentences_removed"] = (
                    int(metadata.get("exact_duplicate_sentences_removed", 0)) + removed
                )
                self.last_provider_metadata = metadata
        if length_target:
            body_match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)", story_text, re.I | re.S)
            body = body_match.group(1).strip() if body_match else ""
            if len(re.findall(r"\b[\w'-]+\b", body)) > int(length_target.group(1)):
                trimmed, changed = self._trim_to_complete_sentences(body, int(length_target.group(1)))
                minimum = int(os.getenv("NARRATIVE_MIN_SCENE_WORDS", "45"))
                if changed and len(re.findall(r"\b[\w'-]+\b", trimmed)) >= minimum:
                    story_text = f"[SCENE {scene_index + 1}]\nNARRATIVE:\n{trimmed}"
                    metadata = dict(self.last_provider_metadata or {})
                    metadata["bounded_complete_sentence_trim"] = True
                    self.last_provider_metadata = metadata
        narrative = re.search(
            r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)",
            story_text,
            re.IGNORECASE | re.DOTALL,
        )
        sentences = re.split(r"(?<=[.!?])\s+", narrative.group(1).strip()) if narrative else []
        summary = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip())[:320]
        return story_text, summary, prompt, None
