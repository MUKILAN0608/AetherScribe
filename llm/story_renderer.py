import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from llm.prompt_templates import build_prompt, MODE_SPECS

# Ensure .env is loaded from project root (Streamlit cwd may differ)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

_SECTION_NARRATIVE = re.compile(
    r"\[SCENE\s+\d+\]\s*NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_TRACE = re.compile(
    r"\[QUANTUM_TRACE\]\s*(.*?)(?=\[COHERENCE\]|$)",
    re.DOTALL | re.IGNORECASE,
)
_LEGACY_RESULT = re.compile(r"RESULT:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)", re.DOTALL | re.IGNORECASE)

# Models to try in order (verified against current Google AI API)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# gemini-2.5-* reserves output budget for internal reasoning; low caps => empty parts
GEMINI_MIN_OUTPUT_TOKENS = 256
GEMINI_SCENE_OUTPUT_TOKENS = 1536

GEMINI_SAFETY = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


def _normalize_api_key(key: str | None) -> str | None:
    if not key:
        return None
    key = key.strip().strip('"').strip("'")
    return key if key else None


class StoryRenderer:
    """Narrative renderer — Gemini when API key is set; no broken HF fallback if key exists."""

    def __init__(self, api_key=None):
        self.logger = logging.getLogger("AetherScribeLab.StoryRenderer")
        self.api_key = _normalize_api_key(api_key or os.getenv("GOOGLE_API_KEY"))
        self.provider = "local"
        self.model = None
        self.model_name = None
        self.hf_endpoint = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"

        if self.api_key:
            self._init_gemini()
        else:
            self.logger.warning(
                "GOOGLE_API_KEY not set. Add it to .env in the project root. Using local templates only."
            )

    def _init_gemini(self):
        genai.configure(api_key=self.api_key)
        env_model = os.getenv("GEMINI_MODEL", "").strip()
        candidates = [env_model] if env_model else []
        candidates.extend(GEMINI_MODELS)

        last_error = None
        for model_name in candidates:
            if not model_name:
                continue
            if self._try_model(model_name):
                return

            last_error = f"model {model_name} failed smoke test"

        self.logger.error(
            "Could not connect to Gemini. Check GOOGLE_API_KEY. Last: %s",
            last_error,
        )
        self.provider = "local"

    def _try_model(self, model_name: str) -> bool:
        """Pick a model only after a real generate_content call succeeds."""
        try:
            model = genai.GenerativeModel(
                model_name,
                safety_settings=GEMINI_SAFETY,
            )
            response = model.generate_content(
                "Reply with exactly: ok",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=GEMINI_MIN_OUTPUT_TOKENS,
                ),
            )
            text = self._extract_gemini_text(response)
            if not text:
                return False
            self.model = model
            self.model_name = model_name
            self.provider = "gemini"
            self.logger.info("Gemini ready: model=%s", model_name)
            return True
        except Exception as exc:
            self.logger.debug("Model %s unavailable: %s", model_name, exc)
            return False

    def _rotate_model_on_failure(self, exc: Exception) -> bool:
        """If the active model 404s at runtime, try the next candidate."""
        msg = str(exc).lower()
        if "404" not in msg and "not found" not in msg and "no longer available" not in msg:
            return False
        tried = {self.model_name}
        for model_name in GEMINI_MODELS:
            if model_name in tried:
                continue
            if self._try_model(model_name):
                return True
        return False

    def _effective_output_tokens(self, requested: int, for_scene: bool = False) -> int:
        """Raise floor so gemini-2.5 thinking budget does not consume all output tokens."""
        floor = GEMINI_SCENE_OUTPUT_TOKENS if for_scene else GEMINI_MIN_OUTPUT_TOKENS
        if self.model_name and "2.5" in self.model_name:
            return max(int(requested), floor)
        return max(int(requested), 128)

    @staticmethod
    def _finish_reason_label(response) -> str:
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return "no_candidates"
            reason = getattr(candidates[0], "finish_reason", None)
            names = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
            return names.get(int(reason), str(reason))
        except Exception:
            return "unknown"

    def _extract_gemini_text(self, response) -> str:
        """Robust text extraction — response.text fails when finish_reason is MAX_TOKENS."""
        if response is None:
            return ""

        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                cand = candidates[0]
                finish = getattr(cand, "finish_reason", None)
                if int(finish) == 3:
                    raise RuntimeError("Gemini blocked output (safety filter).")

                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    chunks = []
                    for part in parts:
                        text = getattr(part, "text", None)
                        if text:
                            chunks.append(text)
                    if chunks:
                        return "".join(chunks).strip()
        except RuntimeError:
            raise
        except Exception:
            pass

        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        feedback = getattr(response, "prompt_feedback", None)
        if feedback:
            block = getattr(feedback, "block_reason", None)
            if block:
                raise RuntimeError(f"Gemini blocked the request: {block}")

        return ""

    def _token_budget(self, mode):
        return GEMINI_SCENE_OUTPUT_TOKENS if mode == "Long Story" else 512

    @staticmethod
    def _constrain_scene_lines(text: str, min_lines: int = 5, max_lines: int = 8) -> str:
        """Normalize to 5–8 sentences; preserve intentional line breaks from the model."""
        if not text:
            return text

        if "\n" in text.strip():
            lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
            if len(lines) >= min_lines:
                return "\n".join(lines[:max_lines])

        compact = re.sub(r"\s+", " ", text.strip())
        sentences = re.split(r"(?<=[.!?…])\s+", compact)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 12]
        if not sentences:
            return compact
        if len(sentences) > max_lines:
            sentences = sentences[:max_lines]
        return "\n".join(sentences)

    def _call_hf(self, prompt, temperature=0.8, max_output_tokens=180):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max(80, min(int(max_output_tokens), 400)),
                "temperature": float(max(0.1, min(temperature, 1.0))),
                "return_full_text": False,
            },
        }
        request = urllib.request.Request(
            self.hf_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(data["error"])
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("summary_text")
                if text:
                    return text.strip()
        raise RuntimeError("No usable text from Hugging Face.")

    def _gemini_generate(self, prompt: str, temperature: float, max_output_tokens: int, for_scene: bool = False):
        tokens = self._effective_output_tokens(max_output_tokens, for_scene=for_scene)
        return self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=tokens,
            ),
        )

    def _generate_text(
        self,
        prompt,
        temperature=0.8,
        max_output_tokens=180,
        default_text="",
        for_scene: bool = False,
    ):
        if self.provider == "gemini" and self.model is not None:
            token_steps = [
                self._effective_output_tokens(max_output_tokens, for_scene=for_scene),
            ]
            if token_steps[0] < 1024 and for_scene:
                token_steps.append(1024)

            last_reason = ""
            for attempt, tokens in enumerate(token_steps):
                try:
                    response = self._gemini_generate(
                        prompt, temperature, tokens, for_scene=for_scene
                    )
                    text = self._extract_gemini_text(response)
                    if text:
                        return text
                    last_reason = self._finish_reason_label(response)
                except Exception as exc:
                    if self._rotate_model_on_failure(exc):
                        return self._generate_text(
                            prompt,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            default_text=default_text,
                            for_scene=for_scene,
                        )
                    self.logger.warning("Gemini call failed: %s", exc)
                    break

                if attempt < len(token_steps) - 1:
                    self.logger.debug(
                        "Gemini empty (%s); retrying with %s output tokens.",
                        last_reason,
                        token_steps[attempt + 1],
                    )

            if last_reason:
                self.logger.warning(
                    "Gemini returned no text (finish=%s, requested=%s). Using fallback.",
                    last_reason,
                    max_output_tokens,
                )

            if default_text:
                return default_text.strip()

            try:
                short = prompt[-2500:] if len(prompt) > 2500 else prompt
                response = self._gemini_generate(
                    short, temperature, token_steps[-1], for_scene=for_scene
                )
                text = self._extract_gemini_text(response)
                if text:
                    return text
            except Exception:
                pass

            return default_text.strip() if default_text else ""

        # Only use HF when no Google API key was configured
        if not self.api_key:
            try:
                return self._call_hf(
                    prompt, temperature=temperature, max_output_tokens=max_output_tokens
                )
            except Exception as exc:
                self.logger.warning("HF fallback failed: %s", exc)

        return default_text.strip() if default_text else ""

    def _extract_narrative(self, raw_text):
        match = _SECTION_NARRATIVE.search(raw_text)
        if match:
            return match.group(1).strip()
        legacy = _LEGACY_RESULT.search(raw_text)
        if legacy:
            return legacy.group(1).strip()
        cleaned = re.sub(r"\[SCENE\s+\d+\]", "", raw_text, flags=re.IGNORECASE).strip()
        for tag in ("[QUANTUM_TRACE]", "[COHERENCE]", "NARRATIVE:"):
            if tag in cleaned:
                cleaned = cleaned.split(tag)[0].strip()
        return cleaned or raw_text.strip()

    def _extract_trace(self, raw_text):
        match = _SECTION_TRACE.search(raw_text)
        if match:
            return match.group(1).strip()
        if "[QUANTUM_TRACE]" in raw_text:
            return raw_text.split("[QUANTUM_TRACE]")[-1].split("[COHERENCE]")[0].strip()
        return ""

    def _normalize_scene_output(self, raw_text, scene_index, actions, rationale, characters=None):
        from ui.decisions import action_to_readable, build_name_map, build_quantum_trace, format_joint_action

        narrative = self._constrain_scene_lines(self._extract_narrative(raw_text))
        trace = self._extract_trace(raw_text)
        name_map = build_name_map(characters or {})
        chosen = format_joint_action(actions or {}, name_map)
        if not trace or trace.startswith("The joint action (:"):
            trace = build_quantum_trace(chosen, trace)
        return (
            f"[SCENE {scene_index}]\n"
            f"NARRATIVE:\n{narrative}\n\n"
            f"[QUANTUM_TRACE]\n{trace}"
        )

    def _fallback_scene(self, state, actions, rationale, scene_index, mode):
        characters = state.get("director_brief", {}).get("characters", {})
        parts = []
        for role, act in actions.items():
            key = role.lower()
            name = characters.get(key, characters.get(role, role))
            parts.append(f"{name} {act.replace('_', ' ')}")
        event = "; ".join(parts)
        genre = state.get("genre", "Drama")

        prose = self._constrain_scene_lines(
            f"The {genre.lower()} air tightened around them as {event}, and no one pretended otherwise.\n"
            f"Small details accumulated—a held breath, a glance withheld, a hand that did not reach.\n"
            f"The room seemed to register the shift before anyone spoke, as if the walls had been listening.\n"
            f"What they chose in that moment would bind the story to consequence, not comfort.\n"
            f"By the time the silence broke, the cost was already written into what came next."
        )
        trace = f"The scene followed the most coherent path available: {rationale[:180].replace(chr(10), ' ')}."
        return self._normalize_scene_output(
            f"NARRATIVE:\n{prose}\n\n[QUANTUM_TRACE]\n{trace}",
            scene_index,
            actions,
            rationale,
            characters=characters,
        )

    def analyze_rejections(
        self, state, chosen_action, rejected_actions, rationale, language="English"
    ):
        genre = state["genre"]
        characters = state.get("director_brief", {}).get("characters", {})
        name_map = {role: name for role, name in characters.items()}
        name_map.update({k.capitalize(): v for k, v in characters.items()})

        rejections_summary = []
        for rej in rejected_actions[:3]:
            rej_text = ", ".join(
                f"{name_map.get(r, name_map.get(r.capitalize(), r))}: {a.replace('_', ' ')}"
                for r, a in rej["action"].items()
            )
            chosen_text = ", ".join(
                f"{name_map.get(r, name_map.get(r.capitalize(), r))}: {a.replace('_', ' ')}"
                for r, a in chosen_action.items()
            )
            prompt = f"""Explain in one clear sentence why this narrative path was not chosen.

Genre: {genre}
Chosen: {chosen_text}
Not chosen: {rej_text}

Use only the character names given. No role labels. Language: {language}. One sentence."""
            response = self._generate_text(
                prompt,
                temperature=0.5,
                max_output_tokens=120,
                default_text=(
                    f"The path involving {rej_text} fit the story less well than {chosen_text} "
                    f"at this point in the narrative."
                ),
            )
            rejections_summary.append(response)
        return rejections_summary

    def render_scene(
        self,
        state,
        actions,
        rationale,
        previous_summary=None,
        language="English",
        temperature=0.88,
        mode="Long Story",
        lite_mode=False,
    ):
        scene_index = state.get("step", 0)
        token_budget = self._token_budget(mode)

        system_preamble = (
            "You write publication-quality literary fiction for a quantum narrative lab. "
            "Each scene must be 5–8 full sentences (one per line, 180+ words total). "
            "Never write a one-line scene. Never deviate from the story premise. "
            "Use only the character names provided.\n\n"
        )
        prompt = system_preamble + build_prompt(
            state, actions, rationale, scene_index, previous_summary, language, mode
        )

        raw = self._generate_text(
            prompt,
            temperature=temperature,
            max_output_tokens=token_budget,
            default_text="",
            for_scene=True,
        )

        characters = state.get("director_brief", {}).get("characters", {})

        if not raw:
            story_text = self._fallback_scene(state, actions, rationale, scene_index, mode)
        else:
            story_text = self._normalize_scene_output(
                raw, scene_index, actions, rationale, characters=characters
            )

        narrative_only = self._extract_narrative(story_text)
        word_count = len(narrative_only.split())
        line_count = len([ln for ln in narrative_only.split("\n") if ln.strip()])

        if word_count < 120 or line_count < 4:
            extend_prompt = (
                f"{system_preamble}"
                "IMPORTANT: Your previous draft was too short. Rewrite with 5–8 full sentences "
                "(minimum 180 words), one sentence per line.\n\n"
                + build_prompt(
                    state, actions, rationale, scene_index, previous_summary, language, mode
                )
            )
            raw_long = self._generate_text(
                extend_prompt,
                temperature=temperature,
                max_output_tokens=token_budget,
                default_text="",
                for_scene=True,
            )
            if raw_long:
                story_text = self._normalize_scene_output(
                    raw_long, scene_index, actions, rationale, characters=characters
                )
                narrative_only = self._extract_narrative(story_text)

        if lite_mode:
            summary = narrative_only[:180] + ("…" if len(narrative_only) > 180 else "")
            return story_text, summary, prompt, 0.85

        summary = self._generate_text(
            f"Summarize in one sentence:\n\n{narrative_only}",
            temperature=0.3,
            max_output_tokens=80,
            default_text=narrative_only[:200] + ("…" if len(narrative_only) > 200 else ""),
        )

        critique_res = self._generate_text(
            f"Rate narrative coherence from 0.0 to 1.0. Reply with only one decimal number.\n\n{narrative_only[:600]}",
            temperature=0.1,
            max_output_tokens=64,
            default_text="0.88",
        )
        match = re.search(r"(\d+\.\d+)", critique_res or "")
        coherence_score = float(match.group(1)) if match else 0.88
        coherence_score = max(0.0, min(1.0, coherence_score))

        return story_text, summary, prompt, coherence_score

    def status(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "has_api_key": bool(self.api_key),
        }
