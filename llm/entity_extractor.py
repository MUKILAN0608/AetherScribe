"""Gemini-based character and genre extraction from story premises."""

import json
import logging
import re

import google.generativeai as genai

from llm.story_renderer import GEMINI_MIN_OUTPUT_TOKENS, GEMINI_SAFETY, StoryRenderer

logger = logging.getLogger("AetherScribeLab.EntityExtractor")

GENRES = ["Thriller", "Horror", "Comedy", "Romance", "Drama", "Fantasy"]


def _parse_json_object(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def extract_entities_gemini(premise: str, api_key: str) -> dict:
    """
    Returns {
      genre, protagonist, antagonist, ally, detected_names (list)
    }
    """
    if not premise.strip() or not api_key:
        return {}

    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel("gemini-2.5-flash", safety_settings=GEMINI_SAFETY)

    prompt = f"""Extract story entities from this premise. Reply with JSON only, no markdown.

Premise:
{premise.strip()}

Return exactly this JSON shape:
{{
  "genre": one of {GENRES},
  "protagonist": "full character name from premise",
  "antagonist": "full character name from premise",
  "ally": "full character name from premise",
  "detected_names": ["name1", "name2", "name3"]
}}

Rules:
- Use names exactly as they appear in the premise (e.g. Chennai, Vikram, Kavin).
- Assign roles by narrative function (who drives the story, who opposes, who supports).
- detected_names lists the three principal characters in order of importance.
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=GEMINI_MIN_OUTPUT_TOKENS,
        ),
    )

    text = StoryRenderer._extract_gemini_text(response)
    data = _parse_json_object(text)

    genre = data.get("genre", "Drama")
    if genre not in GENRES:
        genre = "Drama"

    roles = {
        "protagonist": str(data.get("protagonist", "")).strip(),
        "antagonist": str(data.get("antagonist", "")).strip(),
        "ally": str(data.get("ally", "")).strip(),
    }
    names = data.get("detected_names") or list(roles.values())
    if isinstance(names, str):
        names = [names]
    names = [str(n).strip() for n in names if n and str(n).strip()]

    for role in roles:
        if not roles[role] and names:
            roles[role] = names.pop(0) if names else ""

    return {
        "genre": genre,
        **roles,
        "detected_names": list(dict.fromkeys([roles["protagonist"], roles["antagonist"], roles["ally"]])),
    }
