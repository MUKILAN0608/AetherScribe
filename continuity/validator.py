from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ContinuityValidation:
    valid: bool
    issues: tuple[str, ...] = ()


def validate_scene_text(text: str, scene_index: int, characters: dict) -> ContinuityValidation:
    """Apply bounded structural/readability checks; never claims semantic proof."""
    issues = []
    narrative = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)", text, re.I | re.S)
    body = narrative.group(1).strip() if narrative else ""
    if not body:
        issues.append("missing_narrative")
    if "<" in body or ">" in body:
        issues.append("placeholder_text")
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
    if scene_index == 0:
        required = [characters.get("protagonist"), characters.get("antagonist")]
        if any(name and name not in body for name in required):
            issues.append("opening_missing_named_conflict")
        if len(sentences) < 3:
            issues.append("opening_too_short")
    elif len(sentences) < 2:
        issues.append("scene_too_short")
    return ContinuityValidation(not issues, tuple(issues))
