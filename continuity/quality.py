from dataclasses import dataclass
import re

from .repetition import RepetitionResult, check_repetition
from .validator import ContinuityValidation, validate_scene_text


_INTERNAL_MARKERS = (
    "system instruction",
    "developer message",
    "internal prompt",
    "tool call",
    "raw metadata",
    "current situation",
    "protagonist goal",
    "established facts",
    "character knowledge",
    "advisory narrative plan",
    "output format",
    "latest accepted change",
    "previous accepted change",
    "scene purpose:",
    "scene event:",
    "required clue:",
    "actions to incorporate",
    "write the following",
    "this event involves",
    "scene number:",
    "write exactly",
)

_GENERIC_FALLBACK_MARKERS = (
    "the story advances as",
    "entered the established setting",
    "locked actions forced the conflict",
    "the conflict reached its sharpest pressure",
    "the selected action carried the story forward",
)


@dataclass(frozen=True)
class SceneQuality:
    accepted: bool
    continuity: ContinuityValidation
    repetition: RepetitionResult
    issues: tuple[str, ...] = ()
    severity: str = "error"
    retry_required: bool = True
    correction_instruction: str = ""
    word_count: int = 0


def _narrative_body(text: str) -> str:
    match = re.search(
        r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def validate_scene_output(
    text: str,
    scene_index: int,
    characters: dict,
    prior: list[dict],
    action_beat: str = "",
    min_words: int = 0,
    max_words: int | None = None,
    required_terms: tuple[str, ...] = (),
    premise: str = "",
) -> SceneQuality:
    """Run the bounded final gate immediately before a scene is accepted."""
    continuity = validate_scene_text(text, scene_index, characters)
    repetition = check_repetition(text, prior, action_beat)
    body = _narrative_body(text).lower()
    issues = list(continuity.issues)
    if any(marker in body for marker in _INTERNAL_MARKERS):
        issues.append("internal_metadata_leak")
    if any(marker in body for marker in _GENERIC_FALLBACK_MARKERS):
        issues.append("generic_fallback_text")
    if re.search(r"\b(?:city|town|village|country|skyscraper)\s+of\s+[A-Z][a-z]+", _narrative_body(text)):
        issues.append("unsupported_location")
    if re.search(r"\b(?:in|during)\s+(?:19|20)\d{2}\b", _narrative_body(text)):
        issues.append("unsupported_time_jump")
    if not body:
        issues.append("missing_user_facing_content")
    words = re.findall(r"\b[\w'-]+\b", body)
    if body and not re.search(r"[.!?…][\"'”’)]*$", body.strip()):
        issues.append("truncated_output")
    if min_words and len(words) < min_words:
        issues.append("scene_below_minimum_length")
    if max_words is not None and len(words) > max_words:
        issues.append("scene_above_maximum_length")
    normalized_body = _normalize_sentence(body)
    # Accept a structured fact once either part of its planned evidence pair is
    # stated, with alternatives for common paraphrases.
    evidence_groups = (tuple(required_terms),) if required_terms else ()
    if len(required_terms) == 2:
        evidence_groups = (tuple(required_terms),)
    for group in evidence_groups:
        matched = False
        missing = []
        for term in group:
            normalized_term = _normalize_sentence(term)
            aliases = {
                "access history": ("badge audit", "credential log", "entry record", "access log"),
                "credentials": ("badge", "account", "credential"),
                "power allocation": ("power allotment", "electricity allocation", "power feed"),
                "maintenance ledger": ("service ledger", "maintenance log", "repair register"),
                "records custodian": ("archivist", "records keeper", "custodian"),
                "vault inventory": ("vault catalogue", "inventory of the vault", "archive inventory"),
                "chain of custody": ("custody record", "evidence transfer log", "custody log"),
            }.get(normalized_term, ())
            if any(_normalize_sentence(candidate) in normalized_body for candidate in (term, *aliases)):
                matched = True
            else:
                missing.append(term)
        if not matched:
            issues.append("missing_planned_evidence:" + " / ".join(missing))
    if scene_index > 0 and premise:
        premise_terms = set(_normalize_sentence(premise).split())
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence_terms = set(_normalize_sentence(sentence).split())
            overlap = len(premise_terms & sentence_terms) / len(premise_terms | sentence_terms) if premise_terms and sentence_terms else 0.0
            if overlap >= 0.55:
                issues.append("premise_restatement")
                break
    sentence_bodies = [
        _normalize_sentence(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if sentence.strip()
    ]
    if len(sentence_bodies) != len(set(sentence_bodies)):
        issues.append("duplicate_sentence")
    issues.extend(repetition.reasons)
    correction = _correction_instruction(issues)
    return SceneQuality(
        accepted=not issues,
        continuity=continuity,
        repetition=repetition,
        issues=tuple(dict.fromkeys(issues)),
        severity="error" if issues else "ok",
        retry_required=bool(issues),
        correction_instruction=correction,
        word_count=len(words),
    )


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"\W+", " ", sentence.lower()).strip()


def _correction_instruction(issues: list[str]) -> str:
    if not issues:
        return ""
    instructions = {
        "scene_above_maximum_length": "Reduce repetition while preserving the concrete event and consequence.",
        "scene_below_minimum_length": "Add concrete action, reaction, and consequence without restating the premise.",
        "duplicate_sentence": "Replace duplicated sentences with a distinct development.",
        "truncated_output": "Finish the final sentence and keep the complete scene within the word limit.",
        "internal_metadata_leak": "Remove all internal instructions, traces, and provider metadata from the narrative.",
        "repeated_narrative_beat": "Advance the conflict with a new event while preserving the locked action.",
        "repeated_sentence": "Replace repeated prose with a new event from this scene's plan.",
        "opening_too_short": "Write at least three complete sentences for the opening scene.",
        "scene_too_short": "Add a complete character action and its consequence.",
        "opening_missing_named_conflict": "Name the protagonist and antagonist and show their conflict.",
        "unsupported_location": "Remove invented locations unless a prior accepted scene or the premise establishes them.",
        "unsupported_time_jump": "Keep the scene in the established timeline; do not introduce an unplanned year or time jump.",
        "premise_restatement": "Do not repeat the premise; advance the current scene's unique event instead.",
        "provider_truncated_output": "Shorten the scene and regenerate it so the model finishes before its output limit.",
    }
    messages = []
    for issue in issues:
        if issue.startswith("missing_planned_evidence:"):
            detail = issue.partition(":")[2].replace(" / ", " or ")
            messages.append(f"Include the planned evidence fact in reader-facing terms ({detail}) and show what it proves.")
        else:
            messages.append(instructions.get(issue, "Preserve the planned event and consequence while correcting the prose."))
    return " ".join(dict.fromkeys(messages))


def validate_story_ending(text: str, characters: dict) -> tuple[bool, tuple[str, ...]]:
    """Apply bounded ending checks without claiming full semantic understanding."""
    body = _narrative_body(text)
    issues = []
    if not body:
        issues.append("missing_final_narrative")
    for role in ("protagonist", "antagonist"):
        name = characters.get(role)
        if name and name not in body:
            issues.append(f"final_scene_missing_{role}")
    if len(re.findall(r"\b[\w'-]+\b", body)) < 30:
        issues.append("final_scene_too_short")
    if not re.search(r"\b(resolved|ended|changed|settled|survived|escaped|defeated|reconciled|outcome|consequence)\b", body, re.I):
        issues.append("final_scene_lacks_resolution_signal")
    return not issues, tuple(issues)
