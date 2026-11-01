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
    "selected_probability",
    "posterior mass",
    "action_id:",
    "scene contract:",
    "policy metadata",
    "decision trace:",
)

_GENERIC_FALLBACK_MARKERS = (
    "the story advances as",
    "entered the established setting",
    "locked actions forced the conflict",
    "the conflict reached its sharpest pressure",
    "the selected action carried the story forward",
)


@dataclass(frozen=True)
class ValidationError:
    code: str
    category: str
    scene_id: str
    severity: str
    message: str
    evidence: str
    expected: str
    actual: str
    repair_suggestion: str
    retry_permitted: bool


def _validation_errors(issues, scene_index, text):
    categories = {
        "truncated_output": "truncation", "scene_above_maximum_length": "length",
        "scene_below_minimum_length": "length", "repeated_scene_opening": "narrative_quality",
        "repeated_sentence": "narrative_quality", "similar_scene": "narrative_quality",
        "missing_user_facing_content": "structure", "internal_metadata_leak": "action_to_prose_leakage",
        "internal_action_label_leak": "action_to_prose_leakage", "unsupported_location": "continuity",
        "unsupported_time_jump": "continuity", "premise_restatement": "causal_progression",
        "duplicate_sentence": "narrative_quality", "generic_fallback_text": "structure",
    }
    messages = {
        "truncated_output": ("Output ended before a complete sentence.", "Complete final sentence", "Incomplete or absent ending"),
        "repeated_scene_opening": ("Opening repeats an earlier scene.", "Distinct opening", "Overlapping opening"),
        "scene_above_maximum_length": ("Scene exceeds the configured word limit.", "Within configured limit", "Over limit"),
        "scene_below_minimum_length": ("Scene is shorter than the configured minimum.", "At least configured minimum", "Under minimum"),
        "missing_user_facing_content": ("No reader-facing narrative was produced.", "Non-empty narrative", "Empty narrative"),
        "internal_metadata_leak": ("Internal planning metadata leaked into prose.", "Reader-facing prose", "Internal marker present"),
        "internal_action_label_leak": ("An internal action label leaked into prose.", "Natural-language realization", "Encoded action label present"),
        "unsupported_location": ("The prose introduces an unsupported location.", "Established location", "New location asserted"),
        "unsupported_time_jump": ("The prose introduces an unsupported time jump.", "Established timeline", "Unplanned year asserted"),
        "premise_restatement": ("The scene restates the premise instead of progressing it.", "New causal progression", "Premise overlap"),
        "duplicate_sentence": ("The scene repeats a sentence.", "Unique sentences", "Duplicate sentence"),
        "generic_fallback_text": ("Generic fallback prose was detected.", "Specific scene prose", "Generic template marker"),
    }
    result = []
    for issue in dict.fromkeys(issues):
        code = issue.split(":", 1)[0]
        message, expected, actual = messages.get(code, (issue.replace("_", " ").capitalize(), "Satisfy scene contract", issue))
        result.append(ValidationError(
            code=code, category=categories.get(code, "continuity"), scene_id=f"scene-{scene_index + 1:04d}",
            severity="error", message=message, evidence=text[:240], expected=expected, actual=actual,
            repair_suggestion=_correction_instruction((issue,)), retry_permitted=True,
        ))
    return tuple(result)


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
    errors: tuple[ValidationError, ...] = ()


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
    forbidden_internal_terms: tuple[str, ...] = (),
) -> SceneQuality:
    """Run the bounded final gate immediately before a scene is accepted."""
    continuity = validate_scene_text(text, scene_index, characters)
    repetition = check_repetition(text, prior, action_beat)
    body = _narrative_body(text).lower()
    issues = list(continuity.issues)
    if any(marker in body for marker in _INTERNAL_MARKERS):
        issues.append("internal_metadata_leak")
    if re.search(r"\b(?:the\s+)?(?:protagonist|antagonist|ally)\b", body, re.I):
        issues.append("internal_metadata_leak")
    # Encoded action identifiers are internal control signals. Match them
    # dynamically from the locked action; do not maintain a story vocabulary.
    for term in forbidden_internal_terms:
        term = str(term).strip()
        if not term or not re.search(r"[_:/#]", term):
            continue
        if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", body):
            issues.append("internal_action_label_leak")
            break
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
        errors=_validation_errors(tuple(dict.fromkeys(issues)), scene_index, body),
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
        "internal_action_label_leak": "Translate the selected action into natural prose; remove its encoded internal identifier.",
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
    if not re.search(r"\b(resolv(?:e|es|ed|ing)|end(?:s|ed|ing)?|chang(?:e|es|ed|ing)|settled|surviv(?:e|es|ed|ing)|escap(?:e|es|ed|ing)|defeat(?:s|ed|ing)?|reconcil(?:e|es|ed|ing)|outcome|consequence)\b", body, re.I):
        issues.append("final_scene_lacks_resolution_signal")
    if re.search(
        r"\b(?:remains? unanswered|still unanswered|remains? unresolved|still unresolved|"
        r"loomed larger|what would (?:they|she|he|we) do|what comes next|what came next|"
        r"face what comes next|impact unknown|outcome hanging on|answer.{0,50}(?:lies|lay) within|"
        r"(?:decision|choice) that (?:awaits?|remains to be made)|"
        r"(?:central question|outcome|consequence).{0,120}\b(?:either|unknown|uncertain|awaited|or|versus)\b|"
        r"no answer (?:came|followed))\b",
        body, re.I,
    ):
        issues.append("final_scene_explicitly_unresolved")
    return not issues, tuple(issues)
