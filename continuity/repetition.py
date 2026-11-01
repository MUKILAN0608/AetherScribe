from dataclasses import dataclass
import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _body(text: str) -> str:
    match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)", text, re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def _opening(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", _body(text))
    return _normalize(" ".join(sentences[:1]))[:180]


def _opening_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]{3,}", _normalize(left)))
    right_tokens = set(re.findall(r"[a-z0-9]{3,}", _normalize(right)))
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", _normalize(_body(text))))


def _sentences(text: str) -> set[str]:
    return {
        normalized
        for normalized in (_normalize(sentence) for sentence in re.split(r"(?<=[.!?])\s+", _body(text)))
        if len(normalized.split()) >= 8
    }


_DECISION = re.compile(
    r"\b(?:decid(?:e|es|ed)|choos(?:e|es|ing)|chose|chooses|refus(?:e|es|ed)|"
    r"agree(?:s|d)?|commit(?:s|ted)?|betray(?:s|ed)?|confess(?:es|ed)?|"
    r"surrender(?:s|ed)?|sacrif(?:ice|ices|iced))\b",
    re.I,
)


def _decision_sentences(text: str) -> list[str]:
    return [
        _normalize(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", _body(text))
        if len(_normalize(sentence).split()) >= 5 and _DECISION.search(sentence)
    ]


@dataclass(frozen=True)
class RepetitionResult:
    repeated: bool
    reasons: tuple[str, ...] = ()
    similarity: float = 0.0
    matched_opening: str = ""


def check_repetition(text: str, prior: list[dict], action_beat: str = "") -> RepetitionResult:
    current = _normalize(_body(text))
    current_tokens = _tokens(text)
    current_opening = _opening(text)
    current_sentences = _sentences(text)
    current_decisions = _decision_sentences(text)
    reasons = []
    best_similarity = 0.0
    matched_opening = ""
    for item in prior[-8:]:
        previous = item.get("story", "")
        normalized_previous = _normalize(_body(previous))
        if current and current == normalized_previous:
            reasons.append("duplicate_scene")
        if current_sentences.intersection(_sentences(previous)):
            reasons.append("repeated_sentence")
        previous_tokens = _tokens(previous)
        similarity = (
            len(current_tokens & previous_tokens) / len(current_tokens | previous_tokens)
            if current_tokens and previous_tokens else 0.0
        )
        best_similarity = max(best_similarity, similarity)
        if similarity >= 0.82:
            reasons.append("similar_scene")
        prior_opening = _normalize(item.get("opening", ""))
        opening_similarity = _opening_similarity(current_opening, prior_opening)
        shared_opening_tokens = set(re.findall(r"[a-z0-9]{3,}", current_opening)) & set(
            re.findall(r"[a-z0-9]{3,}", prior_opening)
        )
        if current_opening and prior_opening and (
            (current_opening == prior_opening and similarity >= 0.65)
            or (len(shared_opening_tokens) >= 6 and opening_similarity >= 0.35)
        ):
            reasons.append("repeated_scene_opening")
            sentence = re.split(r"(?<=[.!?])\s+", _body(previous), maxsplit=1)[0].strip()
            if sentence:
                matched_opening = sentence
        if action_beat and action_beat == item.get("beat", ""):
            reasons.append("repeated_narrative_beat")
        for decision in current_decisions:
            for previous_decision in _decision_sentences(previous):
                left = set(decision.split())
                right = set(previous_decision.split())
                overlap = len(left & right) / len(left | right) if left and right else 0.0
                ratio = SequenceMatcher(None, decision, previous_decision).ratio()
                if overlap >= 0.48 or ratio >= 0.76:
                    reasons.append("repeated_character_decision")
                    break
            if "repeated_character_decision" in reasons:
                break
    return RepetitionResult(bool(reasons), tuple(dict.fromkeys(reasons)), best_similarity, matched_opening)
