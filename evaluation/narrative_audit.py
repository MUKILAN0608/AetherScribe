"""Deterministic whole-story checks that complement, but do not replace, review."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence
from continuity.repetition import check_repetition


_ABSTRACT_CLAIMS = re.compile(
    r"\b(?:decision hung heavily|resolve was solidified|consequences unfolded|"
    r"tension intensified|sacrifice spoke louder|choices carried weight|"
    r"the stage was set|their resolve was strengthened|the outcome was inevitable)\b",
    re.I,
)
_CONCRETE_ACTION = re.compile(
    r"\b(?:open(?:ed|s)?|close(?:d|s)?|take|took|give|gave|find|found|"
    r"show|showed|reveal(?:ed|s)?|read|write|wrote|sign(?:ed|s)?|"
    r"leave|left|enter(?:ed|s)?|run|ran|move(?:d|s)?|place(?:d|s)?|"
    r"break|broke|cut|call(?:ed|s)?|ask(?:ed|s)?|answer(?:ed|s)?|"
    r"choose|chose|decide(?:d|s)?|refuse(?:d|s)?|agree(?:d|s)?|"
    r"hand(?:ed|s)?|press(?:ed|es)?|pull(?:ed|s)?|push(?:ed|es)?|"
    r"unlock(?:ed|s)?|seal(?:ed|s)?|expose(?:d|s)?|escape(?:d|s)?|"
    r"confront(?:ed|s)?|return(?:ed|s)?|reveal(?:ed|s)?|finds?|copies|copied|"
    r"admits?|releases?|releasing|copies?)\b",
    re.I,
)
_UNRESOLVED_ENDING = re.compile(
    r"\b(?:remains? unresolved|still unanswered|what comes next|"
    r"decision (?:still )?awaits|outcome remains unknown|we will see|"
    r"central question.{0,80}(?:hangs|awaits?|remains open)|"
    r"how (?:will|can|should) (?:they|she|he|we) proceed|"
    r"(?:the )?choice is (?:hers|his|theirs|mine) alone to make|"
    r"the stage is set|final showdown loomed|prepared for whatever (?:lies|comes) ahead)\b",
    re.I,
)


def _body(scene: Mapping[str, Any]) -> str:
    text = str(scene.get("story", ""))
    match = re.search(r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)", text, re.I | re.S)
    return match.group(1).strip() if match else text.strip()


_FAMILY_RELATIONSHIPS = re.compile(
    r"\b(?:sister|brother|mother|father|daughter|son|husband|wife|"
    r"aunt|uncle|cousin|grandmother|grandfather|stepmother|stepfather)\b",
    re.I,
)
_OBSERVABLE_RESULT = re.compile(
    r"\b(?:so|therefore|as a result|which meant|leaving|forcing|forced|"
    r"lost|gained|learned|discovered|exposed|secured|failed|escaped|"
    r"blocked|denied|opened|closed|changed|moved|handed|took|gave|"
    r"broke|revealed|proved|decided|chose|refused|agreed|admit(?:s|ted)?|release(?:s|d)?|"
    r"cop(?:y|ies|ied)|consequence)\b",
    re.I,
)

_INTRODUCED_DEVICE = re.compile(
    r"\b(?:a|an|the)\s+(?:(?:new|hidden|secret|mysterious|strange|unfamiliar|rusted|silver|old)\s+){0,2}"
    r"(switch|button|coin|key|gate|passage|tunnel|device|mechanism|machine|artifact|weapon|"
    r"map|ledger|vial|token|remote|code|password|badge|envelope|photograph|document|seal)\b",
    re.I,
)
_PLOT_USE = re.compile(
    r"\b(?:unlock(?:ed|s|ing)?|open(?:ed|s)?|activate(?:d|s)?|reveal(?:ed|s)?|"
    r"expose(?:d|s)?|prove(?:d|s)?|disable(?:d|s)?|trigger(?:ed|s)?|"
    r"switch(?:ed|es)? on|grant(?:ed|s)? access|show(?:ed|s)? the way)\b",
    re.I,
)
_PLACE_TRANSITION = re.compile(
    r"\b(?:entered|arrived at|reached|moved into|ran into|fled to|returned to|"
    r"crossed into|went to|headed for)\s+(?:the\s+)?"
    r"(room|alley|corridor|hallway|basement|rooftop|roof|street|forest|harbor|"
    r"station|hotel|office|warehouse|tunnel|cellar|courtyard|parking lot|laboratory)\b",
    re.I,
)
_COGNITION_CLAIM = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:already\s+)?(?:knew|knows|had learned|learned|"
    r"realized|remembered|understood|recognized)\b",
)


def audit_scene_narrative(
    text: str, scene_index: int, known_context: str = ""
) -> list[dict[str, Any]]:
    """Critique claims of development when the prose shows no concrete event."""
    scene = {"story": text}
    body = _body(scene)
    findings = []
    abstract = _ABSTRACT_CLAIMS.search(body)
    if abstract and not _CONCRETE_ACTION.search(body):
        findings.append({
            "code": "abstract_development_without_event",
            "scene_id": f"scene-{scene_index + 1:04d}",
            "message": "The scene states a development without dramatizing an action that causes it.",
            "evidence": abstract.group(0),
            "repair_suggestion": "Show what a character does and the observable result of that action.",
        })
    relation = _FAMILY_RELATIONSHIPS.search(body)
    if relation and not _FAMILY_RELATIONSHIPS.search(known_context):
        findings.append({
            "code": "unsupported_family_relationship",
            "scene_id": f"scene-{scene_index + 1:04d}",
            "message": "The scene introduces a family relationship absent from the premise and accepted canon.",
            "evidence": relation.group(0),
            "repair_suggestion": "Remove the relationship or establish it through a grounded, approved story event.",
        })
    prior = known_context.casefold()
    device = _INTRODUCED_DEVICE.search(body)
    if device and device.group(1).casefold() not in prior and _PLOT_USE.search(body):
        findings.append({
            "code": "unseeded_plot_device",
            "scene_id": f"scene-{scene_index + 1:04d}",
            "message": "A newly introduced object is used as an essential plot mechanism without prior setup.",
            "evidence": device.group(0),
            "repair_suggestion": "Remove the mechanism or seed it in an earlier accepted scene before relying on it.",
        })
    if scene_index > 0:
        place = _PLACE_TRANSITION.search(body)
        if place and place.group(1).casefold() not in prior:
            findings.append({
                "code": "unexplained_location_transition",
                "scene_id": f"scene-{scene_index + 1:04d}",
                "message": "The scene moves characters to a location that has no established entry or transition.",
                "evidence": place.group(0),
                "repair_suggestion": "Show the transition from the current location or remain in the established setting.",
            })
    cognition = _COGNITION_CLAIM.search(body)
    if cognition and not re.search(
        rf"\b{re.escape(cognition.group(1))}\b.{{0,100}}\b(?:knew|knows|learned|realized|remembered|understood|recognized)\b",
        prior,
        re.I | re.S,
    ):
        findings.append({
            "code": "unsupported_character_knowledge",
            "scene_id": f"scene-{scene_index + 1:04d}",
            "message": "A character is presented as already knowing or recognizing information without an established source.",
            "evidence": cognition.group(0),
            "repair_suggestion": "Show how the character learned the information or frame the claim as a new discovery.",
        })
    return findings


def audit_story(
    scenes: Sequence[Mapping[str, Any]],
    requested_scene_count: int,
    characters: Mapping[str, str],
    story_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit story-level completeness, concrete dramatization, and ending."""
    findings: list[dict[str, Any]] = []
    ordered = sorted(scenes, key=lambda item: int(item.get("step", -1)))
    steps = [int(scene.get("step", -1)) for scene in ordered]
    if len(ordered) != requested_scene_count or steps != list(range(requested_scene_count)):
        findings.append({
            "code": "incomplete_story_sequence", "severity": "error",
            "message": "The whole story does not contain every requested scene in order.",
            "expected": list(range(requested_scene_count)), "actual": steps,
        })
    established_context = " ".join((
        str((story_contract or {}).get("premise", "")),
        *(str(name) for name in characters.values() if name),
        *(str(value) for value in (story_contract or {}).get("important_objects", ())),
        *(str(value) for value in (story_contract or {}).get("world_rules", ())),
    ))
    prior_scenes: list[dict[str, Any]] = []
    for scene in ordered:
        body = _body(scene)
        index = int(scene.get("step", -1))
        if _ABSTRACT_CLAIMS.search(body) and not _CONCRETE_ACTION.search(body):
            findings.append({
                "code": "abstract_development_without_event", "severity": "error",
                "scene_id": f"scene-{index + 1:04d}",
                "message": "The scene claims emotional or plot development without a concrete on-page action.",
                "evidence": _ABSTRACT_CLAIMS.search(body).group(0),
                "repair_suggestion": "Show the character's action and its observable consequence.",
            })
        if body and not _CONCRETE_ACTION.search(body):
            findings.append({
                "code": "scene_lacks_concrete_action", "severity": "error",
                "scene_id": f"scene-{index + 1:04d}",
                "message": "The scene does not dramatize an observable character action.",
                "repair_suggestion": "Show a character taking an action that changes access, knowledge, risk, or relationship.",
            })
        elif body and not _OBSERVABLE_RESULT.search(body):
            findings.append({
                "code": "scene_lacks_observable_consequence", "severity": "error",
                "scene_id": f"scene-{index + 1:04d}",
                "message": "The prose shows activity but no observable result or changed situation.",
                "repair_suggestion": "State what changed because of the action, using an event rather than a summary claim.",
            })
        if prior_scenes:
            repeated = check_repetition(scene.get("story", body), prior_scenes)
            if "repeated_character_decision" in repeated.reasons:
                findings.append({
                    "code": "repeated_character_decision", "severity": "error",
                    "scene_id": f"scene-{index + 1:04d}",
                    "message": "A character repeats a materially similar decision from an earlier scene.",
                    "repair_suggestion": "Change the decision or demonstrate changed stakes and a different consequence.",
                })
        for finding in audit_scene_narrative(body, index, known_context=established_context):
            finding["severity"] = "error"
            findings.append(finding)
        if not scene.get("quality_gate", {}).get("accepted", False):
            findings.append({
                "code": "scene_missing_technical_acceptance", "severity": "error",
                "scene_id": f"scene-{index + 1:04d}",
                "message": "A non-accepted scene cannot pass whole-story audit.",
            })
        if scene.get("quality_gate", {}).get("accepted", False):
            established_context += " " + body
            prior_scenes.append({"story": scene.get("story", body), "beat": ""})
    if len(ordered) == requested_scene_count and ordered:
        ending = _body(ordered[-1])
        if _UNRESOLVED_ENDING.search(ending):
            findings.append({
                "code": "ending_leaves_central_question_unanswered", "severity": "error",
                "scene_id": f"scene-{requested_scene_count:04d}",
                "message": "The final scene explicitly defers or leaves its outcome unresolved.",
                "evidence": _UNRESOLVED_ENDING.search(ending).group(0),
            })
        final_sentence = re.split(r"(?<=[.!?])\s+", ending.strip())[-1] if ending.strip() else ""
        if final_sentence.rstrip().endswith("?"):
            findings.append({
                "code": "ending_ends_with_open_question", "severity": "error",
                "scene_id": f"scene-{requested_scene_count:04d}",
                "message": "The final line leaves the central dramatic question open.",
                "evidence": final_sentence,
            })
        named_roles = [name for name in characters.values() if name]
        if len(named_roles) >= 2 and sum(name.casefold() in ending.casefold() for name in named_roles[:2]) < 2:
            findings.append({
                "code": "ending_not_grounded_in_central_characters", "severity": "error",
                "scene_id": f"scene-{requested_scene_count:04d}",
                "message": "The ending does not bring the central characters into the resolution.",
                "expected": named_roles[:2],
            })
    return {
        "passed": not findings,
        "requested_scene_count": requested_scene_count,
        "audited_scene_count": len(ordered),
        "audited_scene_indices": steps,
        "findings": findings,
        "scope": "deterministic lexical and structural audit; not a semantic proof",
    }
