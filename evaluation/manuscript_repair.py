"""Bounded whole-manuscript revision that preserves every planned scene slot."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence


def repair_manuscript(
    scenes: Sequence[Mapping[str, Any]],
    requested_scene_count: int,
    *,
    audit: Callable[[Sequence[Mapping[str, Any]], int], dict[str, Any]],
    repair_scene: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], list[dict[str, Any]]], Mapping[str, Any] | None],
    validate_scene: Callable[[Mapping[str, Any], int], tuple[bool, list[str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Attempt one focused repair for each affected provisional scene.

    A repaired draft remains provisional because this pass does not replay its
    canonical event commit. Failed or empty repairs leave the original draft in
    place. Scene count and sequence are invariant across the pass.
    """
    ordered = [deepcopy(dict(scene)) for scene in sorted(scenes, key=lambda item: int(item.get("step", -1)))]
    original_steps = [int(scene.get("step", -1)) for scene in ordered]
    if len(ordered) != requested_scene_count or original_steps != list(range(requested_scene_count)):
        return ordered, audit(ordered, requested_scene_count), []

    before = audit(ordered, requested_scene_count)
    issues_by_step: dict[int, list[dict[str, Any]]] = {}
    for finding in before.get("findings", ()):
        scene_id = str(finding.get("scene_id", ""))
        if scene_id.startswith("scene-"):
            try:
                step = int(scene_id.split("-", 1)[1]) - 1
            except ValueError:
                continue
            issues_by_step.setdefault(step, []).append(dict(finding))

    attempts: list[dict[str, Any]] = []
    for step, findings in sorted(issues_by_step.items()):
        if step < 0 or step >= len(ordered):
            continue
        current = ordered[step]
        # Accepted scenes are tied to committed state. Do not rewrite their prose
        # without replaying the canonical event chain.
        if current.get("quality_gate", {}).get("accepted"):
            continue
        try:
            candidate = repair_scene(current, ordered, findings)
            if not candidate or not str(candidate.get("story", "")).strip():
                attempts.append({"scene_index": step, "repaired": False, "issues": ["empty_repair_draft"]})
                continue
            passed, validation_issues = validate_scene(candidate, step)
            attempts.append({
                "scene_index": step,
                "repaired": bool(passed),
                "issues": list(validation_issues),
                "finding_codes": [item.get("code", "unknown") for item in findings],
            })
            if not passed:
                continue
            current["story"] = str(candidate["story"])
            current["generation_status"] = "generated_provisional"
            current["validation_status"] = "prose_repair_passed_canonical_recommit_required"
            current["quality_gate"] = {
                "accepted": False,
                "status": "provisional",
                "issues": ["canonical_state_not_replayed_after_manuscript_repair"],
            }
            current["manuscript_repair"] = {
                "passed_scene_validation": True,
                "canonical_recommit_required": True,
                "prior_diagnostics": current.get("diagnostics", {}),
            }
        except Exception as exc:  # A repair failure must never discard the saved draft.
            attempts.append({
                "scene_index": step,
                "repaired": False,
                "issues": ["repair_provider_or_validator_error"],
                "error": str(exc),
            })

    final_steps = [int(scene.get("step", -1)) for scene in ordered]
    if final_steps != original_steps or len(ordered) != requested_scene_count:
        raise RuntimeError("manuscript repair changed the requested scene count or order")
    return ordered, audit(ordered, requested_scene_count), attempts
