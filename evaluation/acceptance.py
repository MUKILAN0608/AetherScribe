"""Plot agnostic structural acceptance checks for generated episode payloads."""

from __future__ import annotations

import re
from typing import Any, Mapping


def evaluate_episode_acceptance(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Check state, dependency, export, and completion invariants without a plot oracle."""
    scenes = list(episode.get("scenes", ()))
    contracts = list(episode.get("scene_contracts", ()))
    requested = int(episode.get("requested_scene_count", 0))
    steps = [int(scene.get("step", -1)) for scene in scenes]
    accepted = [scene for scene in scenes if scene.get("quality_gate", {}).get("accepted")]
    accepted_steps = {int(scene.get("step", -1)) for scene in accepted}
    checks = {
        "contracts_cover_every_planned_scene": (
            len(contracts) == requested
            and [int(item.get("sequence", 0)) for item in contracts] == list(range(1, requested + 1))
        ),
        "accepted_scenes_are_unique_and_ordered": steps == sorted(set(steps)),
        "only_accepted_scenes_are_exportable": len(accepted) == len(scenes),
        "selected_actions_have_valid_vqc_provenance": all(
            _has_valid_candidate_provenance(scene) for scene in accepted
        ),
        "dependencies_require_accepted_predecessors": all(
            int(scene["step"]) == 0 or int(scene["step"]) - 1 in accepted_steps
            for scene in accepted
        ),
        "failed_scenes_do_not_commit": int(episode.get("narrative_state_version", -1)) == len(accepted),
        "failure_blocks_dependents": all(
            int(index) not in accepted_steps for index in episode.get("blocked_scene_indices", ())
        ),
        "attempted_and_failed_counts_match_records": (
            int(episode.get("attempted_scene_count", len(scenes) + len(episode.get("scene_failures", ()))))
            == len(scenes) + len(episode.get("scene_failures", ()))
            and int(episode.get("failed_scene_count", len(episode.get("scene_failures", ()))))
            == len(episode.get("scene_failures", ()))
        ),
        "episode_status_matches_progress": (
            episode.get("status") == (
                "complete" if len(accepted) == requested and requested > 0
                and bool(episode.get("ending_validation", {}).get("valid"))
                else "partial" if accepted else "failed"
            )
        ),
        "contract_status_matches_scene_progress": all(
            item.get("status") in {"accepted", "failed", "blocked", "pending"}
            for item in contracts
        ) and all(
            (item.get("status") == "accepted") == (int(item.get("sequence", 0)) - 1 in accepted_steps)
            for item in contracts
            if item.get("status") in {"accepted", "blocked", "failed"}
        ),
        "manuscript_has_no_internal_trace_markup": all(
            not re.search(r"\[(?:QUANTUM_TRACE|COHERENCE)\]", str(scene.get("story", "")), re.I)
            for scene in scenes
        ),
        "completion_requires_all_scenes_and_ending": (
            episode.get("status") != "complete"
            or (
                len(accepted) == requested
                and bool(episode.get("ending_validation", {}).get("valid"))
                and bool(episode.get("story_complete"))
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "requested_scene_count": requested,
        "attempted_scene_count": int(episode.get("attempted_scene_count", len(scenes) + len(episode.get("scene_failures", ())))),
        "accepted_scene_count": len(accepted),
        "failed_scene_count": len(episode.get("scene_failures", ())),
        "blocked_scene_indices": list(episode.get("blocked_scene_indices", ())),
        "status": episode.get("status", "unknown"),
    }


def _has_valid_candidate_provenance(scene: Mapping[str, Any]) -> bool:
    selection = scene.get("candidate_selection") or {}
    candidate = scene.get("action_candidate") or {}
    ids = selection.get("candidate_ids") or []
    validity = selection.get("candidate_validity") or []
    posterior = selection.get("valid_conditioned_posterior") or []
    chosen = selection.get("selected_candidate_id")
    if not ids or len(ids) != len(validity) or len(ids) != len(posterior):
        return False
    if chosen not in ids or not candidate.get("valid"):
        return False
    index = ids.index(chosen)
    if not validity[index] or abs(sum(posterior) - 1.0) > 1e-6:
        return False
    return all(validity[i] or float(posterior[i]) == 0.0 for i in range(len(ids)))
