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
    generated_steps = {int(scene.get("step", -1)) for scene in scenes if str(scene.get("story", "")).strip()}
    provisional = [scene for scene in scenes if scene.get("generation_status") == "generated_provisional"]
    complete = (
        len(accepted) == requested and requested > 0
        and bool(episode.get("ending_validation", {}).get("valid"))
        and bool(episode.get("whole_story_audit", {}).get("passed"))
        and bool(episode.get("story_complete"))
    )
    expected_status = (
        "complete" if complete else "generated" if len(generated_steps) == requested
        else "partial" if generated_steps else "failed"
    )
    checks = {
        "contracts_cover_every_planned_scene": (
            len(contracts) == requested
            and [int(item.get("sequence", 0)) for item in contracts] == list(range(1, requested + 1))
        ),
        "story_contract_and_blueprint_exist_before_scene_generation": (
            not episode.get("story_contract") and not episode.get("scene_blueprints")
            or (
                int(episode.get("story_contract", {}).get("required_scene_count", -1)) == requested
                and len(episode.get("scene_blueprints", ())) == requested
                and [int(item.get("sequence", 0)) for item in episode.get("scene_blueprints", ())]
                == list(range(1, requested + 1))
            )
        ),
        "accepted_scenes_are_unique_and_ordered": steps == sorted(set(steps)),
        "accepted_scene_count_matches_records": int(episode.get("accepted_scene_count", len(accepted))) == len(accepted),
        "generated_scene_count_matches_records": int(episode.get("generated_scene_count", len(generated_steps))) == len(generated_steps),
        "provisional_scenes_remain_unaccepted": all(
            not scene.get("quality_gate", {}).get("accepted")
            and bool(scene.get("diagnostics") or scene.get("validation_errors"))
            for scene in provisional
        ),
        "episode_policy_weights_are_present_and_well_formed": (
            isinstance(episode.get("policy_weights"), (list, tuple))
            and len(episode.get("policy_weights", ())) > 0
            and all(isinstance(value, (int, float)) for value in episode.get("policy_weights", ()))
        ),
        "selected_actions_have_valid_vqc_provenance": all(
            _has_valid_candidate_provenance(scene) for scene in accepted
        ),
        "assembled_scene_count_matches_unique_slots": (
            int(episode.get("assembled_scene_count", len(generated_steps))) == len(generated_steps)
        ),
        "failed_scenes_do_not_commit": int(episode.get("narrative_state_version", -1)) == len(accepted),
        "failure_blocks_dependents": all(
            int(index) not in accepted_steps for index in episode.get("blocked_scene_indices", ())
        ),
        "attempted_and_failed_counts_match_records": (
            int(episode.get("attempted_scene_count", len(generated_steps))) >= len(generated_steps)
            and int(episode.get("failed_scene_count", len(episode.get("scene_failures", ()))))
            == len(episode.get("scene_failures", ()))
        ),
        "failure_records_distinguish_provider_from_validation": all(
            failure.get("failure_type") in {"provider", "provider_fallback", "validation", "planning"}
            for failure in episode.get("scene_failures", ())
        ),
        "episode_status_matches_progress": episode.get("status") == expected_status,
        "contract_status_matches_scene_progress": all(
            item.get("status") in {"accepted", "provisional", "failed", "blocked", "pending"}
            for item in contracts
        ) and all(
            (item.get("status") == "accepted") == (int(item.get("sequence", 0)) - 1 in accepted_steps)
            for item in contracts
            if item.get("status") in {"accepted", "provisional", "blocked", "failed"}
        ),
        "provisional_contracts_match_generated_drafts": all(
            (item.get("status") == "provisional") == (
                int(item.get("sequence", 0)) - 1 in {
                    int(scene.get("step", -1)) for scene in provisional
                }
            )
            for item in contracts if item.get("status") in {"accepted", "provisional", "failed", "blocked"}
        ),
        "manuscript_has_no_internal_trace_markup": all(
            not re.search(r"\[(?:QUANTUM_TRACE|COHERENCE)\]", str(scene.get("story", "")), re.I)
            for scene in scenes
        ),
        "completion_requires_all_scenes_and_ending": (
            (episode.get("status") == "complete") == complete
        ),
        "generation_complete_requires_every_scene_entry": (
            bool(episode.get("generation_complete", len(generated_steps) == requested))
            == (len(generated_steps) == requested)
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
        "generated_scene_count": len(generated_steps),
        "provisional_scene_count": len(provisional),
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
