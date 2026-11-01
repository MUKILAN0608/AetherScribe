from evaluation.acceptance import evaluate_episode_acceptance


def _episode(status="complete", steps=(0, 1), ending=True):
    def make_scene(step):
        candidate_id = f"scene-{step}:candidate"
        return {
            "step": step,
            "quality_gate": {"accepted": True},
            "story": f"Scene {step}.",
            "action_candidate": {"action_id": candidate_id, "valid": True},
            "candidate_selection": {
                "candidate_ids": [candidate_id], "candidate_validity": [True],
                "valid_conditioned_posterior": [1.0], "selected_candidate_id": candidate_id,
            },
        }
    scenes = [
        make_scene(step)
        for step in steps
    ]
    return {
        "requested_scene_count": 2,
        "attempted_scene_count": 2,
        "accepted_scene_count": len(scenes),
        "failed_scene_count": 0,
        "blocked_scene_indices": [],
        "scenes": scenes,
        "scene_contracts": [
            {"sequence": 1, "status": "accepted"},
            {"sequence": 2, "status": "accepted" if len(scenes) == 2 else "blocked"},
        ],
        "story_contract": {"required_scene_count": 2},
        "scene_blueprints": [{"sequence": 1}, {"sequence": 2}],
        "scene_failures": [] if len(scenes) == 2 else [{"scene_index": 1, "failure_type": "validation"}],
        "narrative_state_version": len(scenes),
        "ending_validation": {"valid": ending},
        "whole_story_audit": {"passed": ending},
        "story_complete": status == "complete",
        "policy_weights": [0.1, 0.2],
        "status": status,
    }


def test_acceptance_is_plot_agnostic_and_checks_complete_episode_invariants():
    for genre, premise, characters in (
        ("mystery", "An archivist traces a missing letter.", {"protagonist": "Ari"}),
        ("fantasy", "A healer bargains with a river spirit.", {"protagonist": "Tala"}),
        ("comedy", "Two neighbors swap identical parcels.", {"protagonist": "Jo"}),
    ):
        episode = _episode()
        episode.update(genre=genre, premise=premise, characters=characters)
        assert evaluate_episode_acceptance(episode)["passed"]


def test_acceptance_recognizes_well_labeled_partial_episode():
    episode = _episode(status="partial", steps=(0,), ending=False)
    episode["blocked_scene_indices"] = [1]
    episode["failed_scene_count"] = 1
    result = evaluate_episode_acceptance(episode)
    assert result["passed"]
    episode["narrative_state_version"] = 2  # failed draft must not advance canonical state
    assert "failed_scenes_do_not_commit" in evaluate_episode_acceptance(episode)["failed_checks"]


def test_generated_status_is_separate_from_scene_validation_and_provisional_is_not_accepted():
    episode = _episode(status="generated")
    episode["scenes"][1]["quality_gate"] = {"accepted": False, "issues": ["repetition"]}
    episode["scenes"][1]["generation_status"] = "generated_provisional"
    episode["scenes"][1]["diagnostics"] = {"issues": ["repetition"]}
    episode["accepted_scene_count"] = 1
    episode["failed_scene_count"] = 1
    episode["scene_failures"] = [{"scene_index": 1, "failure_type": "validation", "issues": ["repetition"]}]
    episode["scene_contracts"][1]["status"] = "provisional"
    episode["narrative_state_version"] = 1
    episode["generated_scene_count"] = 2
    episode["assembled_scene_count"] = 2
    episode["provisional_scene_count"] = 1
    episode["generation_complete"] = True
    episode["validation_complete"] = False
    episode["ending_validation"] = {"valid": False}
    episode["whole_story_audit"] = {"passed": False}
    episode["story_complete"] = False

    result = evaluate_episode_acceptance(episode)
    assert result["passed"]
    assert result["status"] == "generated"
    assert result["generated_scene_count"] == 2
    assert result["accepted_scene_count"] == 1
    assert result["provisional_scene_count"] == 1
