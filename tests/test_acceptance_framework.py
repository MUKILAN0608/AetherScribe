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
        "scene_failures": [] if len(scenes) == 2 else [{"scene_index": 1}],
        "narrative_state_version": len(scenes),
        "ending_validation": {"valid": ending},
        "story_complete": status == "complete",
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
