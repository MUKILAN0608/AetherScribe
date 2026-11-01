import numpy as np
import pytest

from planning import build_story_plan
from quantum_policy.quantum_action_space import build_scene_action_candidates
from quantum_policy.quantum_policy import QuantumPolicy


@pytest.mark.parametrize(
    "genre,premise,characters",
    [
        ("mystery", "An archivist traces a missing letter through sealed records.",
         {"protagonist": "Ari", "antagonist": "Rook", "ally": "Sol"}),
        ("fantasy", "A healer bargains with a river spirit to restore a village well.",
         {"protagonist": "Tala", "antagonist": "The Spirit", "ally": "Iven"}),
        ("romance", "Two rivals restore a theater while reconsidering their partnership.",
         {"protagonist": "June", "antagonist": "Eli", "ally": "Ren"}),
    ],
)
def test_scene_candidates_are_derived_from_story_contract(genre, premise, characters):
    plan = build_story_plan(premise, genre, characters, scene_count=4)
    candidates = build_scene_action_candidates(
        plan.for_scene(1), characters, scene_index=1, accepted_predecessor=True, tension=.6,
    )
    assert len(candidates) == 16
    assert all(item.valid for item in candidates)
    assert len({tuple(item.actions.items()) for item in candidates}) > 1
    assert all(item.expected_effect == plan.for_scene(1).consequence for item in candidates)
    assert all(item.action_id.startswith(plan.for_scene(1).scene_id) for item in candidates)
    assert all(set(item.actions).issubset({"Protagonist", "Antagonist", "Ally"}) for item in candidates)


def test_precondition_filter_rejects_candidates_before_vqc_selection():
    characters = {"protagonist": "Ari", "antagonist": "Rook", "ally": "Sol"}
    plan = build_story_plan("Ari investigates a sealed archive.", "mystery", characters, 2)
    candidates = build_scene_action_candidates(
        plan.for_scene(1), characters, scene_index=1, accepted_predecessor=False,
    )
    assert not any(candidate.valid for candidate in candidates)
    with pytest.raises(ValueError, match="no narrative-valid"):
        QuantumPolicy.condition_on_valid_actions(np.full(16, 1 / 16), [c.valid for c in candidates])


def test_precondition_checker_filters_individual_contract_candidates():
    characters = {"protagonist": "Ari", "antagonist": "Rook", "ally": "Sol"}
    plan = build_story_plan("Ari investigates a sealed archive.", "mystery", characters, 2)
    candidates = build_scene_action_candidates(
        plan.for_scene(0), characters, scene_index=0, accepted_predecessor=True,
        precondition_check=lambda actions, _plan, _state: (
            ("ally cannot act without a known supporting character",)
            if actions.get("Ally", "").startswith("manages the risk") else ()
        ),
        canonical_state={"version": 0},
    )
    assert any(candidate.valid for candidate in candidates)
    assert any(not candidate.valid for candidate in candidates)
    assert all(candidate.invalid_reasons for candidate in candidates if not candidate.valid)
    posterior = QuantumPolicy.condition_on_valid_actions(
        np.full(16, 1 / 16), [candidate.valid for candidate in candidates]
    )
    assert np.all(posterior[[not candidate.valid for candidate in candidates]] == 0)


def test_vqc_posterior_is_conditioned_on_valid_candidates():
    conditioned = QuantumPolicy.condition_on_valid_actions(
        np.asarray([.1, .2, .3, .4]), [True, False, True, False]
    )
    assert conditioned.tolist() == pytest.approx([.25, 0, .75, 0])
    assert conditioned.sum() == pytest.approx(1.0)


def test_candidate_count_adapts_to_available_cast():
    plan = build_story_plan("Ari studies an impossible letter.", "mystery", {"protagonist": "Ari"}, 2)
    candidates = build_scene_action_candidates(
        plan.for_scene(0), {"protagonist": "Ari"}, scene_index=0, accepted_predecessor=True,
    )
    assert len(candidates) == 2
    assert all(tuple(candidate.actions) == ("Protagonist",) for candidate in candidates)
