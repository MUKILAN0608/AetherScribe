from io import BytesIO

import pytest
from PIL import Image

from agents.orchestration import NarrativeCoordinator, NarrativeEvent, StateUpdate
from future_prediction import FutureWorldPredictor
from temporal_adaptation import TemporalCharacterAdapter
from visual_observation import VisualObserver
from world_representation import WorldState
from evaluation.database import AetherDatabase


def _event(sequence=1, summary="Mira places the lantern on the table.", objects=()):
    return NarrativeEvent(
        event_id=f"e{sequence}", scene_id=f"scene-{sequence:04d}", sequence=sequence,
        summary=summary, participants=("Mira",), location="Workshop", time="Evening",
        facts=(summary,), decisions={"Mira": "protects the lantern"},
        consequences=("The lantern remains visible.",), objects=objects,
    )


def test_coordinator_reviews_without_mutating_and_commits_versioned_update():
    coordinator = NarrativeCoordinator("story-a", {"protagonist": "Mira"}, max_revisions=2)
    event = _event(objects=({"id": "lantern", "kind": "object", "condition": "on table"},))
    plan = {"scene_id": event.scene_id, "purpose": "Protect the lantern."}
    reviewed = coordinator.review(event, plan)
    assert reviewed.accepted
    assert coordinator.state.version == 0
    committed = coordinator.commit(event, plan)
    assert committed.state.version == 1
    assert committed.update.previous_version == 0
    assert coordinator.audit == [committed.update]


def test_canonical_state_seeds_character_profiles_and_applies_only_explicit_knowledge_effects():
    coordinator = NarrativeCoordinator(
        "story-state",
        {"protagonist": "Mira", "antagonist": "Voss"},
        character_profiles={
            "Mira": {"goal": "Recover the index", "knowledge": ["The seal was altered"],
                     "location": "Archive", "relationships": {"Voss": "distrust"}},
        },
        initial_state={
            "facts": ["The archive is sealed"],
            "locations": ["Archive"],
            "objects": {"index": {"name": "Index", "condition": "sealed"}},
            "active_conflicts": ["The records are inaccessible"],
            "open_threads": ["Who altered the index?"],
        },
    )
    assert coordinator.state.character_states["Mira"].knowledge == ("The seal was altered",)
    assert coordinator.state.facts == ("The archive is sealed",)
    assert coordinator.state.open_threads == ("Who altered the index?",)
    event = NarrativeEvent(**{
        **_event().__dict__,
        "facts": ("Mira opens the seal",),
        "character_effects": {"Mira": {"knowledge_added": ("The index was copied",), "location": "Archive"}},
    })
    coordinator.commit(event, {"scene_id": event.scene_id, "purpose": "Recover the index."}, expected_version=0)
    assert "The index was copied" in coordinator.state.character_states["Mira"].knowledge
    assert "Mira opens the seal" not in coordinator.state.character_states["Voss"].knowledge
    assert "Mira opens the seal" in coordinator.state.facts


def test_coordinator_rejects_duplicate_and_returns_bounded_revision():
    coordinator = NarrativeCoordinator("story-a", {"protagonist": "Mira"}, max_revisions=1)
    plan = {"scene_id": "scene-0001", "purpose": "Act."}
    first = _event()
    coordinator.commit(first, plan)
    duplicate = _event(2)
    duplicate = NarrativeEvent(**{**duplicate.__dict__, "scene_id": first.scene_id})
    result = coordinator.review(duplicate, {"scene_id": first.scene_id, "purpose": "Act."}, revision_number=1)
    assert not result.accepted
    assert result.revision.max_revisions == 1
    assert result.disagreements
    assert coordinator.state.version == 1


def test_coordinator_commit_is_idempotency_guarded_and_version_checked():
    coordinator = NarrativeCoordinator("story-a", {"protagonist": "Mira"})
    event = _event()
    plan = {"scene_id": event.scene_id, "purpose": "Protect the lantern."}
    committed = coordinator.commit(event, plan, expected_version=0)
    assert committed.state.version == 1
    with pytest.raises(RuntimeError, match="duplicate"):
        coordinator.commit(event, plan, expected_version=1)
    with pytest.raises(RuntimeError, match="stale"):
        coordinator.commit(_event(2), {"scene_id": "scene-0002", "purpose": "Continue."}, expected_version=0)
    assert coordinator.state.version == 1


def test_world_state_requires_coordinator_commit_and_exports_usda():
    coordinator = NarrativeCoordinator("story-a", {"protagonist": "Mira"})
    event = _event(objects=({"id": "lantern", "kind": "object", "name": "Lantern", "condition": "lit", "position": (1, 2, 0)},))
    plan = {"scene_id": event.scene_id, "purpose": "Act."}
    review = coordinator.review(event, plan)
    assert review.update is None
    with pytest.raises(RuntimeError, match="read-only"):
        coordinator.review(event, plan, commit=True)
    with pytest.raises(ValueError):
        WorldState("story-a").apply_committed(event, None)
    with pytest.raises(ValueError):
        WorldState("story-a").apply_committed(
            event, StateUpdate("story-a", 0, 1, event, (), (), "forged")
        )
    result = coordinator.commit(event, plan)
    world = WorldState("story-a").apply_committed(event, result.update)
    usda = world.to_usda()
    assert world.version == 1
    assert 'def Xform "World"' in usda
    assert 'custom string condition = "lit"' in usda
    assert 'custom string status = ' not in usda


def test_prediction_is_hypothetical_and_does_not_mutate_state():
    state = {"version": 4, "characters": {"Mira": {"location": "Workshop"}}}
    report = FutureWorldPredictor().predict(state, ["investigate archive", "protect witness"], horizon=2)
    assert len(report.futures) == 2
    assert all(future.status == "prediction" for future in report.futures)
    assert all(future.plausibility is None for future in report.futures)
    assert report.method == "transparent-rule-based"
    assert not report.committed
    assert report.authoritative_action is None
    assert state["version"] == 4
    assert "information" in report.futures[0].predicted_state["hypothesized_effect"]


def _png() -> bytes:
    image = Image.new("RGB", (4, 4), color="white")
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_visual_observer_detects_mismatch_without_mutating_expected_state():
    expected = {"flashlight": {"condition": "held by Mira"}}
    before = repr(expected)
    report = VisualObserver().observe(
        _png(), expected,
        detector=lambda _image: [{"entity_id": "flashlight", "status": "on floor", "confidence": .97, "evidence": "bbox 2,2,3,3"}],
    )
    assert report.status == "complete"
    assert report.discrepancies == ("flashlight",)
    assert report.findings[0].confidence == .97
    assert not report.canonical_state_modified
    assert repr(expected) == before


def test_visual_observation_reports_missing_detector_honestly():
    report = VisualObserver().observe(_png(), {"door": "open"})
    assert report.status == "unavailable"
    assert "not inferred" in report.error
    assert not report.canonical_state_modified


def test_character_adaptation_requires_event_evidence_and_experiment_changes_decision():
    supported = NarrativeEvent(**{
        **_event().__dict__,
        "event_id": "betrayal",
        "facts": ("Mira discovers the council sold her route.",),
        "character_effects": {"Mira": {"resolve_delta": .25, "support_fact": "Mira discovers the council sold her route.", "emotional_state": "wary"}},
    })
    unsupported = NarrativeEvent(**{
        **_event().__dict__, "event_id": "empty", "facts": (),
        "character_effects": {"Mira": {"resolve_delta": .25, "support_fact": "not in the event"}},
    })
    adapter = TemporalCharacterAdapter()
    unchanged = adapter.apply_committed_event(unsupported)[0]
    assert unchanged.after.resolve == unchanged.before.resolve
    experiment = TemporalCharacterAdapter.controlled_experiment(
        (), (supported,), lambda profile: "confront" if profile.resolve >= .7 else "wait", "Mira",
    )
    assert experiment["decision_changed"]
    assert experiment["resolve_delta"] > 0
    assert experiment["grounded"]


def test_episode_persists_stage_state_metadata(tmp_path):
    db = AetherDatabase(f"sqlite:///{tmp_path / 'stage-state.db'}")
    episode_id = db.save_episode(
        "fantasy", "A healer makes a choice.", 0.0, [], story_id="story-a",
        system_state={"state_version": 2, "world_usda": "#usda 1.0", "status": "partial"},
    )
    assert db.get_episode_system_state(episode_id) == {
        "state_version": 2, "world_usda": "#usda 1.0", "status": "partial"
    }
