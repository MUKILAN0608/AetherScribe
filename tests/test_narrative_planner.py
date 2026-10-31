import pytest

from memory import MemoryRecord
from planning import (
    CharacterKnowledge,
    LockedActionSnapshot,
    MemoryReference,
    NarrativePlanner,
    PlannerInput,
    PlannerResult,
    PlannerValidationError,
    build_story_plan,
)


def make_input():
    return PlannerInput(
        "story-a", "episode-1", 2, {"tension": 0.8, "phase": "rising"},
        (MemoryReference("m1", "story-a", "fact", "private secret", True),),
        (CharacterKnowledge("hero", ("m1",), ("Hero cannot know m1 yet.",)),),
        LockedActionSnapshot({"hero": "investigate"}),
        400, 7,
    )


def test_valid_output_is_advisory_and_validated():
    result = NarrativePlanner().plan(make_input())
    assert result.validation_status == "validated"
    assert result.relevant_memory_ids == ("m1",)
    assert "private secret" not in repr(result)
    assert not hasattr(result, "replacement_action")


def test_story_isolation_and_malformed_data_are_rejected():
    with pytest.raises(PlannerValidationError):
        PlannerInput("story-a", "ep", 0, {}, (MemoryReference("m", "story-b"),))
    with pytest.raises(PlannerValidationError):
        PlannerInput("story-a", "ep", 0, {}, (object(),))


def test_secret_preservation_and_locked_action_immutability():
    action = {"hero": "investigate"}
    planner_input = make_input()
    before = dict(planner_input.locked_action.actions)
    action["hero"] = "replace"
    NarrativePlanner().plan(planner_input)
    assert dict(planner_input.locked_action.actions) == before
    with pytest.raises(TypeError):
        planner_input.locked_action.actions["hero"] = "replace"


def test_disabled_flag_returns_status_without_advisory_content(monkeypatch):
    monkeypatch.setenv("NARRATIVE_PLANNER_ENABLED", "false")
    result = NarrativePlanner().plan(make_input())
    assert result.validation_status == "disabled"
    assert result.relevant_memory_ids == ()


def test_result_rejects_contradictory_memory_reference():
    result = PlannerResult(
        "objective", "steady", .5, ("outside",), (), (), (), ("guidance",),
        "Do not reveal secrets.", .5, "validated", "local", "test", 0,
    )
    with pytest.raises(PlannerValidationError):
        NarrativePlanner().validate_result(result, make_input())


@pytest.mark.parametrize("scene_count", [6, 8, 10, 12])
def test_story_plan_has_exact_count_and_unique_scene_purposes(scene_count):
    plan = build_story_plan(
        "Mira investigates a vault while Voss obstructs her.",
        "thriller",
        {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
        scene_count,
    )
    assert len(plan.scenes) == scene_count
    assert [scene.scene_index for scene in plan.scenes] == list(range(scene_count))
    assert len({scene.purpose for scene in plan.scenes}) == scene_count
    assert plan.scenes[0].phase == "opening"
    assert plan.scenes[-1].phase == "resolution"


@pytest.mark.parametrize("genre,premise", [
    ("mystery", "A cartographer finds a map that changes after midnight."),
    ("fantasy", "A healer must choose whether to break an ancient oath."),
    ("science fiction", "A pilot wakes alone aboard a ship that remembers its crew."),
    ("romance", "Two rivals inherit a seaside bookshop and must decide its future."),
])
def test_story_plan_is_genre_neutral_and_has_stable_scene_contract(genre, premise):
    plan = build_story_plan(premise, genre, {"protagonist": "Ari"}, 8, ("Magic costs a memory.",))
    assert len(plan.scenes) == 8
    assert [scene.scene_id for scene in plan.scenes] == [f"scene-{i:04d}" for i in range(1, 9)]
    assert len({scene.title for scene in plan.scenes}) == 8
    assert all(scene.pov_character == "Ari" for scene in plan.scenes)
    assert all(scene.story_change and scene.causal_connection for scene in plan.scenes)
    assert all(not scene.validation_terms for scene in plan.scenes)
    assert "maintenance ledger" not in " ".join(scene.event for scene in plan.scenes)
    assert plan.narrative_question
    assert plan.central_conflict
    assert plan.stakes
    assert plan.genre == genre
    assert plan.world_rules == ("Magic costs a memory.",)


def test_plan_purposes_are_unique_for_each_slot():
    plan = build_story_plan("A courier discovers a message meant for tomorrow.", "science fiction", {"protagonist": "Noor"}, 12)
    assert len({scene.purpose for scene in plan.scenes}) == 12
    assert len({scene.scene_id for scene in plan.scenes}) == 12
