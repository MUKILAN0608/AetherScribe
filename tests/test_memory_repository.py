import pytest

from evaluation.database import AetherDatabase, Episode
from memory import MemoryRecord, MemoryStatus, MemoryValidationError
from memory import extract_authoritative_candidates


def make_db():
    return AetherDatabase("sqlite:///:memory:")


def test_schema_is_idempotent_and_preserves_episode():
    db = make_db()
    episode_id = db.save_episode("mystery", "premise", 1.0, [])
    # create_all is intentionally safe to run repeatedly.
    from evaluation.database import Base
    Base.metadata.create_all(db.engine)
    assert db.get_recent_episodes()[0][0] == episode_id
    assert db.retrieve_memory("story").records == ()


def test_story_isolation_and_active_filter():
    db = make_db()
    first = db.add_memory(MemoryRecord("a", "fact", "same", origin="scene"))
    db.add_memory(MemoryRecord("b", "fact", "same", origin="scene"))
    db.memory.mark_conflicted([first.id])
    assert db.retrieve_memory("a").records == ()
    assert [r.story_id for r in db.retrieve_memory("b").records] == ["b"]


def test_deterministic_ranking_dedupe_and_budget():
    db = make_db()
    db.add_memory(MemoryRecord("s", "fact", "alpha clue", .5, scene_index=1))
    db.add_memory(MemoryRecord("s", "fact", "alpha clue", .9, scene_index=2))
    db.add_memory(MemoryRecord("s", "event", "unrelated", .99, scene_index=3))
    result = db.retrieve_memory("s", query="alpha", budget=10)
    assert [r.content for r in result.records] == ["alpha clue"]
    assert result.diagnostics.used <= 10


def test_provenance_and_supersession():
    db = make_db()
    old = db.add_memory(MemoryRecord("s", "fact", "old", origin="episode", episode_id="e1", scene_index=2))
    new = db.memory.supersede(old.id, MemoryRecord("s", "fact", "new", origin="correction"))
    assert new.supersedes_id == old.id
    assert new.origin == "correction"
    assert [r.content for r in db.retrieve_memory("s").records] == ["new"]


def test_validation_rejects_unsafe_records_without_writing():
    db = make_db()
    with pytest.raises(MemoryValidationError):
        db.add_memory(MemoryRecord("s", "fact", "", confidence=2))


def test_episode_story_id_is_optional_and_persisted():
    db = make_db()
    legacy_id = db.save_episode("mystery", "legacy", 1.0, [])
    current_id = db.save_episode("mystery", "current", 1.0, [], story_id="story-1")
    rows = db.get_recent_episodes()
    assert {row[0] for row in rows} == {legacy_id, current_id}
    session = db.Session()
    try:
        assert session.query(Episode).filter_by(id=current_id).one().story_id == "story-1"
    finally:
        session.close()


def test_authoritative_candidate_extraction_is_deterministic_and_whitelisted():
    records = extract_authoritative_candidates(
        "story-1",
        scene={"facts": ["The door is open", "The door is open", 42], "mood": "ignored"},
        action={"name": "open door", "internal_reasoning": "ignored"},
        event={"type": "door_opened", "outcome": "access granted"},
        episode_id="ep-1",
        scene_index=3,
    )
    assert [(r.category, r.content, r.origin) for r in records] == [
        ("fact", "The door is open", "scene"),
        ("action", "open door", "action"),
        ("event", "door_opened", "event"),
        ("event", "access granted", "event"),
    ]
