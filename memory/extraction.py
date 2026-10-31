"""Deterministic extraction of authoritative narrative candidates."""

from typing import Any, Mapping, Optional

from .models import MemoryRecord


def extract_authoritative_candidates(
    story_id: str,
    *,
    scene: Optional[Mapping[str, Any]] = None,
    action: Optional[Mapping[str, Any]] = None,
    event: Optional[Mapping[str, Any]] = None,
    episode_id: Optional[str] = None,
    scene_index: Optional[int] = None,
) -> tuple[MemoryRecord, ...]:
    """Extract only explicit scene/action/event facts; never infers or calls an LLM."""
    candidates = []

    def add(category: str, value: Any, origin: str):
        if isinstance(value, str) and value.strip():
            candidates.append(MemoryRecord(
                story_id=story_id, category=category, content=value.strip(),
                confidence=1.0, origin=origin, episode_id=episode_id,
                scene_index=scene_index,
            ))

    for key in ("facts", "authoritative_facts"):
        values = (scene or {}).get(key, ())
        if isinstance(values, str):
            values = (values,)
        if isinstance(values, (list, tuple)):
            for value in values:
                add("fact", value, "scene")
    for key in ("name", "type", "description"):
        add("action", (action or {}).get(key), "action")
    for key in ("type", "description", "outcome"):
        add("event", (event or {}).get(key), "event")

    unique = []
    seen = set()
    for record in candidates:
        key = (record.category, record.content.casefold())
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return tuple(unique)


extract_candidates = extract_authoritative_candidates
