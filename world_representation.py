"""Explicit narrative-event to structured-world updates and USDA export."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from agents.orchestration import NarrativeEvent, StateUpdate


def _usd_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _prim_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    return name if name and not name[0].isdigit() else f"Entity_{name or 'unknown'}"


@dataclass(frozen=True)
class WorldEntity:
    entity_id: str
    kind: str
    name: str
    location: str = ""
    owner: str = ""
    condition: str = "unknown"
    position: tuple[float, float, float] | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldState:
    story_id: str
    version: int = 0
    entities: Mapping[str, WorldEntity] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    committed_event_ids: tuple[str, ...] = ()

    @classmethod
    def from_narrative_state(cls, state) -> "WorldState":
        """Seed the world view from the coordinator's trusted story configuration."""
        entities = {}
        for name, profile in state.character_states.items():
            key = f"character:{name}"
            entities[key] = WorldEntity(
                key, "character", name, location=profile.location,
                condition=profile.condition,
                attributes={"goal": profile.goal, "knowledge": list(profile.knowledge),
                            "beliefs": list(profile.beliefs),
                            "relationships": dict(profile.relationships)},
            )
        for object_id, data in state.objects.items():
            key = str(object_id)
            entities[key] = WorldEntity(
                key, str(data.get("kind", "object")), str(data.get("name", key)),
                location=str(data.get("location", "")), owner=str(data.get("owner", "")),
                condition=str(data.get("condition", "unknown")),
                position=tuple(data["position"]) if data.get("position") is not None else None,
                attributes=dict(data.get("attributes", {})),
            )
        environment = {
            "locations": list(state.locations),
            "facts": list(state.facts),
            "open_threads": list(state.open_threads),
        }
        return cls(state.story_id, state.version, entities, environment)

    def apply_committed(self, event: NarrativeEvent, update: StateUpdate) -> "WorldState":
        if not isinstance(update, StateUpdate):
            raise ValueError("world changes require a coordinator-issued state update")
        if update.story_id != self.story_id or update.event.event_id != event.event_id:
            raise ValueError("world changes require the matching coordinator-committed event")
        if update.previous_version != self.version or update.version != self.version + 1:
            raise ValueError("world update version is stale or out of order")
        expected_id = sha256(f"{update.story_id}:{update.version}:{event.event_id}".encode()).hexdigest()[:20]
        if update.update_id != expected_id:
            raise ValueError("world update does not carry a valid coordinator trace ID")
        entities = dict(self.entities)
        for data in event.objects:
            key = str(data.get("id", "")).strip()
            if not key:
                raise ValueError("world objects require stable IDs")
            prior = entities.get(key)
            entities[key] = WorldEntity(
                key, str(data.get("kind", prior.kind if prior else "object")),
                str(data.get("name", prior.name if prior else key)),
                str(data.get("location", event.location or (prior.location if prior else ""))),
                str(data.get("owner", prior.owner if prior else "")),
                str(data.get("condition", prior.condition if prior else "unknown")),
                tuple(data["position"]) if data.get("position") is not None else (prior.position if prior else None),
                {**(prior.attributes if prior else {}), **dict(data.get("attributes", {}))},
            )
        for participant in event.participants:
            key = f"character:{participant}"
            if key not in entities:
                entities[key] = WorldEntity(key, "character", participant, event.location)
        return WorldState(self.story_id, update.version, entities, dict(self.environment),
                          (*self.committed_event_ids, event.event_id))

    def to_usda(self) -> str:
        lines = ["#usda 1.0", "(", '    defaultPrim = "World"', ")", 'def Xform "World" {',
                 f'    custom string storyId = {_usd_string(self.story_id)}',
                 f'    custom int version = {self.version}', '    def Scope "Entities" {']
        for entity in sorted(self.entities.values(), key=lambda item: item.entity_id):
            lines.extend([
                f'        def Xform "{_prim_name(entity.entity_id)}" {{',
                f'            custom string entityId = {_usd_string(entity.entity_id)}',
                f'            custom string kind = {_usd_string(entity.kind)}',
                f'            custom string name = {_usd_string(entity.name)}',
                f'            custom string location = {_usd_string(entity.location)}',
                f'            custom string owner = {_usd_string(entity.owner)}',
                f'            custom string condition = {_usd_string(entity.condition)}',
            ])
            if entity.position is not None:
                pos = ", ".join(str(float(v)) for v in entity.position)
                lines.append(f"            double3 xformOp:translate = ({pos})")
                lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append("        }")
        lines.extend(["    }", "}", ""])
        return "\n".join(lines)


class OmniverseAdapter:
    """Optional USD/Omniverse capability probe; never required for manuscript work."""
    @staticmethod
    def status() -> dict[str, Any]:
        try:
            import pxr  # type: ignore # noqa: F401
            usd_available = True
            error = None
        except ImportError as exc:
            usd_available = False
            error = str(exc)
        return {"available": usd_available, "usd_available": usd_available,
                "omniverse_available": False, "provider": "OpenUSD" if usd_available else None,
                "error": error, "text_only_core_available": True}
