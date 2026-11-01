"""Typed, bounded coordination contracts for accepted narrative state."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping
from threading import RLock


@dataclass(frozen=True)
class DirectorBrief:
    premise: str
    genre: str
    characters: dict[str, str]
    language: str = "English"
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CharacterState:
    character_id: str
    goal: str = ""
    motivation: str = ""
    beliefs: tuple[str, ...] = ()
    knowledge: tuple[str, ...] = ()
    emotional_state: str = "unestablished"
    relationships: Mapping[str, str] = field(default_factory=dict)
    condition: str = "unestablished"
    location: str = ""


@dataclass(frozen=True)
class NarrativeEvent:
    event_id: str
    scene_id: str
    sequence: int
    summary: str
    participants: tuple[str, ...] = ()
    location: str = ""
    time: str = ""
    facts: tuple[str, ...] = ()
    decisions: Mapping[str, str] = field(default_factory=dict)
    consequences: tuple[str, ...] = ()
    objects: tuple[Mapping[str, Any], ...] = ()
    character_effects: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveConflict:
    conflict_id: str
    description: str
    status: str = "active"
    stakes: str = ""


@dataclass(frozen=True)
class VersionedNarrativeState:
    story_id: str
    version: int = 0
    facts: tuple[str, ...] = ()
    character_states: Mapping[str, CharacterState] = field(default_factory=dict)
    locations: tuple[str, ...] = ()
    timeline: tuple[str, ...] = ()
    objects: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    events: tuple[NarrativeEvent, ...] = ()
    active_conflicts: tuple[ActiveConflict, ...] = ()
    open_threads: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentProposal:
    agent: str
    accepted: bool
    findings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevisionRequest:
    scene_id: str
    reasons: tuple[str, ...]
    revision_number: int
    max_revisions: int


@dataclass(frozen=True)
class AgentDisagreement:
    agents: tuple[str, ...]
    topic: str
    resolution: str


@dataclass(frozen=True)
class StateUpdate:
    story_id: str
    previous_version: int
    version: int
    event: NarrativeEvent
    added_facts: tuple[str, ...]
    changed_characters: tuple[str, ...]
    update_id: str


@dataclass(frozen=True)
class AgentTrace:
    agent: str
    provider: str = "local-contract"
    success: bool = True
    duration_ms: float = 0.0
    fallback: bool = False
    error: str | None = None


class DirectorAgent:
    name = "director"
    def parse(self, premise: str, genre: str, characters: dict[str, str], language: str = "English") -> DirectorBrief:
        if not premise.strip():
            raise ValueError("a story premise is required")
        return DirectorBrief(premise.strip(), genre, dict(characters), language)


class PlanningAgent:
    name = "planning"
    def propose(self, brief: DirectorBrief, scene_count: int = 8) -> dict[str, Any]:
        from planning import build_story_plan
        plan = build_story_plan(brief.premise, brief.genre, brief.characters, scene_count)
        return {"premise": plan.premise, "genre": plan.genre, "narrative_question": plan.narrative_question,
                "central_conflict": plan.central_conflict, "stakes": plan.stakes,
                "scene_ids": tuple(scene.scene_id for scene in plan.scenes),
                "purposes": tuple(scene.purpose for scene in plan.scenes)}


class ContinuityAgent:
    name = "continuity"
    def review(self, event: NarrativeEvent, state: VersionedNarrativeState) -> AgentProposal:
        findings = []
        if event.sequence != state.version + 1:
            previous_sequence = state.events[-1].sequence if state.events else 0
            if event.sequence <= previous_sequence:
                findings.append("event sequence is not later than the accepted story state")
        if not event.summary.strip():
            findings.append("scene event has no summary")
        if any(not value.strip() for value in event.facts):
            findings.append("scene contains an empty established fact")
        return AgentProposal(self.name, not findings, tuple(findings), evidence=event.facts)


class CharacterIntentAgent:
    name = "character_intent"
    def review(self, event: NarrativeEvent, state: VersionedNarrativeState) -> AgentProposal:
        referenced = (*event.participants, *event.character_effects.keys())
        unknown = tuple(name for name in dict.fromkeys(referenced) if name not in state.character_states)
        return AgentProposal(self.name, True, recommendations=tuple(
            f"Initialize state from premise for participant {name}." for name in unknown))


class ConflictProgressionAgent:
    name = "conflict_progression"
    def review(self, event: NarrativeEvent, previous: NarrativeEvent | None) -> AgentProposal:
        findings = () if previous is None or event.summary != previous.summary else ("scene repeats the prior event summary",)
        return AgentProposal(self.name, not findings, findings, recommendations=event.consequences)


class SceneQualityAgent:
    name = "scene_quality"
    def review(self, event: NarrativeEvent, plan: Mapping[str, Any]) -> AgentProposal:
        findings = []
        if not event.scene_id or event.scene_id != plan.get("scene_id"):
            findings.append("scene identity does not match the accepted plan")
        if not plan.get("purpose"):
            findings.append("planned scene purpose is missing")
        return AgentProposal(self.name, not findings, tuple(findings), evidence=(event.summary,))


@dataclass(frozen=True)
class CoordinationResult:
    accepted: bool
    proposals: tuple[AgentProposal, ...]
    disagreements: tuple[AgentDisagreement, ...]
    revision: RevisionRequest | None = None
    update: StateUpdate | None = None
    state: VersionedNarrativeState | None = None


class NarrativeCoordinator:
    """Agents can only propose. This object serializes review and state commits."""
    def __init__(self, story_id: str, characters: Mapping[str, str] | None = None,
                 max_revisions: int = 3, character_profiles: Mapping[str, Mapping[str, Any]] | None = None,
                 initial_state: Mapping[str, Any] | None = None):
        profiles = character_profiles or {}
        initial = initial_state or {}
        character_states = {}
        for role, name in (characters or {}).items():
            if not name:
                continue
            raw = profiles.get(name, profiles.get(role, {})) or {}
            if not isinstance(raw, Mapping):
                raise ValueError(f"character profile for {name!r} must be a mapping")
            character_states[name] = CharacterState(
                character_id=name,
                goal=str(raw.get("goal", f"Pursue the role established for {role}.") or ""),
                motivation=str(raw.get("motivation", "") or ""),
                beliefs=tuple(str(value) for value in raw.get("beliefs", ()) if str(value).strip()),
                knowledge=tuple(str(value) for value in raw.get("knowledge", ()) if str(value).strip()),
                emotional_state=str(raw.get("emotional_state", "unestablished") or "unestablished"),
                relationships=dict(raw.get("relationships", {}) or {}),
                condition=str(raw.get("condition", "unestablished") or "unestablished"),
                location=str(raw.get("location", "") or ""),
            )
        self.state = VersionedNarrativeState(
            story_id=story_id,
            version=0,
            facts=tuple(str(value) for value in initial.get("facts", ()) if str(value).strip()),
            character_states=character_states,
            locations=tuple(str(value) for value in initial.get("locations", ()) if str(value).strip()),
            objects=dict(initial.get("objects", {}) or {}),
            active_conflicts=tuple(
                item if isinstance(item, ActiveConflict) else (
                    ActiveConflict(f"conflict-{index + 1}", str(item))
                    if not isinstance(item, Mapping) else ActiveConflict(
                        str(item.get("conflict_id", f"conflict-{index + 1}")),
                        str(item.get("description", "")),
                        str(item.get("status", "active")),
                        str(item.get("stakes", "")),
                    )
                )
                for index, item in enumerate(initial.get("active_conflicts", ()))
            ),
            open_threads=tuple(str(value) for value in initial.get("open_threads", ()) if str(value).strip()),
        )
        self.max_revisions = max(0, int(max_revisions))
        self.agents = (ContinuityAgent(), CharacterIntentAgent(), ConflictProgressionAgent(), SceneQualityAgent())
        self.audit: list[StateUpdate] = []
        self._commit_lock = RLock()

    def review(self, event: NarrativeEvent, plan: Mapping[str, Any], revision_number: int = 0,
               commit: bool = False) -> CoordinationResult:
        if commit:
            raise RuntimeError("review is read-only; use NarrativeCoordinator.commit")
        previous = self.state.events[-1] if self.state.events else None
        proposals = (
            self.agents[0].review(event, self.state),
            self.agents[1].review(event, self.state),
            self.agents[2].review(event, previous),
            self.agents[3].review(event, plan),
        )
        disagreements = ()
        rejected = tuple(f"{p.agent}: {finding}" for p in proposals if not p.accepted for finding in p.findings)
        if rejected:
            revision = RevisionRequest(event.scene_id, rejected, revision_number, self.max_revisions)
            dissent = tuple(p.agent for p in proposals if p.accepted) + tuple(p.agent for p in proposals if not p.accepted)
            disagreements = (AgentDisagreement(dissent, "scene acceptance", "Coordinator applies fail-closed validation; rejected proposals require bounded revision."),)
            return CoordinationResult(False, proposals, disagreements, revision=revision)
        return CoordinationResult(True, proposals, disagreements)

    def _apply_reviewed_update(
        self, event: NarrativeEvent, reviewed: CoordinationResult
    ) -> CoordinationResult:
        # Coordinator is the only authority allowed to construct a next version.
        next_chars = dict(self.state.character_states)
        changed = []
        participants = tuple(dict.fromkeys((*event.decisions.keys(), *event.character_effects.keys())))
        for participant in participants:
            existing = next_chars.get(participant, CharacterState(participant))
            effect = event.character_effects.get(participant, {})
            if not isinstance(effect, Mapping):
                effect = {}
            next_chars[participant] = CharacterState(
                character_id=existing.character_id,
                goal=str(effect.get("goal", existing.goal)),
                motivation=str(effect.get("motivation", existing.motivation)),
                beliefs=tuple(dict.fromkeys((
                    *existing.beliefs, *(str(value) for value in effect.get("beliefs_added", ()))
                ))),
                knowledge=tuple(dict.fromkeys((
                    *existing.knowledge, *(str(value) for value in effect.get("knowledge_added", ()))
                ))),
                emotional_state=str(effect.get("emotional_state", existing.emotional_state)),
                relationships={**dict(existing.relationships), **dict(effect.get("relationships", {}) or {})},
                condition=str(effect.get("condition", existing.condition)),
                location=str(effect.get("location", existing.location)),
            )
            changed.append(participant)
        next_objects = dict(self.state.objects)
        for item in event.objects:
            object_id = str(item.get("id", "")).strip()
            if object_id:
                next_objects[object_id] = dict(item)
        state = VersionedNarrativeState(
            self.state.story_id, self.state.version + 1,
            tuple(dict.fromkeys((*self.state.facts, *event.facts))), next_chars,
            tuple(dict.fromkeys((*self.state.locations, *((event.location,) if event.location else ())))),
            (*self.state.timeline, event.time) if event.time else self.state.timeline,
            next_objects, (*self.state.events, event), self.state.active_conflicts,
            self.state.open_threads,
        )
        import hashlib
        update = StateUpdate(state.story_id, self.state.version, state.version, event, event.facts,
                             tuple(changed), hashlib.sha256(f"{state.story_id}:{state.version}:{event.event_id}".encode()).hexdigest()[:20])
        self.state = state
        self.audit.append(update)
        return CoordinationResult(
            True, reviewed.proposals, reviewed.disagreements, update=update, state=state
        )

    def commit(self, event: NarrativeEvent, plan: Mapping[str, Any],
               expected_version: int | None = None) -> CoordinationResult:
        """Revalidate and atomically commit after all narrative gates pass."""
        with self._commit_lock:
            if expected_version is not None and self.state.version != expected_version:
                raise RuntimeError(
                    f"stale narrative commit: expected version {expected_version}, current version {self.state.version}"
                )
            if event.event_id in {accepted.event_id for accepted in self.state.events}:
                raise RuntimeError(f"duplicate narrative event commit rejected: {event.event_id}")
            reviewed = self.review(event, plan)
            result = (
                self._apply_reviewed_update(event, reviewed)
                if reviewed.accepted else reviewed
            )
            if result.accepted and (result.update is None or result.state is None):
                raise RuntimeError("coordinator accepted a scene without a traceable state update")
            if result.accepted and result.update.previous_version != self.state.version - 1:
                raise RuntimeError("coordinator commit version changed during atomic update")
            return result
