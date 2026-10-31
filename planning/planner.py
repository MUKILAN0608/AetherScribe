"""Stage 3 advisory narrative planner.

The planner advises a renderer; it does not select, amend, or return an action.
All decisions are local and deterministic so this module is safe to use offline.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from memory.models import MemoryRecord


class PlannerError(ValueError):
    """Base error for invalid planner input or output."""


class PlannerValidationError(PlannerError):
    """Raised when planner input/output violates the narrative contract."""


@dataclass(frozen=True)
class ScenePlan:
    """Deterministic arc slot used to keep each requested scene purposeful."""

    scene_index: int
    phase: str
    purpose: str
    event: str
    consequence: str
    unresolved_objective: str
    protagonist_objective: str = ""
    antagonist_opposition: str = ""
    ally_contribution: str = ""
    new_information: str = ""
    next_scene_effect: str = ""
    validation_terms: tuple[str, ...] = ()
    scene_id: str = ""
    title: str = ""
    pov_character: str = ""
    location: str = "Established setting"
    time: str = "Continuous with prior events"
    character_interaction: str = ""
    story_change: str = ""
    causal_connection: str = ""


@dataclass(frozen=True)
class StoryPlan:
    premise: str
    scenes: tuple[ScenePlan, ...]
    genre: str = "fiction"
    narrative_question: str = ""
    central_conflict: str = ""
    stakes: str = ""
    character_goals: Mapping[str, str] = field(default_factory=dict)
    world_rules: tuple[str, ...] = ()
    open_threads: tuple[str, ...] = ()
    turning_points: tuple[str, ...] = ()

    def for_scene(self, scene_index: int) -> ScenePlan:
        if not self.scenes:
            raise PlannerValidationError("story plan has no scenes")
        return self.scenes[min(max(scene_index, 0), len(self.scenes) - 1)]


def build_story_plan(premise: str, genre: str, characters: Mapping[str, str], scene_count: int,
                     world_rules: Sequence[str] = ()) -> StoryPlan:
    """Build a bounded premise-aware arc plan without making model output authoritative."""
    if scene_count <= 0:
        raise PlannerValidationError("scene_count must be positive")
    protagonist = characters.get("protagonist", "the protagonist")
    antagonist = characters.get("antagonist", "the opposing force")
    ally = characters.get("ally", "the ally")
    stages = ("opening", "development", "escalation", "climax", "resolution")
    # Genre-neutral arc: each slot changes the protagonist's options and carries
    # a consequence into the following scene. Story-specific evidence is never
    # invented by the planner; it comes from the user premise or model.
    scenes = []
    development_beats = (
        ("secure a first commitment", "A reluctant person or resource must be persuaded to participate."),
        ("test the current belief", "An attempt to verify the protagonist's working belief produces a concrete result."),
        ("pay for an earlier choice", "A cost caused by an earlier decision arrives at an inconvenient moment."),
        ("change an important relationship", "A character's action shifts trust, loyalty, or emotional distance."),
        ("give up a safe fallback", "The protagonist voluntarily loses a comfortable option to preserve the goal."),
        ("reframe the central problem", "A consequence forces the protagonist to understand the central problem differently."),
    )
    for index in range(scene_count):
        progress = index / max(scene_count - 1, 1)
        if index == 0:
            phase = "opening"
            purpose = f"Establish {protagonist}'s concrete goal, personal stake, and the premise's immediate disruption."
            event = f"{protagonist} makes an irreversible first move in response to the central situation."
            consequence = "That choice creates a cost or obligation that changes the next available move."
        elif index == scene_count - 1:
            phase = "resolution"
            purpose = f"Resolve or intentionally transform the central conflict through {protagonist}'s accumulated choices."
            event = f"{protagonist} makes the decisive choice and faces its established consequences."
            consequence = "The central question receives a meaningful outcome and the characters' situation is changed."
        elif progress < 0.4:
            phase = "development"
            purpose = f"Complicate {protagonist}'s first strategy with a consequence caused by the previous scene."
            event = f"A new obstacle forces {protagonist} to revise the approach while protecting the original goal."
            consequence = "The revised approach exposes a new cost and narrows the remaining options."
        elif progress < 0.72:
            phase = "escalation"
            purpose = f"Reveal the cost or hidden implication of the current approach and force a consequential decision."
            event = f"{protagonist} chooses between competing priorities under pressure from the established conflict."
            consequence = "The choice changes an important relationship, resource, belief, or immediate risk."
        else:
            phase = "climax"
            purpose = f"Bring the central conflict to its decisive confrontation using consequences already established."
            event = f"{protagonist} acts on the strongest accumulated consequence; opposition answers in kind."
            consequence = "The confrontation determines what can be resolved in the final scene."
        beat_name, beat_event = development_beats[(index - 1) % len(development_beats)]
        if 0 < index < scene_count - 1 and phase in {"development", "escalation"}:
            purpose = f"{purpose} Specifically, {beat_name}."
            event = f"{event} {beat_event}"
        if antagonist and antagonist != "the opposing force":
            opposition = f"{antagonist} or the premise's opposing force resists the current objective in a way grounded in prior events."
        else:
            opposition = "The premise's established forces, circumstances, or the protagonist's own conflict obstruct the objective."
        ally_work = (
            f"{ally} contributes only through established abilities, knowledge, or relationship stakes."
            if ally and ally != "the ally" else "A supporting character or established resource changes the available choices if the premise supports it."
        )
        next_effect = consequence if index + 1 < scene_count else "Close the central question and show the lasting consequence."
        scenes.append(ScenePlan(
            index, phase, f"Scene {index + 1}: {purpose}", event, consequence,
            f"Make the next scene necessary through this consequence.",
            protagonist_objective=purpose,
            antagonist_opposition=opposition,
            ally_contribution=ally_work,
            new_information="Introduce only a discovery, decision, or emotional change supported by the user's premise and events.",
            next_scene_effect=next_effect,
            scene_id=f"scene-{index + 1:04d}",
            title=(f"Opening choice · {index + 1}" if phase == "opening" else
                   f"Earned resolution · {index + 1}" if phase == "resolution" else
                   f"{phase.title()} · {development_beats[(index - 1) % len(development_beats)][0] if index else 'central turn'} · {index + 1}"),
            pov_character=protagonist,
            character_interaction=f"Let established characters reveal their priorities through action and exchange.",
            story_change=consequence,
            causal_connection=("The premise creates the opening pressure." if index == 0 else
                               "The scene follows from the accepted consequence of the preceding scene."),
        ))
    clean_premise = premise.strip()
    opposition = characters.get("antagonist", "the central obstacle")
    return StoryPlan(
        clean_premise,
        tuple(scenes),
        genre=genre.strip() or "fiction",
        narrative_question=(f"Can {protagonist} respond to the central situation without losing what matters most?"),
        central_conflict=f"{protagonist}'s goal conflicts with {opposition} and the consequences of the premise.",
        stakes=f"The outcome changes {protagonist}'s future and the relationships or world described in the premise.",
        character_goals={key: f"Act according to the goal and role established for {name}."
                         for key, name in characters.items() if name},
        world_rules=tuple(str(rule).strip() for rule in world_rules if str(rule).strip()),
        open_threads=("Resolve or intentionally carry forward the central question raised by the premise.",),
        turning_points=tuple(scene.purpose for scene in scenes if scene.phase in {"escalation", "climax", "resolution"}),
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class MemoryReference:
    """A safe reference to memory; content is intentionally optional."""

    memory_id: str
    story_id: str
    category: str = "unknown"
    content: Optional[str] = None
    secret: bool = False

    def validate(self, story_id: str) -> "MemoryReference":
        if (not isinstance(self.memory_id, str) or not self.memory_id.strip()
                or not isinstance(self.story_id, str) or not self.story_id.strip()
                or self.story_id != story_id):
            raise PlannerValidationError("memory reference is outside the story scope")
        if (not isinstance(self.category, str) or not self.category.strip()
                or (self.content is not None and not isinstance(self.content, str))):
            raise PlannerValidationError("malformed memory reference")
        return self


@dataclass(frozen=True)
class CharacterKnowledge:
    character_id: str
    known_memory_ids: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.character_id.strip():
            raise PlannerValidationError("character_id is required")
        if any(not isinstance(item, str) or not item.strip() for item in self.known_memory_ids):
            raise PlannerValidationError("known_memory_ids must contain non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.constraints):
            raise PlannerValidationError("constraints must contain non-empty strings")


@dataclass(frozen=True)
class LockedActionSnapshot:
    """Immutable action chosen by the decision stage."""

    actions: Mapping[str, str]

    def __post_init__(self) -> None:
        frozen = _freeze(self.actions)
        if not isinstance(frozen, MappingProxyType) or not frozen:
            raise PlannerValidationError("locked action must be a non-empty mapping")
        if any(not key.strip() or not isinstance(value, str) or not value.strip()
               for key, value in frozen.items()):
            raise PlannerValidationError("locked action contains malformed role/action")
        object.__setattr__(self, "actions", frozen)


@dataclass(frozen=True)
class PlannerInput:
    story_id: str
    episode_id: str
    scene_index: int
    current_state: Mapping[str, Any]
    memories: tuple[MemoryRecord | MemoryReference, ...]
    character_knowledge: tuple[CharacterKnowledge, ...] = ()
    locked_action: LockedActionSnapshot = field(default_factory=lambda: LockedActionSnapshot({"scene": "observe"}))
    token_budget: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.story_id, str) or not self.story_id.strip():
            raise PlannerValidationError("story_id is required")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise PlannerValidationError("episode_id is required")
        if not isinstance(self.scene_index, int) or self.scene_index < 0:
            raise PlannerValidationError("scene_index must be a non-negative integer")
        if not isinstance(self.token_budget, int) or self.token_budget <= 0:
            raise PlannerValidationError("token_budget must be positive")
        if not isinstance(self.seed, int):
            raise PlannerValidationError("seed must be an integer")
        if not isinstance(self.current_state, Mapping):
            raise PlannerValidationError("current_state must be a mapping")
        object.__setattr__(self, "current_state", _freeze(self.current_state))
        object.__setattr__(self, "memories", tuple(self.memories))
        object.__setattr__(self, "character_knowledge", tuple(self.character_knowledge))
        if not isinstance(self.locked_action, LockedActionSnapshot):
            raise PlannerValidationError("locked_action must be a LockedActionSnapshot")
        if any(not isinstance(item, CharacterKnowledge) for item in self.character_knowledge):
            raise PlannerValidationError("character_knowledge contains an invalid entry")
        seen: set[str] = set()
        for memory in self.memories:
            if isinstance(memory, MemoryRecord):
                try:
                    memory.validate()
                except (ValueError, TypeError) as exc:
                    raise PlannerValidationError("malformed memory record") from exc
                if memory.story_id != self.story_id:
                    raise PlannerValidationError("memory record is outside the story scope")
                memory_id = str(memory.id) if memory.id is not None else f"{memory.category}:{memory.content}"
            elif isinstance(memory, MemoryReference):
                memory.validate(self.story_id)
                memory_id = memory.memory_id
            else:
                raise PlannerValidationError("memories must be validated records or references")
            if memory_id in seen:
                raise PlannerValidationError("duplicate memory reference")
            seen.add(memory_id)


@dataclass(frozen=True)
class PlannerResult:
    scene_objective: str
    pacing: str
    tension: float
    relevant_memory_ids: tuple[str, ...]
    knowledge_constraints: tuple[str, ...]
    open_threads: tuple[str, ...]
    continuity_risks: tuple[str, ...]
    rendering_guidance: tuple[str, ...]
    secret_reveal_guidance: str
    confidence: float
    validation_status: str
    provider: str
    model: str
    latency_ms: float
    fallback: bool = False
    error: Optional[str] = None

    def validate(self) -> "PlannerResult":
        if not self.scene_objective.strip() or self.pacing not in {"slow", "steady", "fast"}:
            raise PlannerValidationError("malformed planner narrative")
        if not 0 <= self.tension <= 1 or not 0 <= self.confidence <= 1:
            raise PlannerValidationError("tension and confidence must be between 0 and 1")
        if self.validation_status not in {"validated", "disabled", "fallback", "error"}:
            raise PlannerValidationError("invalid validation status")
        if self.latency_ms < 0 or any(not isinstance(item, str) or not item.strip()
                                     for item in self.relevant_memory_ids):
            raise PlannerValidationError("malformed planner output")
        if self.fallback and not self.error:
            raise PlannerValidationError("fallback output requires an error")
        return self


class NarrativePlanner:
    """Produces advisory narrative metadata without changing the locked action."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = _env_bool("NARRATIVE_PLANNER_ENABLED", True) if enabled is None else enabled

    def plan(self, planner_input: PlannerInput) -> PlannerResult:
        if not isinstance(planner_input, PlannerInput):
            raise PlannerValidationError("plan requires PlannerInput")
        started = time.perf_counter()
        if not self.enabled:
            return PlannerResult(
                "No advisory plan generated.", "steady", 0.0, (), (), (), (),
                ("Render the locked action without advisory changes.",),
                "Do not reveal secrets.", 0.0, "disabled", "local", "disabled",
                _latency(started),
            )
        memory_ids = tuple(self._memory_id(memory) for memory in planner_input.memories)
        state = planner_input.current_state
        tension = _number(state.get("tension", 0.5), 0.5)
        tension = max(0.0, min(1.0, tension))
        pacing = "fast" if tension >= 0.7 else "slow" if tension <= 0.3 else "steady"
        digest = hashlib.sha256(f"{planner_input.seed}:{planner_input.scene_index}".encode()).hexdigest()
        objective = f"Advance the scene while honoring the locked action ({digest[:8]})."
        constraints = tuple(c for knowledge in planner_input.character_knowledge
                            for c in knowledge.constraints)
        result = PlannerResult(
            objective, pacing, tension, memory_ids, constraints,
            ("Resolve the immediate consequence of the locked action.",),
            ("Avoid contradicting validated memory.",),
            ("Keep the locked action unchanged.", "Use implication before exposition."),
            "Preserve secret information; reveal only when character knowledge permits.",
            0.9 if memory_ids else 0.75, "validated", "local", "deterministic-v1",
            _latency(started),
        )
        return result.validate()

    def validate_result(self, result: PlannerResult, planner_input: PlannerInput) -> PlannerResult:
        """Validate provider-shaped output against the input contract."""
        if not isinstance(result, PlannerResult):
            raise PlannerValidationError("result must be a PlannerResult")
        result.validate()
        allowed = {self._memory_id(memory) for memory in planner_input.memories}
        if not set(result.relevant_memory_ids).issubset(allowed):
            raise PlannerValidationError("result references memory outside the input")
        if len(result.relevant_memory_ids) != len(set(result.relevant_memory_ids)):
            raise PlannerValidationError("result contains duplicate memory IDs")
        return result

    @staticmethod
    def _memory_id(memory: MemoryRecord | MemoryReference) -> str:
        if isinstance(memory, MemoryRecord):
            return str(memory.id) if memory.id is not None else f"{memory.category}:{memory.content}"
        return memory.memory_id


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latency(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
