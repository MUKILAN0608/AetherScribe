"""Event-grounded character adaptation and controlled counterfactual experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from agents.orchestration import NarrativeEvent


@dataclass(frozen=True)
class CharacterProfile:
    character_id: str
    goal_consistency: float | None = None
    resolve: float = 0.5
    trust: Mapping[str, float] = field(default_factory=dict)
    emotional_state: str = "unestablished"
    knowledge: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptationRecord:
    character_id: str
    event_id: str
    before: CharacterProfile
    after: CharacterProfile
    grounded_facts: tuple[str, ...]
    metrics: Mapping[str, float]


class TemporalCharacterAdapter:
    def __init__(self, profiles: Mapping[str, CharacterProfile] | None = None):
        self.profiles = dict(profiles or {})
        self.records: list[AdaptationRecord] = []

    def apply_committed_event(self, event: NarrativeEvent) -> tuple[AdaptationRecord, ...]:
        records = []
        evidence = (*event.facts, *event.consequences)
        for character_id, decision in event.decisions.items():
            before = self.profiles.get(character_id, CharacterProfile(character_id))
            effect = event.character_effects.get(character_id, {})
            support = str(effect.get("support_fact", ""))
            grounded = bool(support and support in evidence)
            delta = float(effect.get("resolve_delta", 0.0)) if grounded else 0.0
            trust = dict(before.trust)
            for other, value in dict(effect.get("trust_delta", {})).items():
                if grounded:
                    trust[other] = min(1.0, max(0.0, trust.get(other, 0.5) + float(value)))
            changed = CharacterProfile(
                character_id=character_id,
                goal_consistency=(
                    min(1.0, max(0.0, float(effect["goal_consistency"])))
                    if grounded and "goal_consistency" in effect else before.goal_consistency
                ),
                resolve=min(1.0, max(0.0, before.resolve + delta)),
                trust=trust,
                emotional_state=str(effect.get("emotional_state", before.emotional_state)) if grounded else before.emotional_state,
                knowledge=tuple(dict.fromkeys((*before.knowledge, *event.facts))),
                decisions=(*before.decisions, str(decision)),
                source_event_ids=(*before.source_event_ids, event.event_id),
            )
            record = AdaptationRecord(character_id, event.event_id, before, changed,
                                      tuple(evidence),
                                      {"goal_consistency": changed.goal_consistency,
                                       "behavioral_adaptation": abs(changed.resolve-before.resolve),
                                       "knowledge_delta": float(len(changed.knowledge)-len(before.knowledge))})
            self.profiles[character_id] = changed
            self.records.append(record)
            records.append(record)
        return tuple(records)

    @staticmethod
    def controlled_experiment(condition_a: Sequence[NarrativeEvent], condition_b: Sequence[NarrativeEvent],
                              decision_policy: Callable[[CharacterProfile], str],
                              character_id: str) -> dict:
        """Replay two event conditions through the same deterministic policy."""
        def run(events):
            adapter = TemporalCharacterAdapter({character_id: CharacterProfile(character_id)})
            for event in events:
                adapter.apply_committed_event(event)
            profile = adapter.profiles[character_id]
            return {"profile": profile, "decision": decision_policy(profile),
                    "event_ids": profile.source_event_ids}
        a, b = run(condition_a), run(condition_b)
        return {"condition_a": a, "condition_b": b,
                "decision_changed": a["decision"] != b["decision"],
                "resolve_delta": b["profile"].resolve - a["profile"].resolve,
                "grounded": bool(a["event_ids"] or b["event_ids"])}
