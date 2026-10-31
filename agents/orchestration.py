"""Typed, sequential agent contracts for the narrative workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DirectorBrief:
    premise: str
    genre: str
    characters: dict[str, str]
    language: str = "English"
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentTrace:
    agent: str
    provider: str
    success: bool
    duration_ms: float
    fallback: bool = False
    error: str | None = None


class DirectorAgent:
    name = "director"

    def parse(self, premise: str, genre: str, characters: dict[str, str], language: str = "English") -> DirectorBrief:
        return DirectorBrief(premise.strip(), genre, dict(characters), language)


class PlanningAgent:
    name = "planning"

    def propose(self, brief: DirectorBrief) -> dict[str, Any]:
        return {"premise": brief.premise, "genre": brief.genre, "goals": list(brief.characters.values())}


class CriticAgent:
    name = "critic"

    def evaluate(self, scene: str, locked_action: dict[str, str]) -> dict[str, Any]:
        return {"follows_locked_action": bool(scene.strip()), "findings": [], "locked_action": dict(locked_action)}
