from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import FrozenSet, Optional, Tuple


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SecretStatus(str, Enum):
    HIDDEN = "hidden"
    REVEALED = "revealed"


@dataclass(frozen=True)
class RevealConditions:
    min_scene: Optional[int] = None
    required_event: Optional[str] = None
    character_knowledge: Tuple[str, ...] = ()
    prerequisite_secret: Optional[str] = None
    explicit_instruction: Optional[str] = None


@dataclass(frozen=True)
class SecretRecord:
    story_id: str
    secret_id: str
    protected_value: str
    aliases: Tuple[str, ...] = ()
    conditions: RevealConditions = field(default_factory=RevealConditions)
    status: SecretStatus = SecretStatus.HIDDEN
    revealed_scene: Optional[int] = None
    id: Optional[int] = None
    created_at: datetime = field(default_factory=_now)
    revealed_at: Optional[datetime] = None

    @property
    def value(self) -> str:
        return self.protected_value

    @property
    def min_scene(self) -> Optional[int]:
        return self.conditions.min_scene

    def validate(self) -> "SecretRecord":
        if not self.story_id.strip() or not self.secret_id.strip():
            raise ValueError("story_id and secret_id are required")
        if not self.protected_value:
            raise ValueError("protected_value is required")
        if self.min_scene is not None and self.min_scene < 0:
            raise ValueError("min_scene cannot be negative")
        return self


@dataclass(frozen=True)
class RevealContext:
    scene_index: int = 0
    events: FrozenSet[str] = frozenset()
    character_knowledge: FrozenSet[str] = frozenset()
    revealed_secret_ids: FrozenSet[str] = frozenset()
    explicit_instructions: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class RevealDecision:
    allowed: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RevealAudit:
    story_id: str
    secret_id: str
    authorized: bool
    actor: str
    scene_index: int
    reasons: Tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=_now)

