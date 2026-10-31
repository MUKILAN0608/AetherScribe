from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"


class MemoryValidationError(ValueError):
    """Raised when a memory record cannot safely be persisted."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class MemoryRecord:
    story_id: str
    category: str
    content: str
    confidence: float = 1.0
    origin: str = "unknown"
    episode_id: Optional[str] = None
    scene_index: Optional[int] = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    id: Optional[int] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    supersedes_id: Optional[int] = None

    def validate(self) -> "MemoryRecord":
        if not isinstance(self.story_id, str) or not self.story_id.strip():
            raise MemoryValidationError("story_id is required")
        if not isinstance(self.category, str) or not self.category.strip():
            raise MemoryValidationError("category is required")
        if not isinstance(self.content, str) or not self.content.strip():
            raise MemoryValidationError("content is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise MemoryValidationError("confidence must be between 0 and 1")
        if self.scene_index is not None and int(self.scene_index) < 0:
            raise MemoryValidationError("scene_index cannot be negative")
        return self
