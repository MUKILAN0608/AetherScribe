"""Validated scene lifecycle transitions used by episode execution."""

from dataclasses import dataclass, field


ALLOWED_TRANSITIONS = {
    "pending": {"generating", "blocked"},
    "generating": {"generated", "failed"},
    "generated": {"validating", "failed"},
    "validating": {"revision_required", "accepted", "failed"},
    "revision_required": {"retrying", "failed"},
    "retrying": {"generating", "failed"},
    "accepted": set(), "failed": set(), "blocked": set(),
}


@dataclass
class SceneLifecycle:
    scene_id: str
    status: str = "pending"
    history: list[dict] = field(default_factory=list)

    def transition(self, target: str, *, reason: str | None = None) -> None:
        if target not in ALLOWED_TRANSITIONS.get(self.status, set()):
            raise ValueError(f"Invalid scene lifecycle transition: {self.status} -> {target}")
        self.status = target
        self.history.append({"status": target, **({"reason": reason} if reason else {})})

    def snapshot(self) -> dict:
        return {"scene_id": self.scene_id, "status": self.status, "history": list(self.history)}
