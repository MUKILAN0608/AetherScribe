"""Counterfactual future proposals; this module is intentionally read-only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PredictedFuture:
    future_id: str
    label: str
    proposed_action: str
    predicted_events: tuple[str, ...]
    predicted_state: Mapping[str, Any]
    risks: tuple[str, ...]
    plausibility: float | None = None
    status: str = "prediction"


@dataclass(frozen=True)
class PredictionReport:
    source_version: int
    source_digest: str
    futures: tuple[PredictedFuture, ...]
    authoritative_action: str | None = None
    committed: bool = False
    method: str = "transparent-rule-based"


class FutureWorldPredictor:
    """Creates bounded, explicitly hypothetical consequence branches."""
    def predict(self, authoritative_state: Mapping[str, Any],
                candidate_actions: Sequence[str], horizon: int = 2) -> PredictionReport:
        if horizon < 1 or horizon > 5:
            raise ValueError("prediction horizon must be between one and five steps")
        before = json.dumps(authoritative_state, sort_keys=True, default=str)
        version = int(authoritative_state.get("version", 0))
        futures = []
        for action in tuple(candidate_actions)[:8]:
            if not str(action).strip():
                continue
            label = str(action).casefold()
            if any(term in label for term in ("protect", "secure", "defend", "shelter")):
                effect, risks = "immediate exposure may decrease while the protected objective remains unresolved", ("protection may consume time or resources",)
            elif any(term in label for term in ("investigate", "search", "analyze", "decipher")):
                effect, risks = "available information may increase, changing which choices remain plausible", ("investigation may expose the actor or delay another goal",)
            elif any(term in label for term in ("negotiate", "reconcile", "persuade", "ally")):
                effect, risks = "the relationship may shift and enable or foreclose cooperation", ("the other party may reject or exploit the offer",)
            elif any(term in label for term in ("attack", "confront", "strike", "ambush")):
                effect, risks = "conflict pressure and physical risk may rise", ("injury, escalation, or lasting relationship damage",)
            elif any(term in label for term in ("retreat", "escape", "withdraw")):
                effect, risks = "immediate danger may lessen while the central objective is delayed", ("opposition may gain time or control",)
            else:
                effect, risks = "the situation changes and requires reassessment before any commitment", ("the result is uncertain without a domain transition model",)
            events = tuple(
                f"Hypothesis {n}: {effect}. This remains a prediction until the authoritative action is committed and observed."
                for n in range(1, horizon + 1)
            )
            payload = {"source_version": version, "action": str(action), "events": events}
            future_id = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            future_state = {"status": "prediction", "based_on_version": version,
                            "proposed_action": str(action), "hypothesized_effect": effect,
                            "sequence": list(events)}
            futures.append(PredictedFuture(future_id, f"Possible outcome {len(futures)+1}", str(action),
                                           events, future_state,
                                           (*risks, "Outcome is hypothetical and requires VQC evaluation."), None))
        after = json.dumps(authoritative_state, sort_keys=True, default=str)
        if before != after:
            raise RuntimeError("prediction changed authoritative state")
        digest = sha256(before.encode()).hexdigest()
        return PredictionReport(version, digest, tuple(futures))
