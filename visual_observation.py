"""Read-only rendered-frame observation through an explicit detector adapter."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class VisualFinding:
    entity_id: str
    status: str
    expected: Any
    observed: Any
    confidence: float
    evidence: str
    suggested_correction: str


@dataclass(frozen=True)
class ObservationReport:
    status: str
    frame_digest: str
    image_size: tuple[int, int] | None
    findings: tuple[VisualFinding, ...]
    discrepancies: tuple[str, ...]
    canonical_state_modified: bool = False
    error: str | None = None


class VisualObserver:
    """An image detector is injected so absent vision weights never look like success."""
    def observe(self, frame: bytes | bytearray | str | Path,
                expected_state: Mapping[str, Any],
                detector: Callable[[bytes], list[Mapping[str, Any]]] | None = None) -> ObservationReport:
        if isinstance(frame, (str, Path)):
            data = Path(frame).read_bytes()
        elif isinstance(frame, (bytes, bytearray)):
            data = bytes(frame)
        else:
            raise TypeError("frame must be image bytes or a file path")
        if not data:
            raise ValueError("rendered frame is empty")
        digest = sha256(data).hexdigest()
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(BytesIO(data)) as image:
                image_size = tuple(image.size)
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"invalid rendered image: {exc}") from exc
        if detector is None:
            return ObservationReport("unavailable", digest, image_size, (), (), False,
                                     "No visual detector is configured; image semantics were not inferred.")
        observed_items = detector(data)
        observed = {str(item.get("entity_id")): item for item in observed_items}
        findings, discrepancies = [], []
        for entity_id, expected in expected_state.items():
            actual = observed.get(str(entity_id))
            if actual is None:
                findings.append(VisualFinding(str(entity_id), "uncertain", expected, None, 0.0,
                                              "detector returned no entity match", "Review frame or detector output."))
                continue
            confidence = max(0.0, min(1.0, float(actual.get("confidence", 0.0))))
            expected_status = (expected.get("status", expected.get("condition"))
                               if isinstance(expected, Mapping) else expected)
            observed_status = actual.get("status", actual.get("condition"))
            mismatch = expected_status is not None and observed_status != expected_status
            status = "discrepancy" if mismatch and confidence >= .5 else "uncertain" if mismatch else "confirmed"
            evidence = str(actual.get("evidence", "detector observation"))
            finding = VisualFinding(str(entity_id), status, expected_status, observed_status,
                                    confidence, evidence,
                                    "Update the scene only through the coordinator after human review." if mismatch else "No correction suggested.")
            findings.append(finding)
            if mismatch and confidence >= .5:
                discrepancies.append(str(entity_id))
        return ObservationReport("complete", digest, image_size, tuple(findings), tuple(discrepancies), False)
