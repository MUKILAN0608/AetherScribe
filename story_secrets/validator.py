import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Tuple
from .models import SecretRecord


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


@dataclass(frozen=True)
class ProseViolation:
    secret_id: str
    evidence: str


@dataclass(frozen=True)
class ProseValidationResult:
    valid: bool
    violations: Tuple[ProseViolation, ...] = ()


def validate_prose(prose: str, secrets: Iterable[SecretRecord]) -> ProseValidationResult:
    """Reject protected values in prose, without returning the matched value."""
    normalized_prose = _normalized(prose)
    violations = []
    for secret in secrets:
        candidates = (secret.protected_value,) + tuple(secret.aliases)
        if any(candidate and _normalized(candidate) in normalized_prose for candidate in candidates):
            violations.append(ProseViolation(secret.secret_id, "[REDACTED]"))
    return ProseValidationResult(not violations, tuple(violations))


validate_prose_for_secrets = validate_prose
