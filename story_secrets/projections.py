from typing import Iterable, Mapping
from .models import SecretRecord, SecretStatus


def knowledge_projection(
    secrets: Iterable[SecretRecord], known_secret_ids: Iterable[str] = ()
) -> Mapping[str, str]:
    known = set(known_secret_ids)
    return {
        secret.secret_id: secret.protected_value
        for secret in secrets
        if secret.status is SecretStatus.REVEALED or secret.secret_id in known
    }


def audience_projection(
    secrets: Iterable[SecretRecord], audience: str = "audience"
) -> Mapping[str, str]:
    """Project only secrets authorized for general audience consumption."""
    del audience  # reserved for future audience-specific policy
    return {
        secret.secret_id: secret.protected_value
        for secret in secrets
        if secret.status is SecretStatus.REVEALED
    }


project_knowledge = knowledge_projection
project_audience = audience_projection
