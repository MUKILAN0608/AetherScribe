"""Story-scoped secret protection and authorized reveal primitives."""

# Keep the package compatible with Python's stdlib ``secrets`` surface. The
# repository package has this established name, and third-party libraries may
# import ``secrets.randbits`` during startup.
from random import SystemRandom
from hmac import compare_digest

_sysrand = SystemRandom()
randbits = _sysrand.getrandbits
choice = _sysrand.choice
randbelow = _sysrand.randrange

def token_bytes(nbytes=None):
    return _sysrand.randbytes(nbytes or 32)

def token_hex(nbytes=None):
    return token_bytes(nbytes).hex()

def token_urlsafe(nbytes=None):
    import base64
    return base64.urlsafe_b64encode(token_bytes(nbytes)).rstrip(b"=").decode("ascii")

from .models import (
    RevealConditions,
    RevealContext,
    RevealDecision,
    RevealAudit,
    SecretRecord,
    SecretStatus,
)
from .gate import RevealGate, authorize_reveal
from .projections import audience_projection, knowledge_projection
from .validator import ProseValidationResult, validate_prose, validate_prose_for_secrets

__all__ = [
    "RevealConditions", "RevealContext", "RevealDecision", "RevealAudit",
    "SecretRecord", "SecretStatus", "RevealGate", "authorize_reveal",
    "audience_projection", "knowledge_projection", "ProseValidationResult",
    "validate_prose", "validate_prose_for_secrets",
]
