from story_secrets import (
    RevealConditions, RevealContext, SecretRecord, SecretStatus,
    RevealGate, audience_projection, knowledge_projection, validate_prose,
)
from evaluation.database import AetherDatabase


def test_story_isolation_and_conditions():
    secret = SecretRecord(
        "one", "origin", "The Origin", aliases=("first cause",),
        conditions=RevealConditions(2, "opened", ("alice",), "key", "reveal"),
    )
    gate = RevealGate()
    denied = gate.evaluate(secret, RevealContext())
    assert not denied.allowed
    assert "required_event_missing" in denied.reasons
    context = RevealContext(2, frozenset({"opened"}), frozenset({"alice"}),
                            frozenset({"key"}), frozenset({"reveal"}))
    revealed, audit = gate.authorize(secret, context, "tester")
    assert revealed.status is SecretStatus.REVEALED
    assert audit.authorized
    assert gate.evaluate(revealed, context).reasons == ("already_revealed",)


def test_projections_validator_and_rejection_are_redacted():
    hidden = SecretRecord("one", "hidden", "Never shown")
    shown = SecretRecord("one", "shown", "Known", status=SecretStatus.REVEALED)
    assert "hidden" not in audience_projection([hidden, shown])
    assert knowledge_projection([hidden, shown], {"hidden"})["hidden"] == "Never shown"
    result = validate_prose("The first-cause was found.", [
        SecretRecord("one", "origin", "The Origin", aliases=("first cause",))
    ])
    assert not result.valid
    assert result.violations[0].evidence == "[REDACTED]"
    assert "Origin" not in repr(result.violations)


def test_sqlite_secret_storage_is_story_scoped():
    path = "stage4_test_secrets.db"
    try:
        database = AetherDatabase("sqlite:///" + path)
        database.add_secret(SecretRecord("a", "same", "A"))
        database.add_secret(SecretRecord("b", "same", "B"))
        assert database.get_secret("a", "same").value == "A"
        assert database.get_secret("b", "same").value == "B"
        assert len(database.list_secrets("a")) == 1
    finally:
        import os
        if "database" in locals():
            database.engine.dispose()
        if os.path.exists(path):
            os.remove(path)
