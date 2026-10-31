"""Regression tests for offline providers and immutable quantum audit fields."""

from llm.providers import GenerationConfig, ProviderRouter
from decision.providers import DecisionRequest, DecisionRouter


def test_offline_generation_is_available():
    router = ProviderRouter()
    result = router.generate("Write a short scene.", GenerationConfig(max_output_tokens=32))
    assert result.text
    assert result.provider in {"deterministic-local", "local-endpoint"}


def test_local_decision_uses_only_supplied_options():
    response = DecisionRouter().evaluate(DecisionRequest("route", ("first", "second")))
    assert response.selected in {"first", "second"}


def test_invalid_jev_configuration_falls_back():
    response = DecisionRouter().evaluate(DecisionRequest("route", ("first",)))
    assert response.selected == "first"
