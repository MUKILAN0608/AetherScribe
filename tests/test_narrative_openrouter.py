"""Mocked hosted narrative-provider and routing coverage."""

import json
from unittest.mock import patch

import pytest

from llm.providers import (
    GenerationConfig,
    OpenRouterNarrativeProvider,
    ProviderRouter,
)


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openrouter_narrative_success_records_usage():
    provider = OpenRouterNarrativeProvider("test-key", "qwen/qwen3.5-35b-a3b")
    response = {
        "id": "request-1",
        "model": "qwen/qwen3.5-35b-a3b",
        "choices": [{"message": {"content": "A valid scene."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }
    with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
        result = provider.generate("render the locked action", GenerationConfig())
    assert result.text == "A valid scene."
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.provider == "openrouter"


@pytest.mark.parametrize("error", [TimeoutError("timeout"), OSError("401"), ValueError("bad")])
def test_openrouter_narrative_errors_are_explicit(error):
    provider = OpenRouterNarrativeProvider("test-key", "qwen/qwen3.5-35b-a3b")
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(RuntimeError):
            provider.generate("render", GenerationConfig())


def test_narrative_model_is_independent_from_jev_model():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "narrative-key",
        "NARRATIVE_OPENROUTER_MODEL": "qwen/qwen3.5-35b-a3b",
        "JEV_MODEL": "meta-llama/llama-3.1-8b-instruct",
    }, clear=False):
        router = ProviderRouter()
    assert router.provider.model_name == "qwen/qwen3.5-35b-a3b"


def test_hosted_failure_uses_local_or_deterministic_fallback():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "narrative-key",
        "NARRATIVE_OPENROUTER_MODEL": "qwen/qwen3.5-35b-a3b",
        "LOCAL_LLM_MODEL_PATH": "missing.gguf",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "true",
    }, clear=False):
        router = ProviderRouter()
        with patch.object(router.provider, "generate", side_effect=RuntimeError("rate limited")):
            result = router.generate("render", GenerationConfig())
    assert result.used_fallback is True
    assert result.provider == "deterministic-local"


def test_fallback_can_be_disabled():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "narrative-key",
        "NARRATIVE_OPENROUTER_MODEL": "qwen/qwen3.5-35b-a3b",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "false",
    }, clear=False):
        router = ProviderRouter()
        with patch.object(router.provider, "generate", side_effect=RuntimeError("unavailable")):
            with pytest.raises(RuntimeError):
                router.generate("render", GenerationConfig())
