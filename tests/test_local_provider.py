"""Local provider contract tests; no model download or external service required."""

import json
from unittest.mock import patch

import pytest

from llm.providers import (
    DeterministicLocalProvider,
    GenerationConfig,
    OpenAICompatibleLocalProvider,
    ProviderRouter,
)
from llm.story_renderer import StoryRenderer


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_deterministic_provider_is_offline():
    result = DeterministicLocalProvider().generate("Write a scene.", GenerationConfig())
    assert result.text
    assert result.used_fallback is True


def test_project_env_has_one_consistent_local_context_size():
    from pathlib import Path

    env_path = Path(__file__).resolve().parents[1] / ".env"
    configured = [
        line.split("=", 1)[1].strip()
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("LOCAL_LLM_CONTEXT_SIZE=")
    ]
    assert configured == ["2048"]


def test_local_runtime_unavailable_falls_back():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "",
        "LLM_PROVIDER": "local",
        "LOCAL_LLM_RUNTIME": "llama_cpp",
        "LOCAL_LLM_BASE_URL": "http://127.0.0.1:1/v1",
        "LOCAL_LLM_MODEL_PATH": "missing.gguf",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "true",
    }, clear=False):
        router = ProviderRouter()
    assert router.provider.provider_name == "deterministic-local"
    assert router.provider_error
    assert router.health_check()["state"] == "provider_error"


def test_unavailable_local_provider_does_not_silently_fallback():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "local",
        "LLM_PROVIDER": "local",
        "LOCAL_LLM_RUNTIME": "llama_cpp",
        "LOCAL_LLM_BASE_URL": "http://127.0.0.1:1/v1",
        "LOCAL_LLM_MODEL_PATH": "missing.gguf",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "false",
    }, clear=False):
        router = ProviderRouter()
        health = router.health_check()
        assert health["available"] is False
        assert health["provider"] == "local"
        assert health["model"] == "missing"
        assert health["endpoint"] == "http://127.0.0.1:1/v1"
        with pytest.raises(RuntimeError):
            router.generate("Write a scene.", GenerationConfig())


def test_unavailable_provider_diagnostics_include_server_output(monkeypatch, tmp_path):
    from llm.providers import LlamaCppServerProvider

    log_path = tmp_path / "llama.log"
    provider = LlamaCppServerProvider.__new__(LlamaCppServerProvider)
    provider._server_log = open(log_path, "w+b")
    try:
        provider._server_log.write(b"error: Vulkan device could not allocate model buffers")
        message = provider._server_diagnostics()
    finally:
        provider._server_log.close()
    assert "Vulkan device could not allocate" in message


def test_qwen30_configuration_must_match_local_artifact():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "local",
        "LOCAL_LLM_BASE_URL": "http://127.0.0.1:1/v1",
        "LOCAL_LLM_MODEL_NAME": "Qwen3-30B-A3B-Q4_K_M",
        "LOCAL_LLM_RUNTIME": "llama_cpp",
        "LOCAL_LLM_MODEL_PATH": "models/OLMoE-1B-7B-0125-Instruct.Q4_K_M.gguf",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "false",
    }, clear=False):
        router = ProviderRouter()
    assert "does not match local model file" in router.provider_error


def test_removed_hosted_provider_falls_back_without_network():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "removed-hosted-provider",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "true",
    }, clear=False):
        router = ProviderRouter()
        result = router.generate("Write a short scene.", GenerationConfig(max_output_tokens=32))
    assert result.provider == "deterministic-local"
    assert result.used_fallback is True
    assert result.text


def test_local_endpoint_success():
    provider = OpenAICompatibleLocalProvider("http://127.0.0.1:8080/v1", "test-model")
    with patch("urllib.request.urlopen", return_value=FakeResponse({
        "choices": [{"message": {"content": "local scene"}}],
    })):
        result = provider.generate("prompt", GenerationConfig())
    assert result.text == "local scene"
    assert result.used_fallback is False


def test_router_reuses_healthy_endpoint_and_discovered_model():
    responses = [
        FakeResponse({"data": [{"id": "loaded/qwen-model"}]}),
        FakeResponse({
            "model": "loaded/qwen-model",
            "choices": [{"message": {"content": "Elias opened the evidence ledger."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8},
        }),
    ]
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "local",
        "LLM_PROVIDER": "local",
        "LOCAL_LLM_RUNTIME": "llama_cpp",
        "LOCAL_LLM_BASE_URL": "http://127.0.0.1:8080/v1",
        "LOCAL_LLM_MODEL_NAME": "qwen-model",
        "LOCAL_LLM_MODEL_PATH": "missing.gguf",
    }, clear=False), patch("urllib.request.urlopen", side_effect=responses) as urlopen:
        router = ProviderRouter()
        result = router.generate("Write the next scene.", GenerationConfig(seed=17))
    assert router.provider.provider_name == "local-endpoint"
    assert router.provider.discovered_model == "loaded/qwen-model"
    assert result.model == "loaded/qwen-model"
    assert result.input_tokens == 30
    assert result.output_tokens == 8
    request = urlopen.call_args_list[1].args[0]
    assert json.loads(request.data)["model"] == "loaded/qwen-model"
    assert json.loads(request.data)["seed"] == 17


def test_router_auto_selects_discovered_model():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "local",
        "LOCAL_LLM_BASE_URL": "http://127.0.0.1:8080/v1",
        "LOCAL_LLM_MODEL_NAME": "auto",
        "LOCAL_LLM_RUNTIME": "",
        "LOCAL_LLM_MODEL_PATH": "",
    }, clear=False), patch("urllib.request.urlopen", return_value=FakeResponse({
        "data": [{"id": "fast-qwen-quant"}],
    })):
        router = ProviderRouter()
    assert router.provider.model_name == "fast-qwen-quant"
    assert router.health_check()["model_selection"] == "discovered"


def test_generation_configures_fast_output_cap(monkeypatch):
    from llm.story_renderer import StoryRenderer

    renderer = StoryRenderer.__new__(StoryRenderer)
    renderer.router = type("Router", (), {"provider": type("Provider", (), {"provider_name": "local-endpoint"})()})()
    monkeypatch.setenv("NARRATIVE_FAST_MODE", "true")
    monkeypatch.setenv("NARRATIVE_FAST_MAX_OUTPUT_TOKENS", "64")
    monkeypatch.setenv("NARRATIVE_MAX_OUTPUT_TOKENS", "256")
    captured = {}

    class Router:
        provider = type("Provider", (), {"provider_name": "local-endpoint"})()
        def generate(self, _prompt, config):
            captured["config"] = config
            return type("Result", (), {
                "provider": "local-endpoint", "model": "fast-test", "error": None,
                "latency_ms": 1.0, "used_fallback": False, "input_tokens": None,
                "output_tokens": None, "estimated_cost_usd": None, "provider_metadata": {},
                "text": "fast response",
            })()

    renderer.router = Router()
    renderer.scene_latency_ms = renderer.scene_input_tokens = renderer.scene_output_tokens = 0
    renderer.scene_estimated_cost_usd = 0
    renderer._generate_text("prompt", max_output_tokens=128)
    assert captured["config"].max_output_tokens == 64


def test_retry_after_truncation_gets_more_output_tokens(monkeypatch):
    from llm.story_renderer import StoryRenderer

    renderer = StoryRenderer.__new__(StoryRenderer)
    renderer.router = type("Router", (), {"provider": type("Provider", (), {"provider_name": "local-endpoint"})()})()
    monkeypatch.setenv("NARRATIVE_FAST_MODE", "true")
    monkeypatch.setenv("NARRATIVE_FAST_MAX_OUTPUT_TOKENS", "96")
    captured = {}

    class Router:
        provider = type("Provider", (), {"provider_name": "local-endpoint"})()
        def generate(self, _prompt, config):
            captured["tokens"] = config.max_output_tokens
            captured["temperature"] = config.temperature
            captured["top_p"] = config.top_p
            return type("Result", (), {
                "provider": "local-endpoint", "model": "fast-test", "error": None,
                "latency_ms": 1.0, "used_fallback": False, "input_tokens": None,
                "output_tokens": None, "estimated_cost_usd": None, "provider_metadata": {},
                "text": "short complete result",
            })()

    renderer.router = Router()
    renderer.scene_latency_ms = renderer.scene_input_tokens = renderer.scene_output_tokens = 0
    renderer.scene_estimated_cost_usd = 0
    renderer._generate_text("prompt Retry 2 of 3.", max_output_tokens=128)
    assert captured["tokens"] == 192
    assert captured["temperature"] > 0.8
    assert captured["top_p"] > 0.8


def test_router_reports_model_id_mismatch_without_fallback():
    with patch.dict("os.environ", {
        "NARRATIVE_LLM_PROVIDER": "local",
        "LOCAL_LLM_BASE_URL": "http://127.0.0.1:8080/v1",
        "LOCAL_LLM_MODEL_NAME": "Qwen3-30B-A3B-IQ2_M",
        "LOCAL_LLM_RUNTIME": "",
        "LOCAL_LLM_MODEL_PATH": "",
        "NARRATIVE_ALLOW_LOCAL_FALLBACK": "false",
    }, clear=False), patch("urllib.request.urlopen", return_value=FakeResponse({
        "data": [{"id": "qwen2.5-0.5b-instruct-q4_k_m.gguf"}],
    })):
        router = ProviderRouter()
        assert router.provider.provider_name == "deterministic-local"
        assert router.health_check()["state"] == "provider_error"
        with pytest.raises(RuntimeError, match="configuration requests"):
            router.generate("Write a scene.", GenerationConfig())


def test_local_endpoint_malformed_output_raises():
    provider = OpenAICompatibleLocalProvider("http://127.0.0.1:8080/v1", "test-model")
    with patch("urllib.request.urlopen", return_value=FakeResponse({"unexpected": True})):
        with pytest.raises(RuntimeError):
            provider.generate("prompt", GenerationConfig())


def test_local_endpoint_timeout_raises():
    provider = OpenAICompatibleLocalProvider("http://127.0.0.1:8080/v1", "test-model")
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(RuntimeError):
            provider.generate("prompt", GenerationConfig(timeout_seconds=0.01))


def test_malformed_scene_is_rejected():
    assert StoryRenderer._normalize_scene(
        "[SCENE 0]\nNARRATIVE:\n<generated prose>\n[QUANTUM_TRACE]\n<reason>",
        {"Protagonist": "Investigate Anomaly"},
    ) is None


def test_compact_local_prompt_requests_reader_facing_prose_and_evidence():
    from llm.prompt_templates import build_compact_local_prompt

    prompt = build_compact_local_prompt(
        {"director_brief": {"characters": {"protagonist": "Mira"}}},
        {}, 0, planning_guidance=("VALIDATION CLUES: badge audit",),
    )
    assert "reader-facing scene prose" in prompt
    assert "Do not add headings" in prompt
    assert "badge audit" in prompt


def test_compact_prompt_respects_user_genre_and_writing_preferences():
    from llm.prompt_templates import build_compact_local_prompt

    prompt = build_compact_local_prompt(
        {"genre": "romance", "director_brief": {
            "characters": {"protagonist": "Ari"}, "tone": "wry and tender",
            "writing_style": "spare", "pov_preference": "First person",
            "ending_preference": "Bittersweet", "world_rules": "No magic.",
        }}, {}, 0,
    )
    assert "romance sentences" in prompt
    assert "Tone: wry and tender" in prompt
    assert "Style: spare" in prompt
    assert "POV: First person" in prompt
    assert "Ending: Bittersweet" in prompt
    assert "World rules: No magic." in prompt


def test_scene_normalization_keeps_locked_action_out_of_reader_text():
    normalized = StoryRenderer._normalize_scene(
        "[SCENE 0]\nNARRATIVE:\nMira entered the archive.\n"
        "[QUANTUM_TRACE]\nThe door opened.",
        {"Protagonist": "Investigate Anomaly"},
    )
    assert normalized is not None
    assert "Mira entered the archive." in normalized
    assert "Investigate Anomaly" not in normalized


def test_scene_normalization_wraps_plain_prose_with_one_based_scene_number():
    normalized = StoryRenderer._normalize_scene(
        "Mira enters the records room. Voss locks the archive. Nia sends a ledger number.",
        {},
        scene_index=0,
    )
    assert normalized is not None
    assert normalized.startswith("[SCENE 1]")
    assert "NARRATIVE:\nMira enters" in normalized
    assert "QUANTUM_TRACE" in normalized
