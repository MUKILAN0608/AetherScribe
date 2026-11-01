"""Provider-neutral local language generation with CPU-safe fallbacks."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import atexit
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


logger = logging.getLogger("AetherScribeLab.LLM")


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    repeat_penalty: float = 1.05
    max_output_tokens: int = 320
    timeout_seconds: float = 180.0
    context_size: int = 4096
    seed: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    used_fallback: bool = False
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    provider_metadata: Mapping[str, Any] | None = None


class LocalLLMProvider(Protocol):
    provider_name: str
    model_name: str

    def health_check(self) -> dict[str, Any]:
        ...

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        ...


class ResourceManager:
    """Small, dependency-free CPU and model-resource probe."""

    @staticmethod
    def snapshot(model_path: str | None = None) -> dict[str, Any]:
        logical_cores = os.cpu_count() or 1
        available_ram_mb = None
        try:
            import psutil  # type: ignore

            available_ram_mb = int(psutil.virtual_memory().available / 1024**2)
        except ImportError:
            pass

        model_size_mb = None
        if model_path and os.path.isfile(model_path):
            model_size_mb = int(os.path.getsize(model_path) / 1024**2)
        return {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cores": logical_cores,
            "available_ram_mb": available_ram_mb,
            "cuda_available": False,
            "model_path": model_path or "",
            "model_size_mb": model_size_mb,
            "recommended_threads": max(1, min(logical_cores, 8)),
        }


class DeterministicLocalProvider:
    """Offline provider used when no verified local runtime is configured."""

    provider_name = "deterministic-local"
    model_name = "template-fallback"

    def health_check(self) -> dict[str, Any]:
        return {
            "available": True,
            "provider": self.provider_name,
            "model": self.model_name,
            "offline": True,
            "message": "Deterministic local fallback is ready.",
        }

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        started = time.perf_counter()
        text = self._respond(prompt)
        return GenerationResult(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            used_fallback=True,
        )

    @staticmethod
    def _respond(prompt: str) -> str:
        if "single floating-point number" in prompt:
            return "0.90"
        if prompt.lower().startswith("summarize"):
            source = prompt.split("\n\n", 1)[-1].strip().replace("\n", " ")
            return re.sub(r"\s+", " ", source)[:180] or "The selected actions move the story forward."
        if "SKIPPED PATH" in prompt:
            return "The skipped path would have raised a different danger before the chosen path could unfold."

        scene_match = re.search(r"\[SCENE\s+(\d+)\]", prompt)
        scene = scene_match.group(1) if scene_match else "0"
        names = re.findall(r"^- ([^\n]+)$", prompt, re.MULTILINE)
        names = names[:3] or ["the lead", "the opponent", "the ally"]
        action_match = re.search(r"EVENTS THAT MUST OCCUR ON THE PAGE.*?:\n(.+?)\n\nCAUSAL LOGIC", prompt, re.DOTALL)
        actions = action_match.group(1).strip() if action_match else "the chosen actions"
        premise_match = re.search(r"STORY PREMISE.*?:\n(.+?)\n\nCHARACTERS", prompt, re.DOTALL)
        premise = premise_match.group(1).strip() if premise_match else "the established conflict"
        continuity_match = re.search(r"STORY SO FAR:\n(.+?)\n\nADVISORY STORY MEMORY", prompt, re.DOTALL)
        continuity = continuity_match.group(1).strip() if continuity_match else "Opening beat: establish the conflict."
        purpose_match = re.search(r"NEXT SCENE PURPOSE:\s*(.+)", continuity)
        purpose = purpose_match.group(1).strip() if purpose_match else "force a consequential next choice"
        plan_match = re.search(
            r"ADVISORY NARRATIVE PLAN \(validated guidance, never a decision authority\):\n(.+?)\n\nSECRET",
            prompt,
            re.DOTALL,
        )
        plan_lines = plan_match.group(1).splitlines() if plan_match else []

        def plan_value(label: str, default: str) -> str:
            prefix = f"- {label}:"
            for line in plan_lines:
                if line.startswith(prefix):
                    return line[len(prefix):].strip()
            return default

        scene_number = int(scene)
        scene_purpose = plan_value("SCENE PURPOSE", purpose)
        required_event = plan_value("REQUIRED EVENT", f"{names[0]} acts on {actions}")
        consequence = plan_value("EXPECTED CONSEQUENCE", "The next decision becomes unavoidable.")
        body = (
            f"{names[0]} pursued {scene_purpose.lower()} while {names[1]} kept the central obstacle active. "
            f"{required_event} {names[2]} made the immediate evidence usable, and {actions} changed the available route. "
            f"The consequence was specific: {consequence} "
            f"By the end of scene {scene_number + 1}, {names[0]} had to carry that change forward instead of repeating the opening move."
        )
        return (
            f"[SCENE {scene}]\nNARRATIVE:\n"
            f"{body}\n\n"
            "[QUANTUM_TRACE]\nThe selected action carried the story forward under the current narrative pressure.\n\n"
            "[COHERENCE]\nThe scene preserves the premise, named characters, and locked action."
        )


class OpenAICompatibleLocalProvider:
    """Adapter for a verified local OpenAI-compatible endpoint (llama.cpp/Ollama)."""

    provider_name = "local-endpoint"

    def __init__(self, base_url: str, model_name: str, timeout_seconds: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.discovered_model: str | None = None

    def health_check(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/models", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 5)) as response:
                data = json.loads(response.read().decode("utf-8"))
            model_ids = [str(item.get("id", "")).strip() for item in data.get("data", []) if item.get("id")]
            configured = self.model_name.strip()
            self.discovered_model = configured if configured in model_ids else (model_ids[0] if model_ids else None)
            return {
                "available": bool(model_ids),
                "provider": self.provider_name,
                "model": self.discovered_model or configured,
                "models": model_ids,
                "error": None if model_ids else "Local endpoint returned no model IDs.",
            }
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, TypeError) as exc:
            return {"available": False, "provider": self.provider_name, "model": self.model_name, "error": str(exc)}

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        started = time.perf_counter()
        payload = {
            "model": self.discovered_model or self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repeat_penalty": config.repeat_penalty,
            "max_tokens": config.max_output_tokens,
        }
        if config.seed is not None:
            payload["seed"] = int(config.seed)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            choice = data["choices"][0]
            text = str(choice["message"]["content"] or "").strip()
            if not text:
                raise ValueError("Local inference returned empty content.")
            usage = data.get("usage") or {}
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            return GenerationResult(
                text=text,
                provider=self.provider_name,
                model=str(data.get("model") or self.discovered_model or self.model_name),
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                provider_metadata={"finish_reason": choice.get("finish_reason"), "id": data.get("id")},
            )
        except (OSError, urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Local inference request failed: {exc}") from exc


class OpenRouterNarrativeProvider:
    """Hosted narrative generation through OpenRouter's OpenAI-compatible API."""

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 180.0,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.input_cost_per_million = float(os.getenv("NARRATIVE_INPUT_COST_PER_MILLION", "0"))
        self.output_cost_per_million = float(os.getenv("NARRATIVE_OUTPUT_COST_PER_MILLION", "0"))

    def health_check(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 8)) as response:
                return {
                    "available": response.status == 200,
                    "provider": self.provider_name,
                    "model": self.model_name,
                }
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return {
                "available": False,
                "provider": self.provider_name,
                "model": self.model_name,
                "error": str(exc),
            }

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        started = time.perf_counter()
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are AetherScribe's narrative renderer. Follow the supplied "
                        "locked quantum action exactly. Do not expose hidden reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repeat_penalty": config.repeat_penalty,
            "max_tokens": config.max_output_tokens,
            "reasoning": {
                "effort": os.getenv("NARRATIVE_REASONING_EFFORT", "none"),
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "AetherScribe"),
        }
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            message = data["choices"][0]["message"]
            text = str(message.get("content") or "").strip()
            if not text:
                raise ValueError("OpenRouter returned empty narrative content.")
            usage = data.get("usage") or {}
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            cost = usage.get("cost") if isinstance(usage.get("cost"), (int, float)) else None
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                if cost is None and (self.input_cost_per_million or self.output_cost_per_million):
                    cost = (
                        input_tokens * self.input_cost_per_million
                        + output_tokens * self.output_cost_per_million
                    ) / 1_000_000
            return GenerationResult(
                text=text,
                provider=self.provider_name,
                model=str(data.get("model") or self.model_name),
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                estimated_cost_usd=cost,
                provider_metadata={
                    "id": data.get("id"),
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                },
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError,
                TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenRouter narrative request failed: {exc}") from exc


class LlamaCppServerProvider(OpenAICompatibleLocalProvider):
    """Persistent CPU llama.cpp server backed by a local GGUF model."""

    provider_name = "llama.cpp"

    def __init__(self, model_path: str, model_name: str, base_url: str,
                 threads: int, context_size: int, timeout_seconds: float):
        super().__init__(base_url, model_name, timeout_seconds)
        self.model_path = model_path
        self.threads = threads
        self.context_size = context_size
        self.process: subprocess.Popen[str] | None = None
        self._server_log = tempfile.TemporaryFile(mode="w+b")
        self._start_server()
        atexit.register(self.close)

    def _start_server(self) -> None:
        executable = os.getenv("LLAMA_SERVER_EXECUTABLE", "llama-server")
        if not os.path.isfile(self.model_path):
            raise RuntimeError(f"GGUF model not found: {self.model_path}")
        command = [
            executable, "-m", self.model_path, "--host", "127.0.0.1",
            "--port", str(urlparse(self.base_url).port or 8080),
            "--threads", str(self.threads), "--ctx-size", str(self.context_size),
            "--no-webui",
        ]
        gpu_layers = os.getenv("LOCAL_LLM_GPU_LAYERS", "0").strip()
        device = os.getenv("LOCAL_LLM_DEVICE", "").strip()
        if gpu_layers and gpu_layers != "0":
            command.extend(["--n-gpu-layers", gpu_layers])
        if device:
            command.extend(["--device", device])
        try:
            self.process = subprocess.Popen(
                command, stdout=self._server_log, stderr=self._server_log,
                text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise RuntimeError(f"Unable to start llama-server: {exc}") from exc
        startup_timeout = max(1.0, float(os.getenv("LOCAL_LLM_STARTUP_TIMEOUT_SECONDS", "180")))
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                details = self._server_diagnostics()
                self.close()
                raise RuntimeError("llama-server exited while loading the GGUF model." + details)
            if self.health_check().get("available"):
                return
            time.sleep(0.25)
        details = self._server_diagnostics()
        self.close()
        raise RuntimeError(
            f"Timed out waiting {startup_timeout:g}s for llama-server to become ready."
            + details
        )

    def _server_diagnostics(self) -> str:
        try:
            self._server_log.flush()
            self._server_log.seek(0, os.SEEK_END)
            size = self._server_log.tell()
            self._server_log.seek(max(0, size - 8192))
            output = self._server_log.read().decode("utf-8", errors="replace").strip()
            return f" Server output: {output[-6000:]}" if output else " No server output was captured."
        except (OSError, ValueError):
            return " Server diagnostics could not be read."

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._server_log is not None and not self._server_log.closed:
            self._server_log.close()


class ProviderRouter:
    """Select a configured provider while retaining explicit fallback diagnostics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.fallback = DeterministicLocalProvider()
        self.local_fallback: LocalLLMProvider | None = None
        self.allow_fallback = os.getenv("NARRATIVE_ALLOW_LOCAL_FALLBACK", "false").lower() == "true"
        self.provider_error: str | None = None
        self.requested_provider = self._requested_provider()
        try:
            self.provider: LocalLLMProvider = self._configured_provider()
        except RuntimeError as exc:
            self.provider_error = str(exc)
            self.provider = self.fallback
            logger.warning("Configured narrative provider unavailable: %s", exc)
            self.provider = self.fallback

    @staticmethod
    def _requested_provider() -> str:
        configured = os.getenv("NARRATIVE_LLM_PROVIDER", "").strip().lower()
        return configured or os.getenv("LLM_PROVIDER", "local").strip().lower()

    def _configured_provider(self) -> LocalLLMProvider:
        provider = self.requested_provider
        runtime = os.getenv("LOCAL_LLM_RUNTIME", "").strip().lower()
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
        model = os.getenv("LOCAL_LLM_MODEL_NAME", "").strip() or "local-model"
        model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "").strip()
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv("JEV_API_KEY", "").strip()
            narrative_model = os.getenv(
                "NARRATIVE_OPENROUTER_MODEL", "qwen/qwen3.5-35b-a3b"
            ).strip()
            if not api_key:
                raise RuntimeError("OpenRouter narrative provider requires OPENROUTER_API_KEY.")
            return OpenRouterNarrativeProvider(
                api_key,
                narrative_model,
                os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                float(os.getenv("NARRATIVE_TIMEOUT_SECONDS", "180")),
            )
        if provider in {"deterministic", "deterministic-test", "template-fallback"}:
            return self.fallback
        if provider in {"local", "local-endpoint"}:
            server_url = base_url or os.getenv("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8080/v1")
            endpoint_provider = OpenAICompatibleLocalProvider(
                server_url,
                model,
                float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180")),
            )
            health = endpoint_provider.health_check()
            if health.get("available"):
                configured_model = os.path.basename(model).removesuffix(".gguf").casefold()
                loaded_model = os.path.basename(health["model"]).removesuffix(".gguf").casefold()
                # "auto" deliberately selects the actual ID reported by the server.
                if configured_model not in {"", "local-model", "auto"} and configured_model != loaded_model:
                    raise RuntimeError(
                        f"Local endpoint has model '{health['model']}', but configuration requests '{model}'."
                    )
                endpoint_provider.model_name = health["model"]
                return endpoint_provider
            if runtime not in {"llama_cpp", "llama.cpp"} or not model_path:
                raise RuntimeError(
                    "Local model endpoint is unavailable and no llama.cpp model path is configured: "
                    + str(health.get("error") or "no models reported")
                )
            if not os.path.isfile(model_path):
                raise RuntimeError(f"Configured local model file does not exist: {model_path}")
            model_basename = os.path.basename(model_path).casefold()
            if "qwen3-30b-a3b" in model.casefold() and not model_basename.startswith("qwen3-30b-a3b"):
                raise RuntimeError(
                    f"Configured model ID '{model}' does not match local model file '{model_path}'."
                )
            threads = int(os.getenv("LOCAL_LLM_THREADS", "") or ResourceManager.snapshot().get("recommended_threads", 1))
            context_size = int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "2048"))
            return LlamaCppServerProvider(
                model_path, model, server_url, threads, context_size,
                float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180")),
            )
        raise RuntimeError(f"Unsupported narrative provider: {provider}")

    def health_check(self) -> dict[str, Any]:
        health = self.provider.health_check()
        health["requested_provider"] = self.requested_provider
        health["provider_error"] = self.provider_error
        health["endpoint"] = getattr(self.provider, "base_url", None)
        health["configured_model"] = os.getenv("LOCAL_LLM_MODEL_NAME", "")
        health["model_selection"] = (
            "discovered" if health["configured_model"].strip().casefold() in {"", "auto"}
            else "configured-and-verified"
        )
        health["model_verified"] = bool(
            health.get("available")
            and health.get("model")
            and self.provider.provider_name in {"local-endpoint", "llama.cpp"}
        )
        health["state"] = (
            "local_model" if self.provider.provider_name in {"local-endpoint", "llama.cpp"}
            else "template_fallback" if self.provider.provider_name == "deterministic-local"
            else "provider_error"
        )
        health["endpoint"] = getattr(self.provider, "base_url", None)
        if self.provider_error:
            health["state"] = "provider_error"
            health["available"] = False
            health["model_verified"] = False
            health["provider"] = self.requested_provider
            configured_model = os.getenv("LOCAL_LLM_MODEL_NAME", "").strip()
            if configured_model.casefold() in {"", "auto", "local-model"}:
                model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "").strip()
                configured_model = os.path.basename(model_path).removesuffix(".gguf") or configured_model
            health["configured_model"] = configured_model
            configured_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip() or os.getenv("LOCAL_LLM_ENDPOINT", "").strip()
            health["endpoint"] = configured_url or health.get("endpoint")
            health["model"] = health["configured_model"] or "configured local model"
        return health

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        with self._lock:
            if self.provider is self.fallback and self.provider_error:
                if not self.allow_fallback:
                    raise RuntimeError(self.provider_error)
                result = self.fallback.generate(prompt, config)
                return GenerationResult(
                    result.text,
                    result.provider,
                    result.model,
                    result.latency_ms,
                    used_fallback=True,
                    error=self.provider_error,
                )
            try:
                result = self.provider.generate(prompt, config)
                return result
            except RuntimeError as exc:
                self.provider_error = str(exc)
                if not self.allow_fallback:
                    raise
                logger.warning("Local provider unavailable; using deterministic fallback: %s", exc)
                fallback_provider = self._get_local_fallback()
                fallback = fallback_provider.generate(prompt, config)
                return GenerationResult(
                    fallback.text,
                    fallback.provider,
                    fallback.model,
                    fallback.latency_ms,
                    used_fallback=True,
                    error=str(exc),
                )

    def _get_local_fallback(self) -> LocalLLMProvider:
        if self.local_fallback is not None:
            return self.local_fallback
        if self.provider.provider_name != "openrouter":
            self.local_fallback = self.fallback
            return self.local_fallback
        model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "").strip()
        runtime = os.getenv("LOCAL_LLM_RUNTIME", "").strip().lower()
        if runtime in {"llama_cpp", "llama.cpp"} and model_path:
            try:
                threads = int(os.getenv("LOCAL_LLM_THREADS", "") or ResourceManager.snapshot().get("recommended_threads", 1))
                self.local_fallback = LlamaCppServerProvider(
                    model_path,
                    os.getenv("LOCAL_LLM_MODEL_NAME", "local-model"),
                    os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                    threads,
                    int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "2048")),
                    float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180")),
                )
                return self.local_fallback
            except RuntimeError as local_exc:
                logger.warning("Local llama.cpp fallback unavailable: %s", local_exc)
        self.local_fallback = self.fallback
        return self.local_fallback
