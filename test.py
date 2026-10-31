"""Offline smoke test for the configured local narrative provider."""

from llm.providers import GenerationConfig, ProviderRouter


if __name__ == "__main__":
    router = ProviderRouter()
    print(router.health_check())
    result = router.generate("Write one sentence about a locked quantum action.", GenerationConfig(max_output_tokens=32))
    assert result.text
    print(result.text)