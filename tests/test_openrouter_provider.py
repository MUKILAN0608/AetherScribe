"""Contract test for the OpenRouter response parser without network access."""

import json
from unittest.mock import patch

from decision.providers import DecisionRequest, JevDecisionProvider


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps({
            "choices": [{
                "message": {
                    "content": '{"selected":"second","approved":true,"rationale":"Fits the brief."}'
                }
            }]
        }).encode()


def test_openrouter_response_is_validated():
    with patch.dict("os.environ", {"JEV_PROVIDER": "openrouter"}), patch(
        "urllib.request.urlopen", return_value=FakeResponse()
    ):
        provider = JevDecisionProvider(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key="test-key",
            model="test/model",
        )
        result = provider.evaluate(DecisionRequest("choose", ("first", "second")))
    assert result.selected == "second"
    assert result.approved is True
