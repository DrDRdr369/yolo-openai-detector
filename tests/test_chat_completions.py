"""OpenAI compatibility facade tests.

Implements: PR-3.

Cover: a request built with the stock `openai` Python client (base_url + key swapped, image
attached as base64) succeeds and returns parseable detection JSON; streaming -> 400;
multiple/zero images -> 400; remote URL -> 400.
"""

import pytest


@pytest.mark.skip(reason="PR-3: implement stock-OpenAI-SDK integration test")
def test_openai_sdk_vision_request_succeeds():
    ...


@pytest.mark.skip(reason="PR-3: streaming must be rejected")
def test_streaming_returns_400():
    ...
