"""Native /v1/detections tests.

Implements: PR-1 (inference core) and PR-2 (endpoint).

Cover: golden-image detection (deterministic); oversized/corrupt/bad-base64 -> 400;
remote URL -> 400; no key -> 401; model not loaded -> 503; classes filter; threshold
overrides; statelessness (identical request -> identical response). Place golden images in
tests/fixtures/.
"""

import pytest


@pytest.mark.skip(reason="PR-1/PR-2: implement detection tests with a golden fixture image")
def test_golden_image_detection():
    ...


@pytest.mark.skip(reason="PR-2: implement fail-closed input validation tests")
def test_bad_input_returns_400():
    ...


@pytest.mark.skip(reason="PR-2: remote URL must be rejected")
def test_remote_url_returns_400():
    ...
