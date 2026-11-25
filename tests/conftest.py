"""Test-wide isolation from the developer's real credentials.

LiteLLM calls ``load_dotenv()`` when it is imported in its default ``DEV`` mode, which pulled
the repo's real ``.env`` (OpenRouter / Anthropic / Gemini keys) into the test process: a test
that forgot to mock a provider could then spend real money, and adapter tests behaved
differently depending on import order. Production mode skips that load, and every test
starts with provider credentials removed (tests that need one set a fake value).
"""

from __future__ import annotations

import os

import pytest

os.environ["LITELLM_MODE"] = "PRODUCTION"

_CREDENTIAL_PREFIXES = (
    "OPENROUTER_",
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "GOOGLE_API_KEY",
    "ANTIGRAVITY_",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
_CREDENTIAL_MARKERS = ("_API_KEY", "_TOKEN", "_SECRET")


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(_CREDENTIAL_PREFIXES) or any(m in key for m in _CREDENTIAL_MARKERS):
            monkeypatch.delenv(key, raising=False)
