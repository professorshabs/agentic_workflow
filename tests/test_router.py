from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas.router import ModelRouter


def _write_cfg(tmp_path, content: str):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def test_router_validates_limits_and_candidates(tmp_path):
    bad_cfg = _write_cfg(
        tmp_path,
        """
roles:
  parser: [model-a]
limits:
  max_parallel_calls: 2
  max_tokens: 100
  timeout_seconds: 0
  attempts_per_model: 1
""",
    )
    with pytest.raises(ValueError, match="timeout_seconds"):
        ModelRouter(bad_cfg)

    good_cfg = _write_cfg(
        tmp_path,
        """
roles:
  parser: [model-a]
role_fallbacks:
  parser: [model-b]
fallbacks: [model-c]
limits:
  max_parallel_calls: 2
  max_tokens: 100
  timeout_seconds: 10
  attempts_per_model: 1
""",
    )
    router = ModelRouter(good_cfg)
    assert router.candidates_for("model-a", role="parser") == ["model-a", "model-b", "model-c"]


async def test_router_falls_back_and_records_telemetry(tmp_path, monkeypatch):
    cfg = _write_cfg(
        tmp_path,
        """
roles:
  parser: [primary-model]
role_fallbacks:
  parser: [fallback-model]
fallbacks: []
limits:
  max_parallel_calls: 2
  max_tokens: 100
  timeout_seconds: 10
  attempts_per_model: 1
""",
    )
    router = ModelRouter(cfg)

    async def fake_acompletion(*, model, **kwargs):
        if model == "primary-model":
            raise RuntimeError("429 rate limited")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=fake_acompletion))

    used, text = await router.complete("primary-model", "sys", "usr", role="parser")
    assert used == "fallback-model"
    assert text == '{"ok": true}'
    assert len(router.call_log) == 1
    assert router.call_log[0]["used_model"] == "fallback-model"
    assert router.call_log[0]["attempts"] == 2
