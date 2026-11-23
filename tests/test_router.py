from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest

from atlas.router import ModelRouter, TokenBudgetExceededError, resolve_default_config


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


def test_default_models_yaml_resolves_litellm_providers():
    router = ModelRouter()
    configured = [
        *(m for models in router.cfg["roles"].values() for m in models),
        *(m for models in router.cfg.get("role_fallbacks", {}).values() for m in models),
        *router.cfg.get("fallbacks", []),
    ]
    assert configured
    for model in configured:
        _, provider, _, _ = litellm.get_llm_provider(model)
        assert provider


def test_resolve_default_config_env_override_and_models_for_fallback(tmp_path, monkeypatch):
    cfg = _write_cfg(
        tmp_path,
        """
roles:
  judge: [m-judge]
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
""",
    )
    monkeypatch.setenv("ATLAS_CONFIG", str(cfg))
    assert resolve_default_config() == cfg

    router = ModelRouter()
    assert router.models_for("builder", fallback_role="judge") == ["m-judge"]
    with pytest.raises(ValueError, match="Unknown model role"):
        router.models_for("builder")


@pytest.mark.parametrize(
    "content,match",
    [
        ("just a string", "YAML mapping"),
        (
            "roles: {}\nlimits:\n  max_parallel_calls: 1\n"
            "  max_tokens: 10\n  timeout_seconds: 5\n",
            "at least one role",
        ),
        (
            "roles:\n  p: [m]\nlimits:\n  max_parallel_calls: 1\n  max_tokens: 10\n"
            "  timeout_seconds: 5\n  max_fix_rounds: -1\n",
            "max_fix_rounds",
        ),
        (
            "roles:\n  p: [m]\nlimits:\n  max_parallel_calls: 1\n  max_tokens: 10\n"
            "  timeout_seconds: 5\n  max_total_tokens: 0\n",
            "max_total_tokens",
        ),
        (
            "roles:\n  p: [m]\nlimits:\n  max_parallel_calls: 1\n  max_tokens: 10\n"
            "  timeout_seconds: 5\n  max_cost_usd: 0\n",
            "max_cost_usd",
        ),
    ],
)
def test_router_rejects_invalid_configs(tmp_path, content, match):
    cfg = _write_cfg(tmp_path, content)
    with pytest.raises(ValueError, match=match):
        ModelRouter(cfg)

    with pytest.raises(FileNotFoundError):
        ModelRouter(tmp_path / "missing.yaml")


async def test_router_cost_budget_and_all_models_failed(tmp_path, monkeypatch):
    cfg = _write_cfg(
        tmp_path,
        """
roles:
  parser: [m-1]
role_fallbacks:
  parser: [m-2]
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
  attempts_per_model: 1
  max_cost_usd: 0.01
""",
    )
    router = ModelRouter(cfg)

    async def empty_then_fail(*, model, **kwargs):
        if model == "m-1":
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
            )
        raise RuntimeError("boom")

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=empty_then_fail))
    with pytest.raises(RuntimeError, match="All configured models failed"):
        await router.complete("m-1", "sys", "usr", role="parser")

    router.call_log.append(
        {
            "role": "parser",
            "used_model": "m-1",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost_usd": 0.02,
        }
    )
    with pytest.raises(TokenBudgetExceededError, match="Cost budget exceeded"):
        await router.complete("m-1", "sys", "usr", role="parser")
