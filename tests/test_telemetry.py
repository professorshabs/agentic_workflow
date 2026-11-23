from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas.router import ModelRouter, TokenBudgetExceededError
from atlas.telemetry import extract_usage_metrics, render_token_report_md, summarize_usage


def test_extract_usage_metrics_handles_standard_cached_dict_and_missing(monkeypatch):
    monkeypatch.setattr("atlas.telemetry.litellm.completion_cost", lambda **_: 0.00123456)

    obj_resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=45,
            total_tokens=165,
            prompt_tokens_details=SimpleNamespace(cached_tokens=30),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=15),
        )
    )
    m1 = extract_usage_metrics(obj_resp)
    assert m1["prompt_tokens"] == 120
    assert m1["completion_tokens"] == 45
    assert m1["cached_tokens"] == 30
    assert m1["reasoning_tokens"] == 15
    assert m1["total_tokens"] == 165
    assert m1["cost_usd"] == 0.001235
    assert m1["usage_reported"] is True

    dict_resp = {
        "usage": {
            "input_tokens": 50,
            "output_tokens": 25,
            "cache_read_input_tokens": 10,
        }
    }
    m2 = extract_usage_metrics(dict_resp)
    assert m2["prompt_tokens"] == 50
    assert m2["completion_tokens"] == 25
    assert m2["cached_tokens"] == 10
    assert m2["total_tokens"] == 75
    assert m2["usage_reported"] is True

    monkeypatch.setattr(
        "atlas.telemetry.litellm.completion_cost",
        lambda **_: (_ for _ in ()).throw(RuntimeError("unknown model")),
    )
    empty_resp = SimpleNamespace(usage=None)
    m3 = extract_usage_metrics(empty_resp)
    assert m3["prompt_tokens"] == 0
    assert m3["completion_tokens"] == 0
    assert m3["total_tokens"] == 0
    assert m3["cost_usd"] is None
    assert m3["usage_reported"] is False


def test_summarize_usage_and_render_markdown():
    call_log = [
        {
            "role": "parser",
            "requested_model": "m-a",
            "used_model": "m-a",
            "elapsed_ms": 100.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.001,
        },
        {
            "role": "builder",
            "requested_model": "m-b",
            "used_model": "m-b",
            "elapsed_ms": 250.5,
            "prompt_tokens": 300,
            "completion_tokens": 200,
            "cached_tokens": 40,
            "reasoning_tokens": 20,
            "cost_usd": 0.004,
        },
        {
            "role": "fixer",
            "requested_model": "m-b",
            "used_model": "m-b",
            "elapsed_ms": 80.0,
        },
    ]
    summary = summarize_usage(call_log)
    assert summary.total_calls == 3
    assert summary.prompt_tokens == 400
    assert summary.completion_tokens == 250
    assert summary.cached_tokens == 40
    assert summary.reasoning_tokens == 20
    assert summary.total_tokens == 650
    assert summary.cost_usd == 0.005
    assert summary.unreported_usage_calls == 1
    assert [b.name for b in summary.by_stage] == ["parser", "builder", "fixer"]
    assert [b.name for b in summary.by_model] == ["m-a", "m-b"]

    md = render_token_report_md(summary)
    assert "## Token Usage & Cost" in md
    assert "| `parser` | 1 | 100 | 50 | 150 |" in md
    assert "| `m-b` | 2 | 300 | 200 | 500 |" in md
    assert "1 call(s) did not report token usage metadata" in md


async def test_router_enforces_token_budget_cap(tmp_path, monkeypatch):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
roles:
  parser: [test-model]
limits:
  max_parallel_calls: 2
  max_tokens: 100
  timeout_seconds: 10
  attempts_per_model: 1
  max_total_tokens: 50
""",
        encoding="utf-8",
    )
    router = ModelRouter(cfg)

    async def fake_acompletion(*, model, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=25, total_tokens=55),
        )

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=fake_acompletion))
    monkeypatch.setattr("atlas.telemetry.litellm.completion_cost", lambda **_: 0.0005)

    await router.complete("test-model", "sys", "usr", role="parser")
    assert router.usage_summary().total_tokens == 55

    with pytest.raises(TokenBudgetExceededError, match="Token budget exceeded"):
        await router.complete("test-model", "sys", "usr", role="parser")


def test_summarize_usage_empty_and_non_numeric_values():
    empty = summarize_usage([])
    assert empty.total_calls == 0
    assert empty.total_tokens == 0
    assert empty.cost_usd is None
    md_empty = render_token_report_md(empty)
    assert "| Estimated cost (USD) | n/a |" in md_empty

    dirty = summarize_usage(
        [
            {
                "role": "parser",
                "used_model": "m",
                "prompt_tokens": True,
                "completion_tokens": "bad",
                "cost_usd": "not-a-float",
            }
        ]
    )
    assert dirty.prompt_tokens == 0
    assert dirty.completion_tokens == 0
    assert dirty.cost_usd is None
    assert dirty.unreported_usage_calls == 1


def test_extract_usage_metrics_prefers_direct_provider_cost(monkeypatch):
    monkeypatch.setattr(
        "atlas.telemetry.litellm.completion_cost",
        lambda **_: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=14, completion_tokens=4, total_tokens=18, cost=0.00016)
    )
    metrics = extract_usage_metrics(resp)
    assert metrics["total_tokens"] == 18
    assert metrics["cost_usd"] == 0.00016
