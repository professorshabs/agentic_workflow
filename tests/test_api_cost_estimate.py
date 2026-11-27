"""API-equivalent cost: what subscription / login calls would cost on pay-per-token APIs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atlas.harnesses.claude_code import _claude_usage
from atlas.router import ModelRouter
from atlas.runner import _fmt_usage_brief
from atlas.telemetry import (
    api_equivalent_cost,
    price_per_token,
    render_token_report_md,
    summarize_usage,
)

REPO = Path(__file__).resolve().parents[1]
PRICES = {
    "sonnet": {"input": 2.0, "output": 10.0, "cache_read": 0.1},
    "gemini-4-neon": {"input": 3.0, "output": 15.0, "cache_read": 0.15},
}


def _call(**kw):
    base = {"role": "builder", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cached_tokens": 0, "cost_usd": None, "cost_source": "unavailable"}
    base.update(kw)
    return base


def test_billed_calls_use_their_real_cost():
    assert api_equivalent_cost(_call(cost_usd=0.214107, cost_source="reported"), PRICES) == (
        0.214107, "billed"
    )


def test_local_calls_are_free():
    assert api_equivalent_cost(_call(cost_usd=0.0, cost_source="local_free"), PRICES) == (
        0.0, "local"
    )


def test_claude_codes_own_figure_comes_first():
    call = _call(used_model="sonnet", cost_source="subscription", api_cost_usd=0.5,
                 prompt_tokens=1000, completion_tokens=10, total_tokens=1010)
    assert api_equivalent_cost(call, PRICES) == (0.5, "harness")


def test_price_list_uses_cache_and_output_prices_and_strips_effort_suffix():
    # agy reports `gemini-4-neon-high`; the table is keyed by `gemini-4-neon`.
    call = _call(used_model="gemini-4-neon-high", cost_source="unavailable",
                 prompt_tokens=40_000, cached_tokens=10_000, completion_tokens=20_000,
                 total_tokens=60_000)
    est, src = api_equivalent_cost(call, PRICES)
    assert src == "price_list"
    assert est == pytest.approx(30_000 * 3e-6 + 10_000 * 0.15e-6 + 20_000 * 15e-6)


def test_thinking_tokens_outside_completion_are_priced_as_output():
    call = _call(used_model="sonnet", cost_source="subscription",
                 prompt_tokens=1_000, completion_tokens=100, total_tokens=1_600)
    est, _ = api_equivalent_cost(call, PRICES)
    assert est == pytest.approx(1_000 * 2e-6 + 600 * 10e-6)


def test_litellm_price_map_fills_models_missing_from_the_config():
    price = price_per_token("gemini-4-hydrogen-high", {})
    assert price is not None and price["input"] > 0 and price["output"] > price["input"]


def test_unknown_models_and_calls_without_tokens_have_no_estimate():
    assert api_equivalent_cost(
        _call(used_model="mystery-model-9", prompt_tokens=10, total_tokens=10), PRICES
    ) == (None, None)
    assert api_equivalent_cost(_call(used_model="sonnet"), PRICES) == (None, None)


def test_claude_code_keeps_its_api_price_only_as_an_estimate_on_a_subscription():
    parsed = {"total_cost_usd": 0.4321, "usage": {"input_tokens": 10, "output_tokens": 5}}
    sub = _claude_usage(parsed, auth_mode="subscription")
    assert sub["cost_usd"] is None and sub["cost_source"] == "subscription"
    assert sub["api_cost_usd"] == 0.4321
    api = _claude_usage(parsed, auth_mode="api_key")
    assert api["cost_usd"] == 0.4321 and api["api_cost_usd"] is None


def _router(tmp_path: Path, pricing: str = "") -> ModelRouter:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "roles:\n  builder: [openrouter/m]\n"
        "limits:\n  max_parallel_calls: 1\n  max_tokens: 100\n  timeout_seconds: 5\n" + pricing,
        encoding="utf-8",
    )
    return ModelRouter(cfg)


def test_router_records_the_estimate_and_summaries_add_it_up(tmp_path):
    router = _router(tmp_path, "pricing:\n  sonnet: {input: 2, output: 10}\n")
    router.record_harness_call(role="builder", requested_model="sonnet", used_model="sonnet",
                               harness="claude_code", harness_version=None, elapsed_ms=1.0,
                               prompt_tokens=1_000_000, completion_tokens=0,
                               total_tokens=1_000_000, cost_source="subscription")
    router.record_harness_call(role="supervisor", requested_model="x", used_model="mystery",
                               harness="agy", harness_version=None, elapsed_ms=1.0,
                               prompt_tokens=5, total_tokens=5)
    first, second = router.call_log
    assert (first["api_cost_usd"], first["api_cost_source"]) == (2.0, "price_list")
    assert second["api_cost_usd"] is None
    summary = summarize_usage(router.call_log)
    assert summary.api_cost_usd == 2.0 and summary.api_unestimated_calls == 1
    report = render_token_report_md(summary)
    assert "API-equivalent cost (estimate, USD) | $2.000000" in report
    assert "1 call(s) had no price" in report
    assert "~$2.00 at API prices" in _fmt_usage_brief(summary)


@pytest.mark.parametrize(
    "pricing",
    [
        "pricing: [1, 2]\n",
        "pricing:\n  sonnet: {input: 2}\n",
        "pricing:\n  sonnet: {input: -1, output: 10}\n",
        "pricing:\n  sonnet: {input: 2, output: 10, cache_read: true}\n",
    ],
)
def test_bad_pricing_tables_are_rejected(tmp_path, pricing):
    with pytest.raises(ValueError, match="pricing"):
        _router(tmp_path, pricing)


def test_every_cloud_harness_model_in_the_shipped_config_has_a_price():
    cfg = yaml.safe_load((REPO / "config" / "harness.yaml").read_text(encoding="utf-8"))
    prices = cfg.get("pricing") or {}
    models: set[str] = set()

    def collect(node):
        if isinstance(node, dict):
            if node.get("type") in ("harness", None) and node.get("provider") in (
                "claude_code", "antigravity"
            ) and node.get("model"):
                models.add(str(node["model"]))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(cfg.get("backends") or {})
    assert models, "expected harness models in config/harness.yaml"
    missing = [m for m in sorted(models) if price_per_token(m, prices) is None]
    assert not missing, f"no API price for {missing}; add them under `pricing`"
