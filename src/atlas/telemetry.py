"""Token usage extraction, aggregation, and Markdown reporting."""

from __future__ import annotations

from typing import Any

import litellm

from atlas.schemas import ModelCallRecord, TokenUsageSummary, UsageBucket


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int = 0) -> int:
    parsed = _opt_int(value)
    return parsed if parsed is not None else default


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_usage_metrics(response: Any) -> dict[str, Any]:
    """Safely extract token counts and estimated USD cost from a LiteLLM response."""
    usage = _get_field(response, "usage")
    raw_prompt = _opt_int(_get_field(usage, "prompt_tokens"))
    if raw_prompt is None:
        raw_prompt = _opt_int(_get_field(usage, "input_tokens"))
    raw_completion = _opt_int(_get_field(usage, "completion_tokens"))
    if raw_completion is None:
        raw_completion = _opt_int(_get_field(usage, "output_tokens"))
    raw_total = _opt_int(_get_field(usage, "total_tokens"))

    usage_reported = any(v is not None for v in (raw_prompt, raw_completion, raw_total))
    prompt_tokens = raw_prompt if raw_prompt is not None else 0
    completion_tokens = raw_completion if raw_completion is not None else 0
    total_tokens = raw_total if raw_total is not None else prompt_tokens + completion_tokens

    prompt_details = _get_field(usage, "prompt_tokens_details")
    cached_raw = _get_field(prompt_details, "cached_tokens")
    if cached_raw is None:
        cached_raw = _get_field(usage, "cache_read_input_tokens")
    cached_tokens = _as_int(cached_raw, 0)

    completion_details = _get_field(usage, "completion_tokens_details")
    reasoning_raw = _get_field(completion_details, "reasoning_tokens")
    reasoning_tokens = _as_int(reasoning_raw, 0)

    cost_usd: float | None = None
    try:
        computed = litellm.completion_cost(completion_response=response)
        parsed_cost = _as_float(computed)
        if parsed_cost is not None:
            cost_usd = round(parsed_cost, 6)
    except Exception:
        cost_usd = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "usage_reported": usage_reported,
    }


def _normalize_call(raw: dict[str, Any]) -> ModelCallRecord:
    raw_prompt = _opt_int(raw.get("prompt_tokens"))
    raw_completion = _opt_int(raw.get("completion_tokens"))
    raw_total = _opt_int(raw.get("total_tokens"))
    explicit_usage = raw.get("usage_reported")
    if isinstance(explicit_usage, bool):
        usage_reported = explicit_usage
    else:
        usage_reported = any(v is not None for v in (raw_prompt, raw_completion, raw_total))

    prompt_tokens = raw_prompt if raw_prompt is not None else 0
    completion_tokens = raw_completion if raw_completion is not None else 0
    total_tokens = raw_total if raw_total is not None else prompt_tokens + completion_tokens
    cost_usd = _as_float(raw.get("cost_usd"))
    if cost_usd is not None:
        cost_usd = round(cost_usd, 6)

    return ModelCallRecord(
        role=str(raw.get("role") or "unknown"),
        requested_model=str(raw.get("requested_model") or ""),
        used_model=str(raw.get("used_model") or raw.get("requested_model") or "unknown"),
        attempts=_as_int(raw.get("attempts"), 1),
        elapsed_ms=round(float(raw.get("elapsed_ms") or 0.0), 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=_as_int(raw.get("cached_tokens"), 0),
        reasoning_tokens=_as_int(raw.get("reasoning_tokens"), 0),
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        usage_reported=usage_reported,
        timestamp=raw.get("timestamp"),
        error=str(raw["error"]) if raw.get("error") else None,
    )


def _aggregate_buckets(
    records: list[ModelCallRecord], key_fn: Any
) -> list[UsageBucket]:
    grouped: dict[str, list[ModelCallRecord]] = {}
    for rec in records:
        grouped.setdefault(key_fn(rec), []).append(rec)

    buckets: list[UsageBucket] = []
    for name, items in grouped.items():
        costs = [r.cost_usd for r in items if r.cost_usd is not None]
        buckets.append(
            UsageBucket(
                name=name,
                calls=len(items),
                prompt_tokens=sum(r.prompt_tokens for r in items),
                completion_tokens=sum(r.completion_tokens for r in items),
                cached_tokens=sum(r.cached_tokens for r in items),
                reasoning_tokens=sum(r.reasoning_tokens for r in items),
                total_tokens=sum(r.total_tokens for r in items),
                cost_usd=round(sum(costs), 6) if costs else None,
                elapsed_ms=round(sum(r.elapsed_ms for r in items), 1),
            )
        )
    return buckets


def summarize_usage(call_log: list[dict[str, Any]]) -> TokenUsageSummary:
    """Aggregate token, cost, and latency metrics across model calls."""
    records = [_normalize_call(entry) for entry in call_log]
    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    return TokenUsageSummary(
        total_calls=len(records),
        prompt_tokens=sum(r.prompt_tokens for r in records),
        completion_tokens=sum(r.completion_tokens for r in records),
        cached_tokens=sum(r.cached_tokens for r in records),
        reasoning_tokens=sum(r.reasoning_tokens for r in records),
        total_tokens=sum(r.total_tokens for r in records),
        cost_usd=round(sum(costs), 6) if costs else None,
        elapsed_ms=round(sum(r.elapsed_ms for r in records), 1),
        unreported_usage_calls=sum(1 for r in records if not r.usage_reported),
        failed_calls=sum(1 for r in records if r.error),
        by_stage=_aggregate_buckets(records, lambda r: r.role),
        by_model=_aggregate_buckets(records, lambda r: r.used_model),
        calls=records,
    )


def _fmt_cost(cost: float | None) -> str:
    return f"${cost:.6f}" if cost is not None else "n/a"


def render_token_report_md(summary: TokenUsageSummary) -> str:
    """Render a Markdown token and cost breakdown suitable for CLI, reports, and GitHub PRs."""
    lines = [
        "## Token Usage & Cost",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Total LLM calls | {summary.total_calls} |",
        f"| Prompt tokens | {summary.prompt_tokens:,} |",
        f"| Completion tokens | {summary.completion_tokens:,} |",
        f"| Cached tokens | {summary.cached_tokens:,} |",
        f"| Reasoning tokens | {summary.reasoning_tokens:,} |",
        f"| **Total tokens** | **{summary.total_tokens:,}** |",
        f"| Estimated cost (USD) | {_fmt_cost(summary.cost_usd)} |",
        f"| Total LLM latency | {summary.elapsed_ms:,.1f} ms |",
    ]

    if summary.by_stage:
        lines.extend([
            "",
            "### By stage",
            "",
            "| Stage | Calls | Prompt | Completion | Total | Est. Cost | Time |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for bucket in summary.by_stage:
            lines.append(
                f"| `{bucket.name}` | {bucket.calls} | {bucket.prompt_tokens:,} | "
                f"{bucket.completion_tokens:,} | {bucket.total_tokens:,} | "
                f"{_fmt_cost(bucket.cost_usd)} | {bucket.elapsed_ms:,.1f} ms |"
            )

    if summary.by_model:
        lines.extend([
            "",
            "### By model",
            "",
            "| Model | Calls | Prompt | Completion | Total | Est. Cost |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for bucket in summary.by_model:
            lines.append(
                f"| `{bucket.name}` | {bucket.calls} | {bucket.prompt_tokens:,} | "
                f"{bucket.completion_tokens:,} | {bucket.total_tokens:,} | "
                f"{_fmt_cost(bucket.cost_usd)} |"
            )

    if summary.failed_calls > 0:
        lines.extend([
            "",
            f"> Note: {summary.failed_calls} call(s) returned an unusable response (empty, or "
            "cut off at max_tokens). Providers usually bill these, so their tokens are included.",
        ])

    if summary.unreported_usage_calls > 0:
        lines.extend([
            "",
            f"> Note: {summary.unreported_usage_calls} call(s) did not report "
            "token usage metadata.",
        ])

    return "\n".join(lines)
