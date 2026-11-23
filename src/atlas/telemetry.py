"""Token usage extraction, aggregation, and Markdown reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
    direct_cost = _as_float(_get_field(usage, "cost"))
    if direct_cost is not None and direct_cost >= 0:
        cost_usd = round(direct_cost, 6)
    else:
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


# ---------------------------------------------------------------------------
# Project token ledger: one JSON line per Atlas run, committed with the project, so the
# total spend survives after run folders and CI artifacts are gone.

DEFAULT_LEDGER = "usage/ledger.jsonl"


def _bucket_map(buckets: list[UsageBucket]) -> dict[str, dict[str, Any]]:
    return {
        b.name: {"calls": b.calls, "total_tokens": b.total_tokens, "cost_usd": b.cost_usd}
        for b in buckets
    }


def ledger_entry(
    *,
    run_id: str,
    mode: str,
    spec_path: str,
    status: str,
    summary: TokenUsageSummary,
    created_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Compact, append-only record of one run's token use."""
    return {
        "run_id": run_id,
        "mode": mode,
        "spec_path": spec_path,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "calls": summary.total_calls,
        "prompt_tokens": summary.prompt_tokens,
        "completion_tokens": summary.completion_tokens,
        "cached_tokens": summary.cached_tokens,
        "reasoning_tokens": summary.reasoning_tokens,
        "total_tokens": summary.total_tokens,
        "cost_usd": summary.cost_usd,
        "unreported_usage_calls": summary.unreported_usage_calls,
        "failed_calls": summary.failed_calls,
        "by_stage": _bucket_map(summary.by_stage),
        "by_model": _bucket_map(summary.by_model),
    }


def append_ledger(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Read ledger lines; skips blank or broken lines (for example after a bad merge)."""
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def _sum_group(entries: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for e in entries:
        g = grouped.setdefault(
            str(e.get(key) or "-"), {"runs": 0, "calls": 0, "total_tokens": 0, "cost_usd": None}
        )
        g["runs"] += 1
        g["calls"] += _as_int(e.get("calls"))
        g["total_tokens"] += _as_int(e.get("total_tokens"))
        cost = _as_float(e.get("cost_usd"))
        if cost is not None:
            g["cost_usd"] = (g["cost_usd"] or 0.0) + cost
    return grouped


def _sum_nested(entries: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for e in entries:
        for name, b in (e.get(key) or {}).items():
            g = grouped.setdefault(name, {"calls": 0, "total_tokens": 0, "cost_usd": None})
            g["calls"] += _as_int(b.get("calls"))
            g["total_tokens"] += _as_int(b.get("total_tokens"))
            cost = _as_float(b.get("cost_usd"))
            if cost is not None:
                g["cost_usd"] = (g["cost_usd"] or 0.0) + cost
    return grouped


def ledger_totals(entries: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [c for c in (_as_float(e.get("cost_usd")) for e in entries) if c is not None]
    return {
        "runs": len(entries),
        "calls": sum(_as_int(e.get("calls")) for e in entries),
        "prompt_tokens": sum(_as_int(e.get("prompt_tokens")) for e in entries),
        "completion_tokens": sum(_as_int(e.get("completion_tokens")) for e in entries),
        "total_tokens": sum(_as_int(e.get("total_tokens")) for e in entries),
        "cost_usd": round(sum(costs), 6) if costs else None,
        "by_spec": _sum_group(entries, "spec_path"),
        "by_stage": _sum_nested(entries, "by_stage"),
        "by_model": _sum_nested(entries, "by_model"),
    }


def render_ledger_report_md(entries: list[dict[str, Any]]) -> str:
    """Markdown report of every recorded run: totals, then by spec, model, stage, recent runs."""
    if not entries:
        return "## Project token usage\n\nNo runs recorded in the ledger yet."
    t = ledger_totals(entries)
    lines = [
        "## Project token usage",
        "",
        f"**{t['total_tokens']:,} tokens** across {t['runs']} run(s) and {t['calls']} model "
        f"call(s) · estimated cost {_fmt_cost(t['cost_usd'])}",
    ]
    for title, key, first in (("By spec", "by_spec", "Spec"), ("By model", "by_model", "Model"),
                              ("By stage", "by_stage", "Stage")):
        lines.extend(["", f"### {title}", "", f"| {first} | Calls | Total tokens | Est. Cost |",
                      "| :--- | ---: | ---: | ---: |"])
        for name, g in t[key].items():
            lines.append(f"| `{name}` | {g['calls']} | {g['total_tokens']:,} | "
                         f"{_fmt_cost(g['cost_usd'])} |")
    lines.extend(["", "### Recent runs", "",
                  "| Run | Mode | Spec | Status | Total tokens | Est. Cost |",
                  "| :--- | :--- | :--- | :--- | ---: | ---: |"])
    for e in entries[-10:]:
        lines.append(f"| `{e.get('run_id')}` | {e.get('mode')} | `{e.get('spec_path')}` | "
                     f"{e.get('status')} | {_as_int(e.get('total_tokens')):,} | "
                     f"{_fmt_cost(_as_float(e.get('cost_usd')))} |")
    return "\n".join(lines)
