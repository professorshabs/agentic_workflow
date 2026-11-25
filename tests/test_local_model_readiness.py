"""Local builder model readiness: the runtime context check and the tool smoke check parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from atlas.local_models import (
    LocalModelCandidate,
    ollama_runtime_context,
    resolve_local_model,
    run_tool_smoke_check,
)


def _mock_http(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "atlas.local_models.httpx.AsyncClient", lambda **kw: real(transport=transport, **kw)
    )


def _tags_handler(ps_context: int | None, state: dict | None = None):
    """Ollama stand-in; gemma shows up in `/api/ps` only once `state["loaded"]` is set."""
    state = state if state is not None else {"loaded": True}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [
                {"name": "gemma4:12b", "size": 8_000_000_000,
                 "details": {"parameter_size": "12B", "family": "gemma4"}},
                {"name": "qwen3:8b", "size": 5_000_000_000,
                 "details": {"parameter_size": "8B", "family": "qwen3"}},
            ]})
        if url.endswith("/api/ps"):
            if ps_context is None or not state["loaded"]:
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"models": [
                {"name": "gemma4:12b", "model": "gemma4:12b", "context_length": ps_context}
            ]})
        if url.endswith("/api/show"):
            # The trained maximum: what the pre-load check sees.
            return httpx.Response(200, json={"capabilities": ["completion", "tools"],
                                             "model_info": {"gemma4.context_length": 131072}})
        raise AssertionError(url)

    return handler


async def _always_passes(candidate):
    return True, "ok"


async def test_model_served_with_ollamas_default_context_is_rejected(monkeypatch):
    state = {"loaded": False}
    _mock_http(monkeypatch, _tags_handler(ps_context=4096, state=state))

    async def smoke_loads_model(candidate):
        state["loaded"] = True  # the smoke call makes Ollama load the model
        return True, "ok"

    res = await resolve_local_model(
        providers={"ollama": "http://localhost:11434/v1"},
        preferred=["gemma4:12b"],
        min_context_tokens=32768,
        total_memory_bytes=24 * 1024**3,
        memory_budget_fraction=0.78,
        smoke_check_fn=smoke_loads_model,
    )

    # gemma passed the smoke test but runs with 4096 tokens; qwen (not loaded -> unknown) wins.
    assert "4096-token context" in res.rejected["gemma4:12b"]
    assert "OLLAMA_CONTEXT_LENGTH=32768" in res.rejected["gemma4:12b"]
    assert res.selected is not None and res.selected.name == "qwen3:8b"


async def test_model_with_enough_runtime_context_is_selected(monkeypatch):
    state = {"loaded": False}
    _mock_http(monkeypatch, _tags_handler(ps_context=65536, state=state))

    async def smoke_loads_model(candidate):
        state["loaded"] = True
        return True, "ok"

    res = await resolve_local_model(
        providers={"ollama": "http://localhost:11434/v1"},
        preferred=["gemma4:12b"],
        min_context_tokens=32768,
        smoke_check_fn=smoke_loads_model,
    )

    assert res.selected is not None and res.selected.name == "gemma4:12b"
    assert res.selected.context_length == 65536


async def test_every_candidate_too_small_reports_readiness_failure(monkeypatch):
    async def small(candidate):
        return 4096

    _mock_http(monkeypatch, _tags_handler(ps_context=None))
    res = await resolve_local_model(
        providers={"ollama": "http://localhost:11434/v1"},
        min_context_tokens=32768,
        smoke_check_fn=_always_passes,
        runtime_context_fn=small,
    )
    assert res.selected is None
    assert res.reason.startswith("Local model(s) failed readiness checks")


async def test_runtime_context_lookup_tolerates_errors(monkeypatch):
    cand = LocalModelCandidate(provider="ollama", name="gemma4:12b",
                               base_url="http://localhost:11434/v1")

    _mock_http(monkeypatch, lambda request: httpx.Response(500))
    assert await ollama_runtime_context(cand) is None

    def boom(request):
        raise httpx.ConnectError("refused")

    _mock_http(monkeypatch, boom)
    assert await ollama_runtime_context(cand) is None


def _chat_response(message: dict | None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/v1/chat/completions")
        body = json.loads(request.read())
        assert body["tool_choice"] == "required" and body["tools"][0]["function"]["name"] == "ping"
        if status >= 400:
            return httpx.Response(status, text="model not found")
        choices = [] if message is None else [{"message": message}]
        return httpx.Response(200, json={"choices": choices})

    return handler


@pytest.mark.parametrize(
    ("message", "status", "ok", "detail"),
    [
        ({"tool_calls": [{"function": {"name": "ping", "arguments": "{}"}}]}, 200, True, "ok"),
        ({"tool_calls": [{"function": {"name": "ping", "arguments": {}}}]}, 200, True, "ok"),
        ({"tool_calls": [{"function": {"name": "ping", "arguments": "{bad"}}]}, 200, False,
         "valid ping"),
        ({"tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]}, 200, False,
         "valid ping"),
        ({"content": "pong"}, 200, False, "did not call"),
        (None, 200, False, "No choices"),
        ({}, 404, False, "HTTP 404"),
    ],
)
async def test_tool_smoke_check_outcomes(monkeypatch, message, status, ok, detail):
    _mock_http(monkeypatch, _chat_response(message, status))
    cand = LocalModelCandidate(provider="lm_studio", name="m", base_url="http://localhost:1234/v1")
    passed, msg = await run_tool_smoke_check(cand)
    assert passed is ok and detail in msg
