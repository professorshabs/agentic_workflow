import httpx

from atlas.local_models import (
    LocalModelCandidate,
    parse_parameter_billions,
    probe_lm_studio,
    probe_ollama,
    resolve_local_model,
)


def test_parse_parameter_billions():
    assert parse_parameter_billions("26B") == 26.0
    assert parse_parameter_billions("500M") == 0.5
    assert parse_parameter_billions(None, "gemma4:26b") == 26.0
    assert parse_parameter_billions(None, "qwen2.5-coder:32b") == 32.0
    assert parse_parameter_billions(None, "custom-model") is None


async def test_probe_ollama_show_and_lm_studio_filter_embeddings(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "nomic-embed-text:latest",
                            "size": 274_000_000,
                            "details": {"parameter_size": "137M", "family": "nomic-bert"},
                        },
                        {
                            "name": "gpt-oss:20b",
                            "size": 14_000_000_000,
                            "details": {"parameter_size": "20B", "family": "gptoss"},
                        },
                        {
                            "name": "plain-chat:8b",
                            "size": 5_000_000_000,
                            "details": {"parameter_size": "8B", "family": "llama"},
                        },
                    ]
                },
            )
        if url.endswith("/api/ps"):
            return httpx.Response(
                200,
                json={"models": [{"name": "gpt-oss:20b", "context_length": 65536}]},
            )
        if url.endswith("/api/show"):
            body = request.read().decode("utf-8")
            if "gpt-oss:20b" in body:
                return httpx.Response(
                    200,
                    json={
                        "capabilities": ["completion", "tools"],
                        "model_info": {"gptoss.context_length": 131072},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion"],
                    "model_info": {"llama.context_length": 8192},
                },
            )
        if url.endswith("/v1/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "bge-large-en"},
                        {
                            "id": "qwen2.5-coder-14b",
                            "max_context_length": 65536,
                            "capabilities": ["tools"],
                        },
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {url}")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "atlas.local_models.httpx.AsyncClient",
        lambda **kw: real_async_client(transport=transport, **kw),
    )

    ollama_cands, ollama_err = await probe_ollama("http://localhost:11434")
    assert ollama_err is None
    assert [c.name for c in ollama_cands] == ["gpt-oss:20b", "plain-chat:8b"]
    assert ollama_cands[0].loaded is True
    assert ollama_cands[0].supports_tools is True
    assert ollama_cands[0].context_length == 65536
    assert ollama_cands[1].supports_tools is False
    assert ollama_cands[1].context_length == 8192

    lms_cands, lms_err = await probe_lm_studio("http://localhost:1234/v1")
    assert lms_err is None
    assert [c.name for c in lms_cands] == ["qwen2.5-coder-14b"]
    assert lms_cands[0].supports_tools is True


async def test_resolve_local_model_filters_tools_context_memory_preferred_and_smoke(monkeypatch):
    gpt_oss = LocalModelCandidate(
        provider="ollama",
        name="gpt-oss:20b",
        base_url="http://localhost:11434/v1",
        parameter_size_b=20.0,
        size_bytes=14 * 1024**3,
        context_length=65536,
        supports_tools=True,
    )
    gemma_big = LocalModelCandidate(
        provider="ollama",
        name="gemma4:26b",
        base_url="http://localhost:11434/v1",
        parameter_size_b=26.0,
        size_bytes=16 * 1024**3,
        context_length=65536,
        supports_tools=True,
    )
    no_tools = LocalModelCandidate(
        provider="ollama",
        name="chat-only:32b",
        base_url="http://localhost:11434/v1",
        parameter_size_b=32.0,
        size_bytes=10 * 1024**3,
        context_length=65536,
        supports_tools=False,
    )
    small_ctx = LocalModelCandidate(
        provider="ollama",
        name="short-ctx:14b",
        base_url="http://localhost:11434/v1",
        parameter_size_b=14.0,
        size_bytes=9 * 1024**3,
        context_length=4096,
        supports_tools=True,
    )

    async def fake_probe_ollama(*args, **kwargs):
        return [
            gpt_oss.model_copy(),
            gemma_big.model_copy(),
            no_tools.model_copy(),
            small_ctx.model_copy(),
        ], None

    async def fake_probe_lms(*args, **kwargs):
        return [], "offline"

    monkeypatch.setattr("atlas.local_models.probe_ollama", fake_probe_ollama)
    monkeypatch.setattr("atlas.local_models.probe_lm_studio", fake_probe_lms)

    async def smoke(cand: LocalModelCandidate) -> tuple[bool, str]:
        return True, "ok"

    # Discovery filters out no-tools, <64K context, >70% of 24 GiB RAM (16 GiB * 1.15 > 16.8 GiB),
    # and respects `preferred` + `dict` local_providers.
    ram_24g = 24 * 1024**3
    disc_res = await resolve_local_model(
        providers={"ollama": "http://localhost:11434/v1", "lm_studio": "http://localhost:1234/v1"},
        preferred=["gpt-oss:20b", "gemma4:26b"],
        require_tools=True,
        min_context_tokens=64000,
        memory_budget_fraction=0.70,
        total_memory_bytes=ram_24g,
        smoke_check_fn=smoke,
    )
    assert disc_res.selected is not None
    assert disc_res.selected.name == "gpt-oss:20b"
    assert "chat-only:32b" in disc_res.rejected
    assert "short-ctx:14b" in disc_res.rejected
    assert "gemma4:26b" in disc_res.rejected

    # Explicit model fails clearly if missing.
    missing_res = await resolve_local_model(explicit_model="missing:7b", smoke_check_fn=smoke)
    assert missing_res.selected is None
    assert "ollama pull missing:7b" in missing_res.reason

    # Offline servers return clear setup instructions (no cloud builder fallback).
    async def empty_probe(*args, **kwargs):
        return [], "Ollama offline"

    monkeypatch.setattr("atlas.local_models.probe_ollama", empty_probe)
    offline_res = await resolve_local_model()
    assert offline_res.selected is None
    assert "OLLAMA_CONTEXT_LENGTH=64000 ollama serve" in offline_res.reason
