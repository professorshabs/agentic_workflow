"""`ModelRouter` config validation: the shipped configs load, and each bad value is rejected
with a message naming the offending key (so `atlas validate-config` / `doctor` fail early)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from atlas.router import ModelRouter

REPO = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((REPO / "config" / "harness.yaml").read_text(encoding="utf-8"))

R = "backends.supervisor.reviewers"
CASES: list[tuple[str, Any, str]] = [
    ("roles", {}, "must define at least one role"),
    ("roles.parser", [], "Role 'parser' must be a non-empty list"),
    ("role_fallbacks", ["x"], "'role_fallbacks' must be a mapping"),
    ("role_fallbacks.parser", "x", r"role_fallbacks\['parser'\] must be a list"),
    ("fallbacks", "x", "'fallbacks' must be a list"),
    ("limits.max_parallel_calls", 0, "max_parallel_calls must be >= 1"),
    ("limits.max_tokens", 0, "max_tokens must be >= 1"),
    ("limits.timeout_seconds", 0, r"limits.timeout_seconds must be > 0"),
    ("limits.attempts_per_model", 0, "attempts_per_model must be >= 1"),
    ("limits.repair_attempts", -1, "repair_attempts must be >= 0"),
    ("limits.max_fix_rounds", -1, "max_fix_rounds must be >= 0"),
    ("limits.verify_timeout_seconds", 0, "verify_timeout_seconds must be > 0"),
    ("limits.max_total_tokens", 0, "max_total_tokens must be >= 1"),
    ("limits.max_cost_usd", 0, "max_cost_usd must be > 0"),
    ("limits.max_tokens_by_role", ["judge"], "max_tokens_by_role must be a mapping"),
    ("limits.max_tokens_by_role", {"judge": 0}, r"max_tokens_by_role\['judge'\] must be >= 1"),
    ("limits.timeout_seconds_by_role", ["judge"], "timeout_seconds_by_role must be a mapping"),
    ("limits.timeout_seconds_by_role", {"judge": 0}, r"\['judge'\] must be > 0"),
    ("verification", ["x"], "'verification' must be a mapping"),
    ("backends", ["x"], "'backends' must be a mapping"),
    ("backends.parser", "x", r"backends\['parser'\] must be a mapping"),
    ("backends.parser.type", "magic", "type 'magic' must be one of"),
    ("backends.parser.provider", "codex", "provider must be one of"),
    ("backends.judge.read_only", False, "strictly read-only"),
    ("backends.judge.tools", ["Bash"], "tools must be a subset"),
    ("backends.judge.auth", "password", "auth must be one of"),
    ("backends.parser.fallback", "openrouter", "fallback must be 'router'"),
    ("backends.researcher.max_parallel", 0, "max_parallel must be >= 1"),
    ("backends.judge.max_budget_usd", 0, "max_budget_usd must be > 0"),
    ("backends.parser.timeout_seconds", 0, "timeout_seconds must be > 0"),
    ("backends.parser.router_model", " ", "router_model must be a non-empty string"),
    ("backends.supervisor.mode", "majority", "mode must be one of"),
    (R, [], "reviewers must be a non-empty list"),
    (R, ["opus"], r"reviewers\[0\] must be a mapping"),
    (R, [{"type": "magic"}], "type 'magic' must be one of"),
    (R, [{"type": "openrouter"}], "model must be a non-empty string"),
    (R, [{"type": "openrouter", "model": "m", "timeout_seconds": 0}], "timeout_seconds"),
    (R, [{"type": "harness", "provider": "claude_code", "tools": ["Write"]}], "subset"),
    (R, [{"type": "openrouter", "model": "m", "name": " "}], "name must not be empty"),
    (R, [{"type": "openrouter", "model": "m", "name": "A"},
         {"type": "openrouter", "model": "n", "name": "a"}], "duplicate name"),
    ("backends.builder.local_harness", "ollama_cli", "local_harness must be one of"),
    ("backends.builder.allow_cloud_fallback", True, "builder is local-only"),
    ("backends.builder.local_providers", {"vllm": "http://x"}, "keys must be in"),
    ("backends.builder.local_providers", [], "must be a non-empty dict or list"),
    ("backends.fixer.rotate", [], "rotate must be a non-empty list"),
    ("limits_extra", ["x"], "'limits_extra' must be a mapping"),
    ("limits_extra.max_total_tokens_per_run", 0, "max_total_tokens_per_run must be >= 1"),
    ("skills", ["x"], "'skills' must be a mapping"),
    ("skills.sources", "x", "skills.sources must be a list"),
    ("skills.allowlist", "x", "skills.allowlist must be a list"),
    ("skills.by_role", ["x"], "skills.by_role must be a mapping"),
    ("skills.by_role.builder", "x", r"by_role\['builder'\] must be a list"),
    ("skills.by_role.builder", ["not-allowlisted"], "non-allowlisted skill"),
    ("backends.e2e.effort", "turbo", "effort must be one of"),
]


def _with(path: str, value: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(BASE)
    *parents, leaf = path.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value
    return cfg


def _load(tmp_path: Path, cfg: Any) -> ModelRouter:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return ModelRouter(path)


@pytest.mark.parametrize("name", ["models.yaml", "harness.yaml"])
def test_shipped_configs_are_valid(name):
    router = ModelRouter(REPO / "config" / name)
    assert router.cfg["roles"]


@pytest.mark.parametrize(("path", "value", "message"), CASES, ids=[c[0] for c in CASES])
def test_invalid_config_value_is_rejected(tmp_path, path, value, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, _with(path, value))


@pytest.mark.parametrize("missing", ["roles", "limits"])
def test_missing_required_section(tmp_path, missing):
    cfg = copy.deepcopy(BASE)
    del cfg[missing]
    with pytest.raises(ValueError, match=f"Missing or invalid '{missing}'"):
        _load(tmp_path, cfg)


def test_non_mapping_yaml_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        _load(tmp_path, ["not", "a", "mapping"])
