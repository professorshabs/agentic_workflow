import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from atlas import __version__
from atlas.changes import changed_specs
from atlas.harnesses import run_doctor
from atlas.local_models import DEFAULT_LM_STUDIO_URL, DEFAULT_OLLAMA_URL, resolve_local_model
from atlas.pipeline.build import default_app_dir
from atlas.pipeline.verify import verify_app
from atlas.router import ModelRouter
from atlas.runner import run_build, run_plan
from atlas.schemas import TokenUsageSummary
from atlas.telemetry import (
    DEFAULT_LEDGER,
    ledger_totals,
    read_ledger,
    render_ledger_report_md,
    render_token_report_md,
    summarize_usage,
)

logger = logging.getLogger("atlas")


def _find_latest_run(runs_root: Path) -> Path:
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    candidates = sorted(
        (
            d
            for d in runs_root.iterdir()
            if d.is_dir() and ((d / "tokens.json").is_file() or (d / "manifest.json").is_file())
        ),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {runs_root}")
    return candidates[0]


def _load_token_summary(run_dir: Path) -> TokenUsageSummary:
    tokens_path = run_dir / "tokens.json"
    if tokens_path.is_file():
        return TokenUsageSummary.model_validate_json(tokens_path.read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return summarize_usage(manifest.get("model_calls", []))
    raise FileNotFoundError(f"Neither tokens.json nor manifest.json found in {run_dir}")


def _add_ledger_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path(DEFAULT_LEDGER),
        help=f"append this run's token totals to a project ledger (default: {DEFAULT_LEDGER})",
    )
    p.add_argument("--no-ledger", action="store_true", help="do not write to the ledger")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="atlas")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan_parser = sub.add_parser("plan", help="specification -> research -> plan")
    plan_parser.add_argument("spec", type=Path)
    plan_parser.add_argument("--out", type=Path, default=Path("runs"))
    plan_parser.add_argument("--config", type=Path, default=None)
    plan_parser.add_argument("--verbose", action="store_true")
    _add_ledger_args(plan_parser)

    build_parser = sub.add_parser(
        "build",
        aliases=["run"],
        help="specification -> (plan) -> build app -> test/verify -> token report",
    )
    build_parser.add_argument("spec", type=Path)
    build_parser.add_argument(
        "--app-dir",
        type=Path,
        default=None,
        help="output directory for the built product (default: app/<spec-name>)",
    )
    build_parser.add_argument("--out", type=Path, default=Path("runs"))
    build_parser.add_argument("--config", type=Path, default=None)
    build_parser.add_argument("--skip-research", action="store_true")
    build_parser.add_argument("--clean", action="store_true")
    build_parser.add_argument("--max-fix-rounds", type=int, default=None)
    build_parser.add_argument("--report-out", type=Path, default=None)
    build_parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="override explicit local model for workspace builder",
    )
    build_parser.add_argument("--verbose", action="store_true")
    _add_ledger_args(build_parser)

    verify_parser = sub.add_parser(
        "verify", help="detect stack and run test/lint/build verification checks"
    )
    verify_parser.add_argument("--app-dir", type=Path, default=Path("app"))
    verify_parser.add_argument("--timeout", type=float, default=180.0)
    verify_parser.add_argument("--json", action="store_true")

    tokens_parser = sub.add_parser("tokens", help="show token usage and cost for a run")
    tokens_parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    tokens_parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    tokens_parser.add_argument("--json", action="store_true")
    tokens_parser.add_argument(
        "--all", action="store_true", help="every recorded run, from the project ledger"
    )
    tokens_parser.add_argument("--ledger", type=Path, default=Path(DEFAULT_LEDGER))

    changes_parser = sub.add_parser(
        "changed-specs", help="list specs added or changed between two commits (used by CI)"
    )
    changes_parser.add_argument("--before", required=True)
    changes_parser.add_argument("--after", default="HEAD")
    changes_parser.add_argument("--repo", type=Path, default=Path("."))

    app_dir_parser = sub.add_parser(
        "app-dir", help="print default output directory for a spec (app/<spec-name>)"
    )
    app_dir_parser.add_argument("spec", type=Path)
    app_dir_parser.add_argument("--app-root", type=Path, default=Path("app"))

    val_parser = sub.add_parser("validate-config", help="validate config/models.yaml")
    val_parser.add_argument("--config", type=Path, default=None)

    doctor_parser = sub.add_parser(
        "doctor", help="check harness CLIs, live auth status, local daemons, and skills"
    )
    doctor_parser.add_argument("--config", type=Path, default=None)
    doctor_parser.add_argument("--live", action="store_true", help="run live local smoke check")
    doctor_parser.add_argument("--json", action="store_true")

    detect_parser = sub.add_parser(
        "detect-local", help="probe local Ollama / LM Studio models and verify readiness"
    )
    detect_parser.add_argument("--config", type=Path, default=None)
    detect_parser.add_argument("--local-model", type=str, default=None)
    detect_parser.add_argument("--no-smoke-check", action="store_true")
    detect_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not getattr(args, "verbose", False):
        # LiteLLM logs every request at INFO; keep the terminal to Atlas's own messages.
        for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx"):
            logging.getLogger(name).setLevel(logging.WARNING)

    try:
        if args.cmd == "plan":
            out = asyncio.run(
                run_plan(
                    args.spec,
                    args.out,
                    args.config,
                    ledger_path=None if args.no_ledger else args.ledger,
                )
            )
            summary = _load_token_summary(out)
            print(f"Plan written to {out / 'plan.md'}")
            print(
                f"Tokens: {summary.total_tokens} total "
                f"(prompt={summary.prompt_tokens}, completion={summary.completion_tokens}, "
                f"calls={summary.total_calls})"
            )
        elif args.cmd in ("build", "run"):
            app_dir = (
                args.app_dir if args.app_dir is not None else default_app_dir(args.spec)
            )
            out = asyncio.run(
                run_build(
                    args.spec,
                    args.out,
                    app_dir,
                    config_path=args.config,
                    skip_research=args.skip_research,
                    clean_app_dir=args.clean,
                    max_fix_rounds=args.max_fix_rounds,
                    report_out=args.report_out,
                    ledger_path=None if args.no_ledger else args.ledger,
                    local_model=args.local_model,
                )
            )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            summary = _load_token_summary(out)
            print(f"Build status: {manifest['status']}")
            print(f"Product written to {app_dir}")
            print(f"Report written to {out / 'report.md'}")
            print(
                f"Tokens: {summary.total_tokens} total "
                f"(prompt={summary.prompt_tokens}, completion={summary.completion_tokens}, "
                f"calls={summary.total_calls})"
            )
            if manifest.get("status") != "verified":
                sys.exit(1)
        elif args.cmd == "verify":
            stack, checks = asyncio.run(
                verify_app(args.app_dir, timeout_seconds=args.timeout)
            )
            passed = bool(checks) and all(c.passed for c in checks)
            if args.json:
                print(
                    json.dumps(
                        {
                            "stack": stack,
                            "passed": passed,
                            "checks": [c.model_dump(mode="json") for c in checks],
                        },
                        indent=2,
                    )
                )
            else:
                print(f"Stack: {stack}")
                for chk in checks:
                    badge = "PASS" if chk.passed else "FAIL"
                    print(f"[{badge}] {chk.name}: {chk.command} (exit={chk.exit_code})")
                    if not chk.passed and (chk.stderr or chk.stdout):
                        print(chk.stderr or chk.stdout)
            if not passed:
                sys.exit(1)
        elif args.cmd == "tokens" and args.all:
            entries = read_ledger(args.ledger)
            if args.json:
                print(json.dumps(ledger_totals(entries), indent=2))
            else:
                print(render_ledger_report_md(entries))
        elif args.cmd == "changed-specs":
            for path in changed_specs(args.repo, args.before, args.after):
                print(path)
        elif args.cmd == "app-dir":
            print(default_app_dir(args.spec, args.app_root).as_posix())
        elif args.cmd == "tokens":
            target_dir = (
                args.run_dir if args.run_dir is not None else _find_latest_run(args.runs_root)
            )
            summary = _load_token_summary(target_dir)
            if args.json:
                print(summary.model_dump_json(indent=2))
            else:
                print(render_token_report_md(summary))
        elif args.cmd == "validate-config":
            router = ModelRouter(args.config)
            print(f"Config valid: {router.path}")
        elif args.cmd == "doctor":
            router = ModelRouter(args.config)
            report = asyncio.run(run_doctor(router, live=args.live))
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"Config: {router.path}")
                print(f"Overall healthy: {report['healthy']}")
                for chk in report["checks"]:
                    ok = bool(chk.get("available")) and bool(chk.get("authenticated", True))
                    if chk["provider"] == "local_model_daemon":
                        ok = bool(chk.get("ready"))
                    badge = "OK" if ok else "FAIL"
                    ver = f" ({chk['version']})" if chk.get("version") else ""
                    auth_label = f" [auth={chk['auth_mode']}]" if chk.get("auth_mode") else ""
                    print(f"[{badge}] {chk['provider']}{ver}{auth_label}: {chk.get('details', '')}")
                    if not ok and chk.get("login_hint"):
                        print(f"       hint: {chk['login_hint']}")
                skills_info = report.get("skills") or {}
                if skills_info.get("configured"):
                    resolved = ", ".join(skills_info.get("resolved") or []) or "(none)"
                    print(f"Skills allowlist resolved: {resolved}")
            if not report["healthy"]:
                sys.exit(1)
        elif args.cmd == "detect-local":
            router = ModelRouter(args.config)
            builder_cfg = router.backend_for("builder")
            explicit = args.local_model or builder_cfg.get("local_model")
            providers = builder_cfg.get("local_providers") or {
                "ollama": DEFAULT_OLLAMA_URL,
                "lm_studio": DEFAULT_LM_STUDIO_URL,
            }
            res = asyncio.run(
                resolve_local_model(
                    explicit_model=str(explicit) if explicit else None,
                    providers=providers,
                    preferred=(
                        list(builder_cfg["preferred"]) if builder_cfg.get("preferred") else None
                    ),
                    require_tools=bool(builder_cfg.get("require_tools", True)),
                    min_context_tokens=builder_cfg.get("min_context_tokens"),
                    max_model_size_gb=builder_cfg.get("max_model_size_gb"),
                    memory_budget_fraction=builder_cfg.get("memory_budget_fraction"),
                    require_tool_smoke_check=not args.no_smoke_check,
                )
            )
            if args.json:
                print(res.model_dump_json(indent=2))
            else:
                print(res.reason)
                for cand in res.candidates:
                    size_label = (
                        f"{cand.parameter_size_b:g}B" if cand.parameter_size_b else "?B"
                    )
                    flags = []
                    if cand.loaded:
                        flags.append("loaded")
                    if cand.tool_smoke_passed is True:
                        flags.append("smoke=ok")
                    elif cand.tool_smoke_passed is False:
                        flags.append("smoke=fail")
                    if res.selected and res.selected.name == cand.name:
                        flags.append("SELECTED")
                    suffix = f" [{', '.join(flags)}]" if flags else ""
                    print(f"  - {cand.provider}: {cand.name} ({size_label}){suffix}")
                for name, why in res.rejected.items():
                    print(f"  rejected {name}: {why}")
            if res.selected is None:
                sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
