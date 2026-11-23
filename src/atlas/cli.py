import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from atlas import __version__
from atlas.pipeline.verify import verify_app
from atlas.router import ModelRouter
from atlas.runner import run_build, run_plan
from atlas.schemas import TokenUsageSummary
from atlas.telemetry import render_token_report_md, summarize_usage

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

    build_parser = sub.add_parser(
        "build", help="specification -> (plan) -> build app -> test/verify -> token report"
    )
    build_parser.add_argument("spec", type=Path)
    build_parser.add_argument("--app-dir", type=Path, default=Path("app"))
    build_parser.add_argument("--out", type=Path, default=Path("runs"))
    build_parser.add_argument("--config", type=Path, default=None)
    build_parser.add_argument("--skip-research", action="store_true")
    build_parser.add_argument("--clean", action="store_true")
    build_parser.add_argument("--max-fix-rounds", type=int, default=None)
    build_parser.add_argument("--report-out", type=Path, default=None)
    build_parser.add_argument("--verbose", action="store_true")

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

    val_parser = sub.add_parser("validate-config", help="validate config/models.yaml")
    val_parser.add_argument("--config", type=Path, default=None)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.cmd == "plan":
            out = asyncio.run(run_plan(args.spec, args.out, args.config))
            summary = _load_token_summary(out)
            print(f"Plan written to {out / 'plan.md'}")
            print(
                f"Tokens: {summary.total_tokens} total "
                f"(prompt={summary.prompt_tokens}, completion={summary.completion_tokens}, "
                f"calls={summary.total_calls})"
            )
        elif args.cmd == "build":
            out = asyncio.run(
                run_build(
                    args.spec,
                    args.out,
                    args.app_dir,
                    config_path=args.config,
                    skip_research=args.skip_research,
                    clean_app_dir=args.clean,
                    max_fix_rounds=args.max_fix_rounds,
                    report_out=args.report_out,
                )
            )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            summary = _load_token_summary(out)
            print(f"Build status: {manifest['status']}")
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
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
