# Locally Testing Atlas and Generated Products

Use this guide to test the Atlas framework and products built under `app/<spec-name>/` on your Mac. The repository-level `tests/` suite tests Atlas; each generated product has its own tests, which `atlas verify` runs from that product. GitHub product builds and deployment are deferred.

---

## Step 1 — Activate the Environment & Run Offline Checks (0 Tokens)

From the repository root:

```bash
source .venv/bin/activate
```

*(If `.venv` does not exist yet on a fresh machine, create it first:)*

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional: include `google-antigravity` SDK when running `config/harness.yaml`:
pip install -e ".[dev,harnesses]"
```

Run the full offline quality and test suite:

```bash
ruff check .
mypy src/ --ignore-missing-imports
pytest -v
atlas validate-config
atlas validate-config --config config/harness.yaml
```

**What passing looks like:**
- `ruff check .` prints `All checks passed!`
- `mypy src/ --ignore-missing-imports` prints `Success: no issues found ...`
- `pytest -v` passes all automated unit, verification, and workflow tests (no API keys required)
- `atlas validate-config` prints `Config valid: .../config/models.yaml` (and `.../config/harness.yaml`)

Optional deeper checks (still 0 tokens):

```bash
# Line + branch coverage of the framework (about 92 % today)
pytest --cov=atlas --cov-branch --cov-report=term-missing:skip-covered

# Check the Claude adapter against your installed `claude` binary. It talks to a local
# mock of the Anthropic API with a throwaway HOME, so your login is not used or touched.
ATLAS_CLAUDE_CONTRACT=1 pytest tests/test_harness_cli_contracts.py -k real_claude
```

With `pip install -e ".[dev,harnesses]"`, `pytest` also checks that the installed `google-antigravity` SDK still has every API the local builder calls (it is alpha, so re-run this after upgrading it).

*(If you are using the local-first hybrid profile in `config/harness.yaml`, load Gemma 4 12B in LM Studio with `lms unload --all && lms load google/gemma-4-12b-qat --context-length 49152` — or start Ollama (`OLLAMA_CONTEXT_LENGTH=49152 ollama serve`) with `gemma4:12b-mlx` / `gemma4:12b` — and run `atlas doctor --config config/harness.yaml` and `atlas detect-local --config config/harness.yaml` before building to verify `agy`, `claude`, `OPENROUTER_API_KEY`, local daemons, and allowlisted skills.)*

---

## Step 2 — Confirm Local API Keys Are Set & Git-Ignored

Make sure `.env` exists and is ignored by Git so your keys are never committed:

```bash
git check-ignore -v .env
```

Expected output shows `.gitignore` matching `.env`. Confirm your keys are populated (without printing their values):

```bash
grep -E '^(OPENROUTER_API_KEY|GEMINI_API_KEY)=.+' .env | cut -d= -f1
```

You should see `OPENROUTER_API_KEY` and/or `GEMINI_API_KEY`.

---

## Step 3 — Run a Live Local Build, Verify, and Inspect Token Usage

### 3a. Full local-first harness run

```bash
# Optional: export ATLAS_CONFIG=config/harness.yaml once instead of repeating --config
atlas run specs/example-app.md --config config/harness.yaml --clarify --e2e-tests
# Resume straight from build/verify using a prior run's plan + E2E tests:
atlas run specs/example-app.md --config config/harness.yaml --from-run runs/<run-id> --clean
# Non-interactive with prefilled answers JSON:
atlas run specs/example-app.md --config config/harness.yaml --answers answers.json
# Suppress step-by-step stderr progress (or use --verbose for full debug logs):
atlas run specs/example-app.md --config config/harness.yaml --quiet
```

`atlas run`, `atlas build`, and `atlas plan` stream live `[HH:MM:SS]` milestones to `stderr` (`[parse]`, `[research] k/N`, `[clarify]`, `[judge] k/N`, `[e2e]`, `[build]`, per-check `[PASS]`/`[FAIL]` `[verify]` lines, `[fix]` rounds, and `[supervisor]` verdicts) without polluting `stdout`. When using `config/harness.yaml`, Atlas pre-seeds `PLAN.md`, authoritative `tests/` E2E test files, and progressive `.agents/skills/` symlinks inside `app/<spec-name>/` before the local Builder starts so the local model stays within its context window, can test against the E2E suite as it builds, and has any tampered E2E files restored automatically before `verify_app` runs.

### 3b. Re-run product verification alone (0 LLM tokens)

```bash
atlas verify --app-dir app/example-app
# Also enforce the standalone product contract (README, dependency manifest, no Atlas imports):
atlas verify --app-dir app/example-app --product-contract
# Also enforce requirement coverage and E2E test SHA-256 integrity from a recorded run:
atlas verify --app-dir app/example-app --product-contract --from-run runs/<run-id>
```

Bare `atlas verify` runs the product's own tests alongside compile, lint, and build checks; `--product-contract` adds `product-contract`, and `--from-run` adds `spec-coverage` and `e2e-product-tests` integrity verification. With `--from-run` and no `--app-dir`, Atlas verifies the product folder that run built (its `app_dir`), so `atlas verify --from-run runs/<run-id>` is enough. To run the product's test command directly, use its own instructions from its directory, such as `cd app/example-app && pytest`.

### 3c. Inspect token usage and estimated cost

```bash
atlas tokens
atlas tokens --json
atlas tokens --all
```

- `atlas tokens` / `atlas tokens --json` shows the latest run's breakdown (including empty or truncated replies that were still billed).
- `atlas tokens --all` reads `usage/ledger.jsonl` and shows **cumulative totals across all runs** grouped by spec, model, and stage.

You can also inspect the full Markdown report and raw JSON telemetry for the newest run under `runs/<run-id>/`:
- `runs/<run-id>/report.md` — verification table, auto-fix diagnoses, optional **Clarifications**, **End-to-end product tests**, and **Supervisor review** sections, generated files, and token/cost breakdown
- `runs/<run-id>/clarifications.json` — collected spec/research clarifying questions and user answers applied before planning/building
- `runs/<run-id>/e2e_tests.json` — authoritative black-box end-product contract and test files pre-seeded into `app/<spec-name>/tests/` (`e2e-product-tests` check; sets `verifiable: false` with `unverifiable_reason` if offline E2E testing isn't possible)
- `runs/<run-id>/tokens.json` — per-stage, per-model, and per-call token + USD cost records
- `runs/<run-id>/verification.json` — exact check commands, exit codes, `stdout`, and `stderr`
- `runs/<run-id>/supervisor_review.json` — unified post-build review verdict, merged issues, improvement plan, and per-reviewer breakdown (when `backends.supervisor.enabled: true`)
- `runs/<run-id>/events.jsonl` — persisted live workflow UI timeline events
- `runs/<run-id>/manifest.json` — overall run status (`verified`, `verification_failed`, or `failed`), plus `local_model` / `build_offloaded_to` / `build_offload_harness`
- `usage/ledger.jsonl` — append-only cross-run token ledger

### 3d. Visually follow & replay runs with the Workflow UI (`0` extra LLM tokens)

In an interactive terminal outside CI, `atlas run` and `atlas build` automatically start the local read-only workflow UI (`127.0.0.1`) and open it in your browser, showing the horizontal left-to-right transit strip map (`Spec → Parse → Research → Clarify → Judge → E2E (sonnet high) → Build (12B → sonnet/claude_code → hydrogen/agy) → Verify ↔ Fix → Review → Report`), live stall-guard bar, per-stage artifacts, usage-by-stage breakdown, and filterable log drawer.

```bash
# Replay an existing run (0 LLM tokens, useful to test the UI immediately):
atlas ui
atlas ui runs/<run-id>

# Customize live UI behavior during a run:
atlas run specs/example-app.md --config config/harness.yaml --no-open      # print URL only
atlas run specs/example-app.md --config config/harness.yaml --ui-keep-open # keep server open after exit until Ctrl+C
atlas run specs/example-app.md --config config/harness.yaml --no-ui        # terminal output only
atlas plan specs/example-app.md --config config/harness.yaml --ui          # opt-in live UI for planning
```

Clarifying questions remain in your terminal (`--clarify`), even when the browser opens. Older runs created before `events.jsonl` existed are synthesized from their saved JSON artifacts when opened with `atlas ui`.

### Quick Troubleshooting

| Symptom | Likely Cause & Fix |
| :--- | :--- |
| `All configured models failed` / 401 / 403 / 404 | Check `OPENROUTER_API_KEY` / `GEMINI_API_KEY` in `.env` and confirm the model IDs in `config/models.yaml` are enabled on your OpenRouter/Google account. |
| `OutputTruncatedError` / cut-off reply | Raise the per-role limit in `config/models.yaml` or `config/harness.yaml` (`limits.max_tokens_by_role.e2e` / `researcher` / `builder` / `fixer`, or `limits.timeout_seconds_by_role`; reasoning models count thinking tokens inside `max_tokens`). |
| `agy` `DEADLINE_EXCEEDED` / `print_timeout` | Increase `backends.<role>.timeout_seconds` in `config/harness.yaml` (e.g. `researcher: 420`, `parser: 180`). Structured `agy` calls also enforce `--disable-slash-commands` and a single-turn constraint to prevent runaway tool loops. For `e2e`/`judge` (`claude_code`), tune `timeout_seconds` or `max_budget_usd`. |
| LM Studio model rejected for small context | In LM Studio's **Load** panel, set **Context Length** to `>= 49152` (e.g. `lms unload --all && lms load google/gemma-4-12b-qat --context-length 49152`) before running `atlas detect-local --config config/harness.yaml`. |
| Missing requirement-ID test failure | With `verification.require_requirement_tests: true` in `config/models.yaml`, every functional requirement needs a test mentioning its ID (e.g. `R001`, `R002`). The fixer is asked to add it, or you can set this to `false`. |
| `Token budget exceeded` / `Cost budget exceeded` | Increase or set `null` on `max_total_tokens` / `max_cost_usd` in `config/models.yaml`. |
| `Build status: verification_failed` (exit code 1) | Inspect `runs/<run-id>/report.md` or `runs/<run-id>/verification.json` to see which check failed; increase `max_fix_rounds` or refine acceptance criteria in your spec. |

---

## Later: Framework CI and product deployment

These are separate follow-up tasks, not part of current testing. When framework CI is introduced, it should run Atlas's quality checks (`ruff`, `mypy`, `pytest`, and dependency audit) only. Generated products should continue to be built and tested locally; when a product is ready to deploy, give it its own repository and CI/deployment flow for its own tests. See [the harness plan](HARNESS_PLAN.md#9-framework-vs-products-where-things-are-tested).
