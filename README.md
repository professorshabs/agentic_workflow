# Atlas — Spec-to-App GitHub Template

Atlas is a small repository template and local-first harness orchestrator for turning a Markdown specification into a runnable, verified app. Its pipeline parses specs, researches options, pauses for clarifying questions, selects an engineering plan, generates black-box end-to-end product tests, builds into `app/<spec-name>/` (git-ignored so generated products stay out of Git), runs verification & fixes, runs an optional Dual Post-Build Supervisor audit, publishes artifacts/Docker images in CI, and records token telemetry in `usage/ledger.jsonl`.

```text
spec -> parse -> research + clarify -> judge + E2E tests -> build (app/<spec-name>/) -> verify + fix -> dual supervisor -> report + ledger
```

The current deliverable is the GitHub template and CLI. A hosted app or online framework is a later step; this repository does not include one.

## Start from this template

1. Push this repository to GitHub and enable **Settings → General → Template repository**.
2. Choose **Use this template → Create a new repository** for each project.
3. In the new repository's **Settings → Actions → General**, allow GitHub Actions to create pull requests and set workflow permissions to **Read and write permissions**.
4. Add provider keys under **Settings → Secrets and variables → Actions → Secrets**. By default `config/models.yaml` uses `OPENROUTER_API_KEY` as primary and `GEMINI_API_KEY` as fallback (`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are also passed through if you use direct providers).
5. Write a spec at `specs/<project-name>.md`, using [specs/TEMPLATE.md](specs/TEMPLATE.md) as a guide.
6. Push the spec to `main`, or run **Actions → Spec to App → Run workflow** and select the spec. A push builds the one spec it added or changed (in any of its commits) into `app/<spec-name>/`. A push that changes several specs stops with an error instead of guessing; the first commit of a repository created from the template never starts a build.
7. Download the built product from the run's `atlas-product-<spec-name>` artifact (or pull its Docker image from `ghcr.io`), and review the opened pull request containing the updated token ledger (`usage/ledger.jsonl`) and full verification/token report. The same workflow then runs `ci.yml` on the PR branch, because GitHub does not start other workflows for a PR opened with the built-in token.

The GitHub build skips the optional research and judge stages on a spec push to keep the default run smaller. For a manual run, uncheck **Skip research & judge** to include those stages. The generated app is tested before its pull request is opened. The test subprocesses do not receive the provider keys or other common credentials from the build job.

## Run locally

Python 3.11 or newer is required. Node.js 20 is needed only when generating or verifying a Node app.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Or add the optional local Antigravity SDK harness:
# pip install -e ".[dev,harnesses]"
# Equivalent pip requirements files are also provided:
# pip install -r requirements-dev.txt -r requirements-harnesses.txt
cp .env.example .env
# Add a provider key and configure the matching model IDs in config/models.yaml (or config/harness.yaml).

atlas validate-config
atlas build specs/example-app.md --skip-research --clean
```

`atlas validate-config` checks configuration structure and limits. It does not check that a provider key is valid or that an account can use every configured model.

The sample config routes primary roles through OpenRouter (`openrouter/...`) with direct Google Gemini API (`gemini/...`) fallbacks: `openrouter/google/gemini-4-hydrogen` for `parser` and `builder` (with a commented local `ollama_chat/gemma4:26b` option for `builder` — local Ollama models work only locally, not on GitHub-hosted runners), `openrouter/google/gemini-4-neon` for `researcher`, `openrouter/anthropic/claude-opus-5.5` for `judge`, and `openrouter/anthropic/claude-sonnet-5.5` for `e2e` and `fixer`.

| Command | What it does |
| :--- | :--- |
| `atlas build <spec.md>` (or `atlas run <spec.md>`) | Parse the spec, research options + clarify ambiguities (`--clarify` / `--answers`), plan, generate authoritative E2E product tests (`--e2e-tests`), build the app under `app/<spec-name>/`, verify & fix, run optional Supervisor review, and write run artifacts. Pass `--from-run <run-dir>` to resume straight from build/verify using a prior run's plan and E2E tests. Streams live `[HH:MM:SS]` stage, `k/N` requirement, and check progress to `stderr` (`-q` / `--quiet` to suppress, `--verbose` for debug logs). |
| `atlas build <spec.md> --skip-research --clean` | Skip research and planning stages (still clarifying `spec.open_questions` and optional E2E tests) and replace the contents of `app/<spec-name>/` with the generated app. |
| `atlas plan <spec.md>` | Create a planning report (including clarifying questions & optional E2E test plan) without generating an app, streaming live stage progress (`-q` / `--quiet` to suppress, `--ui` to view in browser). |
| `atlas ui [runs/<run-id>]` | Open or replay the workflow UI for a past or in-progress run (`Ctrl+C` to stop; defaults to newest run under `runs/`). |
| `atlas verify --app-dir app/<spec-name>` | Run verification on an existing built app without making model calls. |
| `atlas app-dir <spec.md>` | Print the default product output folder (`app/<spec-name>`) for a spec. |
| `atlas tokens [runs/<run-id>]` | Show usage for a run; without a run path, show the newest run under `runs/`. |
| `atlas tokens --all` | Show every recorded run from the project ledger `usage/ledger.jsonl`: totals by spec, model, and stage. |
| `atlas changed-specs --before <sha> --after <sha>` | List specs added or changed between two commits (used by the workflow). |
| `atlas validate-config [--config <path>]` | Check the model roles, harness backends, fallbacks, skills allowlist, and configured limits. |
| `atlas doctor [--config <path>] [--live]` | Check CLI binaries, auth status (`agy`, `claude`, `OPENROUTER_API_KEY`), Antigravity SDK, local model daemons, and allowlisted skills. |
| `atlas detect-local [--config <path>]` | Probe Ollama and LM Studio, rank eligible local models by tool support, context, memory budget, and tool smoke check. |

Verification currently supports Python and Node.js projects. Python checks include compilation and `pytest`, plus `ruff` when installed. Node checks run the package test script or Node's test runner, along with any declared `typecheck`, `build`, and `lint` scripts. A project with no tests fails verification. Other stacks need verification support added before Atlas can verify them automatically.

`atlas build` also runs a `spec-coverage` check: every functional requirement the parser found (`R001`, `R002`, ...) must be mentioned in at least one test file, in the test's name or a comment next to it. A missing ID fails verification and goes to the fixer like any other failed check, so a generated suite cannot quietly skip a requirement. It is a text search, not proof that a test is meaningful, so still review the PR. Turn it off with `verification.require_requirement_tests: false` in `config/models.yaml`.

Each role can have its own output limit and timeout (`limits.max_tokens_by_role`, `limits.timeout_seconds_by_role`). The builder returns the whole app in one reply, so it gets far more than the other roles. When a model stops at its limit, Atlas stops and reports which setting to raise instead of a JSON parse error. It does not try fallback models: they share the same limit and prompt, so each would be cut off and billed too.

## Hybrid local-first harness mode (`config/harness.yaml`)

Instead of calling every stage via OpenRouter APIs, [`config/harness.yaml`](config/harness.yaml) runs a hybrid local-first workflow on your machine:

| Role | Backend / Harness | Behavior |
| :--- | :--- | :--- |
| **`parser` & `researcher`** | Antigravity CLI (`agy`) | Structured `--json-schema` + `--disable-slash-commands` single-turn calls (`gemini-4-hydrogen-high` / `gemini-4-neon-high`, `420s` timeout & `24K` fallback cap) with automatic fallback to `ModelRouter`. |
| **`e2e`** | Claude Code (`claude`, read-only) | Claude Sonnet (`sonnet`, `--effort high`, zero write tools `--tools ""`, subscription login, `600s` timeout) generates black-box E2E tests and falls back to OpenRouter (`openrouter/anthropic/claude-sonnet-5.5`) only if `claude` errors. |
| **`judge`** | Claude Code (`claude`, read-only) | Claude Opus 5.5 evaluates researched options per requirement with zero write tools (`--tools ""`, subscription billing guard), falling back to OpenRouter only if `claude` errors. |
| **`builder`** | Antigravity Python SDK (`antigravity_sdk`) → offload | **Gemma 4 12B on your Mac** via LM Studio (`google/gemma-4-12b-qat`) or Ollama (`gemma4:12b-mlx` / `gemma4:12b`); nothing else is picked (`only_preferred`, `max_parameter_size_b: 14`). Checks tool calling (`ping` smoke test) and the context window the server actually loaded (`>= 48K`). It reads a short `BUILD_BRIEF.md` instead of the full `PLAN.md`. If the 12B can't finish — no model ready, error, `900s` timeout, no file written for `stall_seconds: 300`, or nothing built — the build is **offloaded** to Claude Sonnet (`claude_code` workspace harness using your subscription login, file-only tools `Read,Grep,Glob,Edit,Write`), then Gemini 4 Hydrogen (`agy` workspace harness), and finally OpenRouter only if those harnesses error. The run manifest records `build_offloaded_to` and `build_offload_harness`. |
| **`fixer`** | Claude Code (`claude`, read-only) ↔ Antigravity CLI (`agy`) | Sequential `PatchResult` repairs alternating between `claude_code` (`sonnet`, subscription, `--tools ""`) and `agy` (`gemini-4-hydrogen-high`), falling back to OpenRouter only if the active harness errors, receiving previous diagnoses + `writing-clean-code`. |
| **`supervisor`** | Dual Post-Build Panel (parallel) | Read-only Claude Opus (`claude_code`, fallback OpenRouter) + Gemini 4 Neon (`agy`, fallback OpenRouter) inspect the verified app independently and merge issues + an improvement plan back into the Fixer loop. |

*(Pass `--config config/harness.yaml` or export `ATLAS_CONFIG=config/harness.yaml` so Atlas does not default to [`config/models.yaml`](config/models.yaml).)*

```bash
lms unload --all && lms load google/gemma-4-12b-qat --context-length 49152   # LM Studio
# or: OLLAMA_CONTEXT_LENGTH=49152 ollama serve   (then: ollama pull gemma4:12b)
atlas doctor --config config/harness.yaml
atlas detect-local --config config/harness.yaml
atlas run specs/example-app.md --config config/harness.yaml
```

## Live workflow UI

`atlas run` / `atlas build` in an interactive terminal opens a live view of the run in your browser (served from `127.0.0.1` only, read-only, no extra dependencies). The pipeline is drawn as a horizontal left-to-right transit strip map: each stage is a station with its own symbol and a status marker, research and judge show `k of N` progress per requirement (highlighting any gaps), E2E contract shows the configured target (such as `sonnet (high)` via `claude_code`) and test files, Build shows the local 12B and its offload stops (`sonnet` via `claude_code`, `hydrogen` via `agy`) with a live stall-guard bar and updated workspace files, Verify shows each check per round, the fix loop branches below Verify, Review shows each reviewer's verdict, and Report summarizes token/cost usage by stage. Click any station for details, artifact previews (`spec.md`, `plan.md`, `e2e_tests.json`, `verification.json`, `supervisor_review.json`, `report.md`), and filtered logs.

```bash
atlas run specs/example-app.md --config config/harness.yaml            # opens the UI
atlas run specs/example-app.md --no-ui                                  # terminal only
atlas run specs/example-app.md --ui --no-open --ui-port 8765            # print the URL only
atlas run specs/example-app.md --ui-keep-open                           # stay alive until Ctrl+C
atlas plan specs/example-app.md --ui                                    # opt-in UI for plan
atlas ui                                       # replay the newest run (Ctrl+C to stop)
atlas ui runs/<run-id>                         # replay a specific run
```

The UI stays off in CI (`CI` / `GITHUB_ACTIONS`) and when stderr isn't a terminal (`atlas plan` stays off by default unless `--ui` is passed). Clarifying questions are still answered in Terminal; the browser shows that Atlas is waiting. Every run writes `runs/<run-id>/events.jsonl`, which `atlas ui` replays (and older runs are rebuilt from their saved JSON artifacts). See [docs/VISUAL_WORKFLOW_UI_PLAN.md](docs/VISUAL_WORKFLOW_UI_PLAN.md).

## Research clarifications, pre-seeded E2E tests & lean local prompts

- **Clarifying questions (`clarify.py`):** Research surfaces `clarifying_questions` alongside `spec.open_questions`. In a terminal (`--clarify` or TTY auto), Atlas prompts you cleanly and folds answers into `spec.assumptions` before planning and building; in non-TTY/CI runs, pass `--answers answers.json` (never blocks by default) and review `runs/<run-id>/clarifications.json`.
- **Authoritative black-box E2E tests (`e2e.py`):** When `verification.require_e2e_tests: true` (default in `config/harness.yaml`) or `--e2e-tests` is set, Atlas generates an external `product_interface` contract and runnable E2E suite (or sets `verifiable: false` with `unverifiable_reason` when offline testing is infeasible).
- **Workspace-first pre-seeding & lean context:** Before the local Builder starts, Atlas writes `PLAN.md`, a short `BUILD_BRIEF.md` (chosen approach per requirement, clarification answers, tasks; about 6K tokens instead of PLAN.md's ~30K) and the authoritative `tests/` E2E files into `app/<spec-name>/`, symlinks only allowlisted skills (`discovery_mode: progressive`) into `.agents/skills/`, and sends a compact prompt capped at 12K chars so smaller local context windows stay focused. After the Builder exits, Atlas removes temporary skill symlinks, restores the authoritative E2E files, and enforces `spec-coverage`, `product-contract`, and `e2e-product-tests` in [`verify.py`](src/atlas/pipeline/verify.py).

## Token and cost reports

Every run writes an audit folder under `runs/<run-id>/`. The same summary appears in `report.md`, the CLI output, the GitHub Actions summary, and the generated pull request.

- `tokens.json` records prompt, completion, cached, reasoning, and total tokens, by model and pipeline stage, with one record per model call.
- `cost_usd` is an estimate calculated from LiteLLM's model pricing data. It may differ from the provider's final bill.
- If a provider does not return token usage for a call, that call is marked as unreported. Its token count cannot be recovered from this report.
- Replies that Atlas cannot use (empty, or cut off at the output limit) are still counted, because providers normally bill them. The report notes how many there were.
- **Project ledger:** every `atlas build` and `atlas plan` appends one line to `usage/ledger.jsonl` (totals plus per-stage and per-model breakdown). The workflow commits it with the generated app, so the project's full token history stays in the repository after run folders and CI artifacts expire. `.gitattributes` merges it line by line. Use `--ledger <path>` to write elsewhere or `--no-ledger` to skip it.
- `clarifications.json` *(when open/research questions exist)* stores the clarifying questions and user answers folded into `spec.assumptions` before planning and building.
- `e2e_tests.json` *(when `verification.require_e2e_tests: true` or `--e2e-tests`)* stores the independent black-box E2E product contract and test files (`verifiable: true` test files are pre-seeded into `app/<spec-name>/PLAN.md` + `tests/` before local builds start, or `verifiable: false` explains why offline E2E verification is unavailable).
- `verification.json` stores each test/check result (including `e2e-product-tests` integrity checks) and any fix rounds. Verification runs do not make model calls; any fixes requested after failed checks do.
- `supervisor_review.json` *(when `backends.supervisor.enabled: true`)* stores the unified post-build review verdict, merged issues, improvement plan, and individual reviewer verdicts (for example Claude Opus + Gemini 4 Neon in parallel).
- `manifest.json` contains the run status, generated file list, and a compact usage summary.

Use `atlas tokens runs/<run-id> --json` for the complete machine-readable breakdown of one run, and `atlas tokens --all` (or `--all --json`) for every run. The optional `max_total_tokens` and `max_cost_usd` settings in `config/models.yaml` stop a run before another model call once recorded usage reaches the configured limit. The check runs right before each call is sent, so parallel research and judge calls respect it; calls already in flight (at most `max_parallel_calls`) can still finish. Leaving them as `null` means no configured cap.

## Main files

- `specs/` — project specifications and the reusable spec guide.
- `requirements.txt`, `requirements-dev.txt`, `requirements-harnesses.txt` — pip-compatible dependency manifests mirrored from [`pyproject.toml`](pyproject.toml).
- `app/` — built products (`app/<spec-name>/`); git-ignored (`app/*`, keeping `!app/.gitkeep`) so generated apps are never pushed to Git, and instead uploaded as `atlas-product-<spec-name>` artifacts and published as Docker images.
- `src/atlas/` — CLI, pipeline stages (`parse`, `research`, `clarify`, `judge`, `e2e`, `build`/`fix`/`supervise`, `verify`), harness adapters (`agy`, `antigravity_sdk`, `claude_code`), skills loader, local model discovery, and token telemetry.
- `config/models.yaml` — pure cloud OpenRouter model routing, fallbacks, per-role output limits and timeouts, optional budgets, and verification switches.
- `config/harness.yaml` — local-first harness profile (`agy` `parser`/`researcher` with OpenRouter fallback, read-only `claude` `judge` Opus & `e2e` Sonnet `--effort high`, Dual Post-Build Supervisor panel pairing Claude Opus + Gemini 4 Neon, Antigravity SDK local builder with ordered `sonnet -> hydrogen` offload, alternating `claude_code`/`agy` fixer, allowlisted skills, and pre-seeded E2E tests).
- `usage/ledger.jsonl` — one line per run with its token totals; created by the first run and updated via PRs.
- `.github/workflows/spec-to-app.yml` — spec-triggered build into `app/<spec-name>/`, verification, product/run artifact upload, `cd.yml` Docker publish, `usage/ledger.jsonl` PR, then `ci.yml` on that PR branch.
- `.github/workflows/ci.yml` — Atlas quality checks and verification for generated app changes on pull requests (also called by `spec-to-app.yml`).
- `.github/workflows/cd.yml` — reusable Docker build and push to GitHub Container Registry (`ghcr.io/<owner>/<repo>/<spec-name>`) triggered by `spec-to-app.yml` when verification passes.
- `runs/` — local run artifacts; ignored by Git and retained as GitHub Actions artifacts for cloud runs.
- [ARCHITECTURE.md](ARCHITECTURE.md) — plain async Python orchestration, end-to-end spec-to-app flow, model routing, and progressive skills.
- [HARNESS_PLAN.md](docs/HARNESS_PLAN.md) — hybrid local-first builder + CLI harness design, auth pre-flight (`atlas doctor`), honest telemetry, dual post-build supervisor, and skills allowlist.
- [TESTING_AND_DEPLOYING.md](docs/TESTING_AND_DEPLOYING.md) — step-by-step local validation, hybrid harness setup, and GitHub template deployment guide.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline & skills breakdown, and [ROADMAP.md](ROADMAP.md) for template experiments and the later hosted-framework direction.
