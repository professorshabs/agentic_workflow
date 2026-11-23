# Atlas — Spec-to-App GitHub Template

Atlas is a small repository template for turning a Markdown specification into a runnable app. Its GitHub Actions pipeline builds the app, runs its tests and checks, tries a limited number of fixes, and opens a pull request with the verification and token report.

```text
spec -> build -> test and verify -> fix failures -> report tokens and cost -> pull request
```

The current deliverable is the GitHub template and CLI. A hosted app or online framework is a later step; this repository does not include one.

## Start from this template

1. Push this repository to GitHub and enable **Settings → General → Template repository**.
2. Choose **Use this template → Create a new repository** for each project.
3. In the new repository's **Settings → Actions → General**, allow GitHub Actions to create pull requests and set workflow permissions to **Read and write permissions**.
4. Add provider keys under **Settings → Secrets and variables → Actions → Secrets**. By default `config/models.yaml` uses `OPENROUTER_API_KEY` as primary and `GEMINI_API_KEY` as fallback (`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are also passed through if you use direct providers).
5. Write a spec at `specs/<project-name>.md`, using [specs/TEMPLATE.md](specs/TEMPLATE.md) as a guide.
6. Push the spec to `main`, or run **Actions → Spec to App → Run workflow** and select the spec. A push builds the one spec it added or changed (in any of its commits). A push that changes several specs stops with an error instead of guessing; the first commit of a repository created from the template never starts a build.
7. Review the generated app pull request. Its description includes the test results and token report; the workflow run also retains the full JSON artifacts. The same workflow then runs `ci.yml` on the PR branch, because GitHub does not start other workflows for a PR opened with the built-in token.

The GitHub build skips the optional research and judge stages on a spec push to keep the default run smaller. For a manual run, uncheck **Skip research & judge** to include those stages. The generated app is tested before its pull request is opened. The test subprocesses do not receive the provider keys or other common credentials from the build job.

## Run locally

Python 3.11 or newer is required. Node.js 20 is needed only when generating or verifying a Node app.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Add a provider key and configure the matching model IDs in config/models.yaml.

atlas validate-config
atlas build specs/example-app.md --skip-research --clean
```

`atlas validate-config` checks configuration structure and limits. It does not check that a provider key is valid or that an account can use every configured model.

The sample config routes primary roles through OpenRouter (`openrouter/...`) with direct Google Gemini API (`gemini/...`) fallbacks: `openrouter/google/gemini-4-hydrogen` for `parser` and `builder` (with commented OpenRouter/Gemini Gemma 4 and local `ollama_chat/gemma4:26b` options for `builder` — local Ollama models work only locally, not on GitHub-hosted runners), parallel `researcher` calls across `openrouter/anthropic/claude-opus-5.5`, `openrouter/google/gemini-4-neon`, and `openrouter/openai/gpt-6.2-sol`, `openrouter/anthropic/claude-opus-5` for `judge`, and `openrouter/anthropic/claude-sonnet-5.5` for `fixer`.

| Command | What it does |
| :--- | :--- |
| `atlas build <spec.md>` | Parse the spec, optionally research and plan, generate the app, test and verify it, try fixes, and write run artifacts. |
| `atlas build <spec.md> --skip-research --clean` | Skip research and planning stages and replace the contents of `app/` with the generated app. |
| `atlas plan <spec.md>` | Create a planning report without generating an app. |
| `atlas verify --app-dir app` | Run verification on an existing app without making model calls. |
| `atlas tokens [runs/<run-id>]` | Show usage for a run; without a run path, show the newest run under `runs/`. |
| `atlas tokens --all` | Show every recorded run from the project ledger `usage/ledger.jsonl`: totals by spec, model, and stage. |
| `atlas changed-specs --before <sha> --after <sha>` | List specs added or changed between two commits (used by the workflow). |
| `atlas validate-config` | Check the model roles, fallbacks, and configured limits. |

Verification currently supports Python and Node.js projects. Python checks include compilation and `pytest`, plus `ruff` when installed. Node checks run the package test script or Node's test runner, along with any declared `typecheck`, `build`, and `lint` scripts. A project with no tests fails verification. Other stacks need verification support added before Atlas can verify them automatically.

`atlas build` also runs a `spec-coverage` check: every functional requirement the parser found (`R001`, `R002`, ...) must be mentioned in at least one test file, in the test's name or a comment next to it. A missing ID fails verification and goes to the fixer like any other failed check, so a generated suite cannot quietly skip a requirement. It is a text search, not proof that a test is meaningful, so still review the PR. Turn it off with `verification.require_requirement_tests: false` in `config/models.yaml`.

Each role can have its own output limit and timeout (`limits.max_tokens_by_role`, `limits.timeout_seconds_by_role`). The builder returns the whole app in one reply, so it gets far more than the other roles. When a model stops at its limit, Atlas reports which setting to raise instead of a JSON parse error, and tries the next fallback model instead of repeating the same cut-off request.

## Token and cost reports

Every run writes an audit folder under `runs/<run-id>/`. The same summary appears in `report.md`, the CLI output, the GitHub Actions summary, and the generated pull request.

- `tokens.json` records prompt, completion, cached, reasoning, and total tokens, by model and pipeline stage, with one record per model call.
- `cost_usd` is an estimate calculated from LiteLLM's model pricing data. It may differ from the provider's final bill.
- If a provider does not return token usage for a call, that call is marked as unreported. Its token count cannot be recovered from this report.
- Replies that Atlas cannot use (empty, or cut off at the output limit) are still counted, because providers normally bill them. The report notes how many there were.
- **Project ledger:** every `atlas build` and `atlas plan` appends one line to `usage/ledger.jsonl` (totals plus per-stage and per-model breakdown). The workflow commits it with the generated app, so the project's full token history stays in the repository after run folders and CI artifacts expire. `.gitattributes` merges it line by line. Use `--ledger <path>` to write elsewhere or `--no-ledger` to skip it.
- `verification.json` stores each test/check result and any fix rounds. Verification runs do not make model calls; any fixes requested after failed checks do.
- `manifest.json` contains the run status, generated file list, and a compact usage summary.

Use `atlas tokens runs/<run-id> --json` for the complete machine-readable breakdown of one run, and `atlas tokens --all` (or `--all --json`) for every run. The optional `max_total_tokens` and `max_cost_usd` settings in `config/models.yaml` stop a run before another model call once recorded usage reaches the configured limit. Leaving them as `null` means no configured cap.

## Main files

- `specs/` — project specifications and the reusable spec guide.
- `app/` — generated application; review its pull request before using it.
- `src/atlas/` — CLI, generation pipeline, verification, and token reporting.
- `config/models.yaml` — model routing, fallbacks, per-role output limits and timeouts, optional budgets, and the requirement coverage switch.
- `usage/ledger.jsonl` — one line per run with its token totals; created by the first run and committed with each generated app.
- `.github/workflows/spec-to-app.yml` — spec-triggered build, verification, artifact upload, pull request, then `ci.yml` on that PR branch.
- `.github/workflows/ci.yml` — Atlas quality checks and verification for generated app changes on pull requests (also called by `spec-to-app.yml`).
- `runs/` — local run artifacts; ignored by Git and retained as GitHub Actions artifacts for cloud runs.

See [ROADMAP.md](ROADMAP.md) for template experiments and the later hosted-framework direction.
