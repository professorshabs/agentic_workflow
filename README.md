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
6. Push the spec to `main`, or run **Actions → Spec to App → Run workflow** and select the spec.
7. Review the generated app pull request. Its description includes the test results and token report; the workflow run also retains the full JSON artifacts.

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
| `atlas validate-config` | Check the model roles, fallbacks, and configured limits. |

Verification currently supports Python and Node.js projects. Python checks include compilation and `pytest`, plus `ruff` when installed. Node checks run the package test script or Node's test runner, along with any declared `typecheck`, `build`, and `lint` scripts. A project with no tests fails verification. Other stacks need verification support added before Atlas can verify them automatically.

## Token and cost reports

Every run writes an audit folder under `runs/<run-id>/`. The same summary appears in `report.md`, the CLI output, the GitHub Actions summary, and the generated pull request.

- `tokens.json` records prompt, completion, cached, reasoning, and total tokens, by model and pipeline stage, with one record per model call.
- `cost_usd` is an estimate calculated from LiteLLM's model pricing data. It may differ from the provider's final bill.
- If a provider does not return token usage for a call, that call is marked as unreported. Its token count cannot be recovered from this report.
- `verification.json` stores each test/check result and any fix rounds. Verification runs do not make model calls; any fixes requested after failed checks do.
- `manifest.json` contains the run status, generated file list, and a compact usage summary.

Use `atlas tokens runs/<run-id> --json` for the complete machine-readable breakdown. The optional `max_total_tokens` and `max_cost_usd` settings in `config/models.yaml` stop a run before another model call once recorded usage reaches the configured limit. Leaving them as `null` means no configured cap.

## Main files

- `specs/` — project specifications and the reusable spec guide.
- `app/` — generated application; review its pull request before using it.
- `src/atlas/` — CLI, generation pipeline, verification, and token reporting.
- `config/models.yaml` — model routing, fallbacks, timeouts, and optional budgets.
- `.github/workflows/spec-to-app.yml` — spec-triggered build, verification, artifact upload, and pull request.
- `.github/workflows/ci.yml` — Atlas quality checks and verification for generated app changes on pull requests.
- `runs/` — local run artifacts; ignored by Git and retained as GitHub Actions artifacts for cloud runs.

See [ROADMAP.md](ROADMAP.md) for template experiments and the later hosted-framework direction.
