# Step-by-Step: Testing Locally & Deploying to GitHub

Follow these steps in order to verify everything locally, safely push the template to GitHub, and test the live GitHub Actions pipeline end-to-end.

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
```

Run the full offline quality and test suite:

```bash
ruff check .
mypy src/ --ignore-missing-imports
pytest -v
atlas validate-config
```

**What passing looks like:**
- `ruff check .` prints `All checks passed!`
- `mypy src/ --ignore-missing-imports` prints `Success: no issues found ...`
- `pytest -v` passes all automated unit, verification, and workflow tests (no API keys required)
- `atlas validate-config` prints `Config valid: .../config/models.yaml`

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

### 3a. Fast live test (skips multi-model research & judge)

```bash
atlas build specs/example-app.md --skip-research --clean
```

### 3b. Full live test (includes parallel 3-model `researcher` + `judge` + `build` + `verify`)

```bash
atlas build specs/example-app.md --clean
```

### 3c. Re-run verification alone (0 LLM tokens)

```bash
atlas verify --app-dir app/example-app
```

### 3d. Inspect token usage and estimated cost

```bash
atlas tokens
atlas tokens --json
atlas tokens --all
```

- `atlas tokens` / `atlas tokens --json` shows the latest run's breakdown (including empty or truncated replies that were still billed).
- `atlas tokens --all` reads `usage/ledger.jsonl` and shows **cumulative totals across all runs** grouped by spec, model, and stage.

You can also inspect the full Markdown report and raw JSON telemetry for the newest run under `runs/<run-id>/`:
- `runs/<run-id>/report.md` — verification table, auto-fix diagnoses, generated files, and token/cost breakdown
- `runs/<run-id>/tokens.json` — per-stage, per-model, and per-call token + USD cost records
- `runs/<run-id>/verification.json` — exact check commands, exit codes, `stdout`, and `stderr`
- `runs/<run-id>/manifest.json` — overall run status (`verified`, `verification_failed`, or `failed`)
- `usage/ledger.jsonl` — append-only cross-run token ledger

### Quick Troubleshooting

| Symptom | Likely Cause & Fix |
| :--- | :--- |
| `All configured models failed` / 401 / 403 / 404 | Check `OPENROUTER_API_KEY` / `GEMINI_API_KEY` in `.env` and confirm the model IDs in `config/models.yaml` are enabled on your OpenRouter/Google account. |
| `OutputTruncatedError` / cut-off reply | Raise the per-role limit in `config/models.yaml` (`limits.max_tokens_by_role.builder` / `fixer`, or `limits.timeout_seconds_by_role`). |
| Missing requirement-ID test failure | With `verification.require_requirement_tests: true` in `config/models.yaml`, every functional requirement needs a test mentioning its ID (e.g. `R001`, `R002`). The fixer is asked to add it, or you can set this to `false`. |
| `Token budget exceeded` / `Cost budget exceeded` | Increase or set `null` on `max_total_tokens` / `max_cost_usd` in `config/models.yaml`. |
| `Build status: verification_failed` (exit code 1) | Inspect `runs/<run-id>/report.md` or `runs/<run-id>/verification.json` to see which check failed; increase `max_fix_rounds` or refine acceptance criteria in your spec. |

---

## Step 4 — Clean Local Test Ledger Before Pushing the Template

Built products inside `app/<spec-name>/` and `runs/` are already git-ignored (keeping `.gitkeep`), so they are **never pushed to Git**. If Step 3 recorded local test entries in `usage/ledger.jsonl` that you don't want in the clean template, remove `usage/`:

```bash
rm -rf usage
```

---

## Step 5 — Stage, Verify No Secrets Are Included, Commit & Push

Check what is modified or untracked:

```bash
git status
```

Stage changes and **verify `.env` and `runs/` are NOT staged**:

```bash
git add -A
git diff --cached --name-only
```

Commit and push `main` to GitHub:

```bash
git commit -m "chore: verify local changes and prepare GitHub template"
git remote add origin https://github.com/professorshabs/agentic_workflow.git
git push -u origin main
```

*(If `origin` is already configured, check with `git remote -v` and just run `git push -u origin main`. If your GitHub repo already has an initial commit, run `git pull --rebase origin main` first.)*

---

## Step 6 — Check Required GitHub Repository Settings

In `https://github.com/professorshabs/agentic_workflow/settings`:

1. **General**:
   - Check **Template repository** so you get the **Use this template** button.
2. **Actions → General**:
   - **Actions permissions**: select **Allow all actions and reusable workflows** and click **Save**.
   - **Workflow permissions**: select **Read and write permissions**, check **Allow GitHub Actions to create and approve pull requests**, and click **Save**.
3. **Secrets and variables → Actions → Secrets** *(use **Secrets**, NOT Variables)*:
   - Add **Repository secret** `OPENROUTER_API_KEY`
   - Add **Repository secret** `GEMINI_API_KEY`

---

## Step 7 — Test the Live GitHub Actions Pipeline

1. Open the **Actions** tab on GitHub → select **Spec to App** → click **Run workflow**.
2. Leave `spec_path` as `specs/example-app.md` (leave **Skip research & judge** checked for a quick test, or uncheck it to test full multi-model research + judge) and click **Run workflow**.
3. Once the run completes, confirm all four outcomes:
   - **Job Summary**: displays the verification table, **Token Usage & Cost** table, and published Docker image tags.
   - **Artifacts**: `atlas-product-<spec-name>` contains the built product (`app/<spec-name>/`) and `run-report.md`; `atlas-run-artifacts` contains `runs/`, `run-report.md`, and `usage/`.
   - **Pull Requests**: a new PR was opened for `usage/ledger.jsonl` with the full report as the PR description (marked **Draft** if verification failed) — `app/` itself is never committed to Git.
   - **PR CI Check**: `spec-to-app.yml` invokes the reusable **CI** workflow (`.github/workflows/ci.yml`) directly against the PR branch (`quality` + `verify-app`) with 0 LLM tokens.

---

## Step 8 — Test Creating a New Project From the Template

1. On your repository homepage, click **Use this template → Create a new repository** (creating the repo does **not** trigger a paid build on its initial commit).
2. In the **new** repository, re-apply Step 6 (`Secrets` and `Actions → General → Workflow permissions`) — GitHub does **not** copy repository secrets from templates.
3. Copy `specs/TEMPLATE.md` to `specs/001-my-app.md`, fill in your requirements and acceptance criteria, commit, and push to `main` (push one spec per push, or trigger each manually with **Actions → Spec to App → Run workflow**):

```bash
cp specs/TEMPLATE.md specs/001-my-app.md
# Edit specs/001-my-app.md
git add specs/001-my-app.md
git commit -m "feat: add 001-my-app spec"
git push origin main
```

GitHub Actions uses `atlas changed-specs` across the full push to detect `specs/001-my-app.md`, builds and verifies the app into `app/001-my-app/` (including requirement-ID test checks), uploads the product artifact, publishes its Docker image, appends to `usage/ledger.jsonl`, and opens a usage PR for review.

---

## Step 9 — Continuous Delivery (Docker to GHCR, Free)

When verification passes, `spec-to-app.yml` automatically calls [`.github/workflows/cd.yml`](../.github/workflows/cd.yml) with the `atlas-product-<spec-name>` artifact:

1. **Downloads the verified product** from the workflow artifact (no built code needs to be committed to Git).
2. **Builds a Docker image** using `Dockerfile` inside the built product (or a minimal Python `3.12-slim` / Node `20-slim` fallback Dockerfile if the generated app did not emit one).
3. **Publishes to GitHub Container Registry (`ghcr.io`)** using the built-in `GITHUB_TOKEN` (no extra secrets or paid services required), tagging `ghcr.io/professorshabs/agentic_workflow/<spec-name>:<short-sha>` and `:latest`:

```bash
docker pull ghcr.io/professorshabs/agentic_workflow/example-app:latest
docker run --rm ghcr.io/professorshabs/agentic_workflow/example-app:latest
```

4. **Optional manual approval gate**: in GitHub **Settings → Environments → `production`**, enable **Required reviewers** if you want every container release to wait for your approval before pushing to `ghcr.io`.
