# Atlas Roadmap

## Current: GitHub template and CLI

The goal for this phase is a downloadable repository that accepts a written spec, generates a runnable app, verifies it, and reports model usage. There is no hosted builder or online framework in this version.

The template currently includes:

- A reusable Markdown spec guide and a small Python CLI example.
- A model-backed pipeline with optional research and planning stages.
- Python and Node.js app generation, test execution, and a limited auto-fix loop.
- Per-run token and estimated cost reports by stage, model, and call, plus a committed project ledger (`usage/ledger.jsonl`, `atlas tokens --all`) across runs.
- A `spec-coverage` check that every functional requirement has a test mentioning its ID, a `product-contract` check for standalone product hygiene, and researcher clarifying questions (`--clarify` / `--answers`) paired with authoritative black-box E2E product tests (`--e2e-tests`, dedicated `e2e` role/limits, `e2e-product-tests`).
- A local-first hybrid harness profile ([`config/harness.yaml`](config/harness.yaml)) with `agy` parser/researcher/e2e, read-only `claude` judge, Antigravity SDK local builder (`ollama` / `lm_studio`), allowlisted skills, alternating OpenRouter fixer, live step-by-step `stderr` progress output (`-q` / `--quiet`, `--verbose`), `atlas doctor` / `atlas detect-local`, and a parallel Dual Post-Build Supervisor (Claude Opus + Gemini 4 Neon) that feeds unified improvement plans back into the fixer loop.
- Per-role output limits and timeouts, with a clear error when a model's reply is cut off.
- Optional token, cost, timeout, and fix-round limits.
- GitHub Actions for spec-triggered generation, verification, report artifacts, pull requests, and repository quality checks.
- Credential filtering for subprocesses that install or execute generated application code.

## Next: validate the template on real projects

- [ ] Publish this repository as a GitHub template and confirm its Actions permissions and secrets setup.
- [ ] Run several small Python and Node specs in repositories created from the template.
- [ ] Compare normal runs with `--skip-research` and tune models and budgets from observed reports.
- [ ] Improve incremental builds so the generator can update an existing `app/` with awareness of its current files.
- [ ] Add more verification support when real projects need it (for example Go, TypeScript, or dbt).
- [ ] Decide whether a human plan approval step is useful for higher impact projects.

## Later: hosted application / online framework

Start this only after the template has been used enough to understand output quality, verification success, and typical token spend.

### Possible user flow

1. Write or select a spec in a project workspace.
2. Choose a model profile and usage budget.
3. Run generation in an isolated execution environment, with progress and verification results visible.
4. Review the generated files, test output, and token/cost breakdown.
5. Download the app or open a GitHub pull request.

### Possible building blocks

- A web UI for specs, generated files, and verification logs.
- A job queue and isolated runner for build and test commands.
- Persistent run history and a project-wide token ledger.
- GitHub authorization for creating branches and pull requests.

### Readiness checks

1. Real specs pass automated verification consistently, with a small number of fix rounds.
2. Token use and cost are measured across a range of projects and remain controllable with budgets.
3. The spec, generated app, verification, and token report formats are stable enough for clients to consume.
