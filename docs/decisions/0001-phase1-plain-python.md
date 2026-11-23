# ADR 0001: Plain Python pipeline for the template
Status: accepted

Use plain async functions for pipeline stages, with typed Pydantic models between them.
Keep orchestration in `atlas.runner` and expose it through the local CLI (`atlas.cli`).
This keeps the current template small and leaves the pipeline reusable if a hosted
application is designed later.

Deferred: hosted API, web UI, persistent database, job queue, and container sandbox.
