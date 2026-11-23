# Tiny URL & Slug Generator CLI

Build a small Python CLI and library that converts titles into URL-safe slugs, tracks short codes in a local JSON file store, and resolves them back to target URLs.

## Functional Requirements
- `slugify(text: str) -> str` normalizes Unicode/punctuation into lowercase hyphen-separated ASCII slugs.
- `ShortenerStore(path)` persists URL mappings (`code -> url`) to a JSON file and supports `shorten(url, custom_code=None)` and `resolve(code)`.
- CLI entrypoint supports `shorten <url>` and `resolve <code>`.

## Constraints
- Stack: Python 3.11+ standard library + `pytest` for tests
- Must be self-contained and testable offline in CI

## Acceptance Criteria
- Duplicate `custom_code` raises a clear error
- Invalid URLs (not starting with `http://` or `https://`) are rejected
- Automated `pytest` suite verifies slugification, persistence, duplicates, and CLI behavior
