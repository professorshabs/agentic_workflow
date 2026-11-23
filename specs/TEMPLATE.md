# <Project name>

Describe the app and who will use it in a few sentences. State the main outcome the user should get.

## Users and use cases

- Who will use the app?
- What are the main actions they need to complete?

## Functional requirements

1. **<Requirement name>** — Describe the inputs, expected behavior, and outputs.
2. **<Requirement name>** — Describe error cases and edge cases.

Add details that make each requirement observable. Avoid broad instructions such as “make it secure” without saying what behavior is required.

## Constraints

- Preferred stack: Python 3.11+ or Node.js 20+ (automated verification currently supports these stacks).
- Dependencies: list required packages, or say “prefer the standard library”.
- Runtime environment: local machine, server, browser, or other target.
- Data and privacy: state where data is stored and what must not be sent elsewhere.
- External services: name required services and whether tests may call them. Prefer offline tests.

## Acceptance criteria

- [ ] Describe the expected result for each core use case.
- [ ] Describe behavior for invalid input and important edge cases.
- [ ] Include automated tests for the criteria above; tests must run without paid services or secrets.
- [ ] State the install, test, and run commands if they are not standard.

## Out of scope

- List features or integrations that must not be included in this version.
