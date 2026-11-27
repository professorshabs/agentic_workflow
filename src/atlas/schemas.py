import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_TEST_FILE_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|specs?)/|(^|/)test_[^/]+$|_test\.[^/]+$|\.(test|spec)\.[^/]+$",
    flags=re.IGNORECASE,
)


def _validate_relative_app_path(path: str) -> str:
    raw = path.strip()
    if not raw:
        raise ValueError("File path must not be empty")
    if raw.startswith(("/", "\\")):
        raise ValueError(f"Absolute paths are not allowed: {path}")
    normalized = raw.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
        raise ValueError(f"Absolute drive paths are not allowed: {path}")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if not parts:
        raise ValueError(f"Invalid relative file path: {path}")
    if any(part == ".." for part in parts):
        raise ValueError(f"Parent traversal '..' is not allowed: {path}")
    if parts[0] in {".git", ".github"}:
        raise ValueError(f"Writing inside {parts[0]} is not allowed: {path}")
    return "/".join(parts)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Requirement(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: Literal["functional", "non_functional", "constraint"] = "functional"
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ParsedSpec(StrictModel):
    summary: str = Field(min_length=1)
    requirements: list[Requirement] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_requirements_graph(self) -> "ParsedSpec":
        ids = [r.id for r in self.requirements]
        id_set = set(ids)
        if len(ids) != len(id_set):
            raise ValueError("Requirement IDs must be unique")

        for req in self.requirements:
            if req.id in req.depends_on:
                raise ValueError(f"Requirement '{req.id}' cannot depend on itself")

        unknown = {dep for r in self.requirements for dep in r.depends_on} - id_set
        if unknown:
            raise ValueError(f"Unknown requirement dependencies: {sorted(unknown)}")

        # Detect cycles in the requirement dependency graph (DFS 3-color).
        deps_map = {r.id: r.depends_on for r in self.requirements}
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(node: str) -> None:
            state[node] = 1
            for nxt in deps_map[node]:
                nxt_state = state.get(nxt, 0)
                if nxt_state == 1:
                    raise ValueError(f"Cyclic requirement dependency detected involving '{nxt}'")
                if nxt_state == 0:
                    dfs(nxt)
            state[node] = 2

        for req_id in ids:
            if state.get(req_id, 0) == 0:
                dfs(req_id)

        return self


class Evidence(StrictModel):
    claim: str = Field(min_length=1)
    source_title: str | None = None
    source_url: str | None = None
    source_type: Literal["official_docs", "repository", "article", "other", "model_knowledge"] = (
        "other"
    )
    checked_at: str | None = None


class CandidateTool(StrictModel):
    name: str = Field(min_length=1)
    category: str | None = None
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    licensing: str | None = None
    cost_notes: str | None = None
    integration_effort: Literal["low", "medium", "high", "unknown"] = "unknown"
    maintenance_burden: Literal["low", "medium", "high", "unknown"] = "unknown"
    docs_url: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    verification_flags: list[str] = Field(default_factory=list)


class ResearchNote(StrictModel):
    requirement_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    candidates: list[CandidateTool] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    e2e_scenarios: list[str] = Field(default_factory=list)


class ClarificationItem(StrictModel):
    """One clarifying question surfaced from the spec or research findings."""

    question: str = Field(min_length=1)
    answer: str = ""
    source: Literal["spec", "research"] = "research"
    requirement_id: str | None = None


class ClarificationReport(StrictModel):
    """Collected clarifying questions and any answers supplied by the user."""

    items: list[ClarificationItem] = Field(default_factory=list)
    interactive: bool = False


class ResearchFailure(StrictModel):
    requirement_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)


class ResearchReport(StrictModel):
    notes: list[ResearchNote] = Field(default_factory=list)
    failures: list[ResearchFailure] = Field(default_factory=list)


class ToolDecision(StrictModel):
    requirement_id: str = Field(min_length=1)
    options: list[CandidateTool]
    pick: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] = "medium"
    needs_verification: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pick(self) -> "ToolDecision":
        if not 2 <= len(self.options) <= 3:
            raise ValueError("Each decision must contain 2-3 candidate options")
        normalized = [o.name.lower() for o in self.options]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Candidate names must be unique within a decision (case-insensitive)")
        names = [o.name for o in self.options]
        if self.pick not in names:
            raise ValueError("The selected tool must be one of the candidate options")
        return self


class JudgeOutput(StrictModel):
    decisions: list[ToolDecision]
    risks: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)


class Plan(StrictModel):
    spec: ParsedSpec
    decisions: list[ToolDecision]
    risks: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decisions(self) -> "Plan":
        expected = {r.id for r in self.spec.requirements}
        actual = [d.requirement_id for d in self.decisions]
        if len(actual) != len(set(actual)):
            raise ValueError("Duplicate requirement decisions")
        if set(actual) != expected:
            raise ValueError("There must be exactly one decision per requirement")
        return self

    def content_sha256(self) -> str:
        """Return a deterministic SHA-256 hash of the validated plan for approval binding."""
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ApprovalRecord(StrictModel):
    run_id: str = Field(min_length=1)
    plan_sha256: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_at: str = Field(min_length=1)
    source: Literal["cli", "api", "telegram", "github"] = "api"
    notes: str | None = None


class SupervisorOutput(StrictModel):
    """Structured per-reviewer acceptance verdict emitted by a post-build supervisor."""

    approved: bool
    summary: str = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    improvement_plan: list[str] = Field(default_factory=list)
    requirement_notes: dict[str, str] = Field(default_factory=dict)


class ReviewerVerdict(StrictModel):
    """Recorded verdict from one named supervisor reviewer."""

    reviewer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    harness: str = Field(min_length=1)
    approved: bool
    summary: str = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    improvement_plan: list[str] = Field(default_factory=list)
    requirement_notes: dict[str, str] = Field(default_factory=dict)


class Review(StrictModel):
    """Unified Supervisor acceptance review across verification and requirements."""

    approved: bool
    summary: str = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    improvement_plan: list[str] = Field(default_factory=list)
    requirement_notes: dict[str, str] = Field(default_factory=dict)
    reviewers: list[ReviewerVerdict] = Field(default_factory=list)


class GeneratedFile(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    path: str = Field(min_length=1)
    content: str
    purpose: str = ""

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_app_path(value)

    @field_validator("purpose")
    @classmethod
    def strip_purpose(cls, value: str) -> str:
        return value.strip()


class GeneratedApp(StrictModel):
    summary: str = Field(min_length=1)
    stack: Literal["python", "node", "mixed", "other"] = "other"
    files: list[GeneratedFile] = Field(min_length=1)
    setup_commands: list[str] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    run_instructions: str = ""

    @model_validator(mode="after")
    def validate_unique_paths(self) -> "GeneratedApp":
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Generated file paths must be unique")
        return self


class E2ETestSuite(StrictModel):
    """Authoritative black-box end-to-end product tests generated before build."""

    summary: str = Field(min_length=1)
    product_interface: str = Field(min_length=1)
    stack: Literal["python", "node", "mixed", "other"] = "other"
    verifiable: bool = True
    unverifiable_reason: str | None = None
    files: list[GeneratedFile] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_e2e_test_files(self) -> "E2ETestSuite":
        if self.verifiable and not self.files:
            raise ValueError("Verifiable E2ETestSuite must include at least one test file")
        if not self.verifiable and not (self.unverifiable_reason or "").strip():
            raise ValueError(
                "Unverifiable E2ETestSuite must explain why in unverifiable_reason"
            )
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("E2E test file paths must be unique")
        non_test = [p for p in paths if not _TEST_FILE_PATH_RE.search(p)]
        if non_test:
            raise ValueError(
                f"E2ETestSuite may only create test files, got non-test path(s): {non_test}"
            )
        return self


class PatchResult(StrictModel):
    diagnosis: str = Field(min_length=1)
    files: list[GeneratedFile] = Field(default_factory=list)
    delete_files: list[str] = Field(default_factory=list)

    @field_validator("delete_files")
    @classmethod
    def validate_delete_paths(cls, values: list[str]) -> list[str]:
        return [_validate_relative_app_path(v) for v in values]

    @model_validator(mode="after")
    def validate_non_empty_patch(self) -> "PatchResult":
        if not self.files and not self.delete_files:
            raise ValueError("PatchResult must modify or delete at least one file")
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Patch file paths must be unique")
        return self


class WorkspaceBuildResult(StrictModel):
    """Build contract for agentic workspace harnesses that edit files directly on disk."""

    summary: str = Field(min_length=1)
    harness: str = Field(min_length=1)
    harness_version: str | None = None
    model: str = Field(min_length=1)
    app_dir: str = Field(min_length=1)
    stack: Literal["python", "node", "mixed", "other"] = "other"
    created_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    files_snapshot: list[str] = Field(default_factory=list)
    exit_code: int = 0
    run_instructions: str = ""
    available_skills: list[str] = Field(default_factory=list)
    used_skills: list[str] = Field(default_factory=list)
    skills_mode: Literal["progressive", "injected"] | None = None

    @field_validator("created_files", "modified_files", "deleted_files", "files_snapshot")
    @classmethod
    def validate_relative_paths(cls, values: list[str]) -> list[str]:
        return [_validate_relative_app_path(v) for v in values]


class CheckResult(StrictModel):
    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    passed: bool
    exit_code: int
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""


class VerificationRound(StrictModel):
    round_number: int = Field(ge=1)
    passed: bool
    stack_detected: str = Field(min_length=1)
    checks: list[CheckResult] = Field(default_factory=list)
    diagnosis: str | None = None


class VerificationReport(StrictModel):
    passed: bool
    stack_detected: str = Field(min_length=1)
    rounds: list[VerificationRound] = Field(default_factory=list)
    final_checks: list[CheckResult] = Field(default_factory=list)


CostSource = Literal["reported", "estimated", "subscription", "local_free", "unavailable"]
AuthMode = Literal["subscription", "oauth_token", "api_key", "cached_login", "local", "unknown"]


class ModelCallRecord(BaseModel):
    role: str = "unknown"
    requested_model: str = ""
    used_model: str = ""
    harness: str | None = None
    harness_version: str | None = None
    auth_mode: AuthMode | None = None
    attempts: int = 1
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    # What the call would cost at pay-per-token API prices: the real bill when there is one,
    # the harness's own figure (Claude Code), or tokens x a price list. None when unknown.
    api_cost_usd: float | None = None
    api_cost_source: str | None = None
    usage_reported: bool = False
    timestamp: str | None = None
    # Set when the provider returned a response Atlas could not use (empty, or cut off at
    # max_tokens). The call is usually still billed, so its tokens are counted.
    error: str | None = None


class UsageBucket(StrictModel):
    name: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_usd: float | None = None
    api_cost_usd: float | None = None
    elapsed_ms: float = 0.0


class TokenUsageSummary(StrictModel):
    total_calls: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_usd: float | None = None
    elapsed_ms: float = 0.0
    unreported_usage_calls: int = 0
    unpriced_calls: int = 0
    failed_calls: int = 0
    # API-equivalent estimate across all calls (see ModelCallRecord.api_cost_usd).
    api_cost_usd: float | None = None
    api_unestimated_calls: int = 0
    by_stage: list[UsageBucket] = Field(default_factory=list)
    by_model: list[UsageBucket] = Field(default_factory=list)
    calls: list[ModelCallRecord] = Field(default_factory=list)
