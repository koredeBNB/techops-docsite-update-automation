from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


RepoRole = Literal["docsite", "playground"]


@dataclass(frozen=True)
class DocUpdateJob:
    source_repo: str
    pr_number: int
    merge_commit_sha: str
    default_branch: str
    installation_id: int
    source_pr_url: str
    correlation_id: str

    @property
    def identity(self) -> str:
        return f"{self.source_repo}#{self.pr_number}@{self.merge_commit_sha}"


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str


@dataclass(frozen=True)
class PullRequestContext:
    source_repo: str
    pr_number: int
    merge_commit_sha: str
    source_pr_url: str
    diff: str
    changed_files: list[ChangedFile]


@dataclass(frozen=True)
class DocFile:
    path: str
    content: str
    repo_role: RepoRole = "docsite"


@dataclass(frozen=True)
class DocUpdate:
    path: str
    content: str
    rationale: str
    confidence: float = 1.0
    repo_role: RepoRole = "docsite"


@dataclass(frozen=True)
class AIUsage:
    model: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class AIUpdateResult:
    status: Literal["updates", "no_changes_needed", "low_confidence"]
    summary: str
    updates: list[DocUpdate] = field(default_factory=list)
    usage: AIUsage = field(default_factory=lambda: AIUsage(model="mock", prompt_version="v1"))


@dataclass(frozen=True)
class AIReviewResult:
    verdict: Literal["looks_good", "needs_changes", "uncertain"]
    summary: str
    findings: list[str] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    usage: AIUsage = field(default_factory=lambda: AIUsage(model="mock", prompt_version="review-v1"))


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    output: str = ""


@dataclass(frozen=True)
class CreatedPullRequest:
    url: str
    number: int
    branch: str
    changed_files: list[str]
    repo_role: RepoRole = "docsite"
    repo: str = ""


@dataclass(frozen=True)
class WorkerResult:
    status: Literal["created_pr", "no_changes_needed", "low_confidence", "validation_failed", "safety_rejected"]
    pr: CreatedPullRequest | None = None
    prs: list[CreatedPullRequest] = field(default_factory=list)
    message: str = ""


class GitHubClient(Protocol):
    def create_installation_token(self, installation_id: int) -> str:
        ...

    def fetch_pr_context(self, job: DocUpdateJob) -> PullRequestContext:
        ...

    def list_doc_files(self) -> list[DocFile]:
        ...

    def list_playground_files(self) -> list[DocFile]:
        ...

    def create_docsite_branch(self, branch_name: str) -> None:
        ...

    def commit_doc_updates(self, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        ...

    def open_docsite_pr(self, branch_name: str, title: str, body: str, changed_files: list[str]) -> CreatedPullRequest:
        ...

    def request_reviewers(self, pr_number: int, reviewers: list[str]) -> None:
        ...

    def create_repository_branch(self, repo_role: RepoRole, branch_name: str) -> None:
        ...

    def commit_repository_updates(self, repo_role: RepoRole, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        ...

    def open_repository_pr(
        self,
        repo_role: RepoRole,
        branch_name: str,
        title: str,
        body: str,
        changed_files: list[str],
    ) -> CreatedPullRequest:
        ...

    def request_repository_reviewers(self, repo_role: RepoRole, pr_number: int, reviewers: list[str]) -> None:
        ...

    def fetch_repository_pr_diff(self, repo_role: RepoRole, pr_number: int) -> str:
        ...

    def comment_on_repository_pr(self, repo_role: RepoRole, pr_number: int, body: str) -> None:
        ...


class AIClient(Protocol):
    def propose_doc_updates(self, pr_context: PullRequestContext, docs: list[DocFile]) -> AIUpdateResult:
        ...


class AIReviewClient(Protocol):
    def review_generated_pr(
        self,
        *,
        pr_context: PullRequestContext,
        repo_role: RepoRole,
        pr_diff: str,
        changed_files: list[str],
    ) -> AIReviewResult:
        ...


class DocsiteValidator(Protocol):
    def validate(self, updates: list[DocUpdate]) -> ValidationResult:
        ...

    def validate_playground(self, updates: list[DocUpdate]) -> ValidationResult:
        ...


class Metrics(Protocol):
    def increment(self, name: str, value: int = 1) -> None:
        ...

    def observe(self, name: str, value: float) -> None:
        ...


class Logger(Protocol):
    def info(self, message: str, **fields: object) -> None:
        ...

    def warning(self, message: str, **fields: object) -> None:
        ...

    def error(self, message: str, **fields: object) -> None:
        ...
