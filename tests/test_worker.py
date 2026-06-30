from __future__ import annotations

import pytest

from docsite_updater.ai import RuleBasedAIClient
from docsite_updater.config import Settings
from docsite_updater.github import InMemoryGitHubClient, SourcePullRequestFixture
from docsite_updater.models import AIReviewResult, AIUpdateResult, AIUsage, ChangedFile, DocFile, DocUpdate, DocUpdateJob, PullRequestContext, RepoRole
from docsite_updater.observability import MemoryLogger, MemoryMetrics
from docsite_updater.queue import PermanentJobError
from docsite_updater.validator import MockMkDocsValidator
from docsite_updater.worker import DocUpdateWorker, build_pr_body


def make_worker_context(*, diff: str, low_confidence: bool = False, invalid_output: bool = False, validator_passes: bool = True):
    settings = Settings(
        github_app_id="123",
        github_private_key="private",
        github_webhook_secret="webhook",
        docsite_repo="bnb-chain/mock-bnb-docsite",
        ai_api_key="ai-key",
        reviewer_logins=("docs-reviewer",),
    )
    job = DocUpdateJob(
        source_repo="bnb-chain/mock-bsc-app",
        pr_number=7,
        merge_commit_sha="abc123def4567890",
        default_branch="main",
        installation_id=42,
        source_pr_url="https://github.com/bnb-chain/mock-bsc-app/pull/7",
        correlation_id="request-1",
    )
    github = InMemoryGitHubClient(
        docsite_repo=settings.docsite_repo,
        source_prs={
            job.identity: SourcePullRequestFixture(
                repo=job.source_repo,
                number=job.pr_number,
                merge_commit_sha=job.merge_commit_sha,
                url=job.source_pr_url,
                diff=diff,
                changed_files=[ChangedFile(path="src/api.py", status="modified")],
            )
        },
        doc_files={"docs/api.md": "# API\n\nCurrent API documentation."},
    )
    logger = MemoryLogger()
    metrics = MemoryMetrics()
    worker = DocUpdateWorker(
        settings=settings,
        github=github,
        ai=RuleBasedAIClient(low_confidence=low_confidence, invalid_output=invalid_output),
        validator=MockMkDocsValidator(should_pass=validator_passes),
        logger=logger,
        metrics=metrics,
    )
    return worker, job, github, logger, metrics


class FixedAIClient:
    def __init__(self, result: AIUpdateResult) -> None:
        self.result = result

    def propose_doc_updates(self, pr_context: PullRequestContext, docs: list[DocFile]) -> AIUpdateResult:
        return self.result


class FixedReviewClient:
    def __init__(self, result: AIReviewResult | None = None, should_fail: bool = False) -> None:
        self.result = result or AIReviewResult(
            verdict="looks_good",
            summary="Generated PR matches the source diff.",
            findings=["No invented behavior found."],
            risk_level="low",
            usage=AIUsage(model="review-model", prompt_version="review-test"),
        )
        self.should_fail = should_fail
        self.calls: list[tuple[RepoRole, list[str]]] = []

    def review_generated_pr(
        self,
        *,
        pr_context: PullRequestContext,
        repo_role: RepoRole,
        pr_diff: str,
        changed_files: list[str],
    ) -> AIReviewResult:
        self.calls.append((repo_role, changed_files))
        if self.should_fail:
            raise RuntimeError("review unavailable")
        return self.result


def test_worker_creates_no_pr_when_ai_reports_no_doc_change() -> None:
    worker, job, github, _, metrics = make_worker_context(diff="NO_DOC_CHANGE: internal refactor")

    result = worker.run(job)

    assert result.status == "no_changes_needed"
    assert github.pull_requests == []
    assert metrics.counters["job.no_changes_needed"] == 1


def test_worker_creates_docsite_pr_for_relevant_api_change() -> None:
    worker, job, github, _, metrics = make_worker_context(diff="+ API endpoint get_validator now returns validator status")

    result = worker.run(job)

    assert result.status == "created_pr"
    assert result.pr is not None
    assert result.pr.url == "https://github.com/bnb-chain/mock-bnb-docsite/pull/1"
    assert result.pr.changed_files == ["docs/api.md"]
    assert "Automated Update" in github.doc_files["docs/api.md"]
    assert github.requested_reviewers[1] == ["docs-reviewer"]
    assert metrics.counters["docs_pr.created"] == 1


def test_worker_creates_docsite_and_playground_prs_for_source_api_change() -> None:
    settings = Settings(
        github_app_id="123",
        github_private_key="private",
        github_webhook_secret="webhook",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        playground_repo="koredeBNB/techops-docsite-interactive-playground",
        ai_api_key="ai-key",
        reviewer_logins=("docs-reviewer",),
    )
    job = DocUpdateJob(
        source_repo="koredeBNB/mock-bsc-app",
        pr_number=7,
        merge_commit_sha="abc123def4567890",
        default_branch="main",
        installation_id=42,
        source_pr_url="https://github.com/koredeBNB/mock-bsc-app/pull/7",
        correlation_id="request-1",
    )
    github = InMemoryGitHubClient(
        docsite_repo=settings.docsite_repo,
        playground_repo=settings.playground_repo,
        source_prs={
            job.identity: SourcePullRequestFixture(
                repo=job.source_repo,
                number=job.pr_number,
                merge_commit_sha=job.merge_commit_sha,
                url=job.source_pr_url,
                diff='+        "reward_rate": 0.12,',
                changed_files=[ChangedFile(path="src/mock_bsc_app/validators.py", status="modified")],
            )
        },
        doc_files={"docs/validators.md": "# Validators\n\nOld docs."},
        playground_files={"src/validatorStatus.ts": "export const response = { validator_id: 'validator-1' }\n"},
    )
    worker = DocUpdateWorker(
        settings=settings,
        github=github,
        ai=FixedAIClient(
            AIUpdateResult(
                status="updates",
                summary="Update docs and playground for validator response fields.",
                updates=[
                    DocUpdate(
                        path="docs/validators.md",
                        content="# Validators\n\nDocuments reward_rate.",
                        rationale="Docs must mention reward_rate.",
                        confidence=0.95,
                        repo_role="docsite",
                    ),
                    DocUpdate(
                        path="src/validatorStatus.ts",
                        content="export const response = { validator_id: 'validator-1', reward_rate: 0.12 }\n",
                        rationale="Playground response mirrors get_validator_status.",
                        confidence=0.95,
                        repo_role="playground",
                    ),
                ],
                usage=AIUsage(model="test", prompt_version="test"),
            )
        ),
        validator=MockMkDocsValidator(),
        logger=MemoryLogger(),
        metrics=MemoryMetrics(),
    )

    result = worker.run(job)

    assert result.status == "created_pr"
    assert len(result.prs) == 2
    assert [pr.repo_role for pr in result.prs] == ["docsite", "playground"]
    assert github.doc_files["docs/validators.md"] == "# Validators\n\nDocuments reward_rate."
    assert github.playground_files["src/validatorStatus.ts"] == "export const response = { validator_id: 'validator-1', reward_rate: 0.12 }\n"
    assert result.prs[0].url == "https://github.com/koredeBNB/mock-mkdocs-repo/pull/1"
    assert result.prs[1].url == "https://github.com/koredeBNB/techops-docsite-interactive-playground/pull/2"


def test_worker_posts_secondary_review_comment_after_pr_creation() -> None:
    worker, job, github, _, metrics = make_worker_context(diff="+ API endpoint get_validator now returns validator status")
    reviewer = FixedReviewClient()
    worker.reviewer = reviewer

    result = worker.run(job)

    assert result.status == "created_pr"
    assert reviewer.calls == [("docsite", ["docs/api.md"])]
    comments = github.pr_comments[("docsite", 1)]
    assert "AI Secondary Review" in comments[0]
    assert "Looks Good" in comments[0]
    assert "`review-model`" in comments[0]
    assert metrics.counters["ai_review.completed"] == 1
    assert metrics.counters["ai_review.comment_posted"] == 1


def test_worker_posts_review_unavailable_comment_when_secondary_review_fails() -> None:
    worker, job, github, logger, metrics = make_worker_context(diff="+ API endpoint get_validator now returns validator status")
    worker.reviewer = FixedReviewClient(should_fail=True)

    result = worker.run(job)

    assert result.status == "created_pr"
    comments = github.pr_comments[("docsite", 1)]
    assert "Verdict:** Not available" in comments[0]
    assert "review unavailable" in comments[0]
    assert metrics.counters["ai_review.failed"] == 1
    assert metrics.counters["ai_review.comment_posted"] == 1
    assert "ai_secondary_review_failed" in logger.as_json_lines()


def test_worker_rejects_ai_update_that_mutates_existing_numeric_value() -> None:
    settings = Settings(
        github_app_id="123",
        github_private_key="private",
        github_webhook_secret="webhook",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        ai_api_key="ai-key",
    )
    job = DocUpdateJob(
        source_repo="koredeBNB/mock-bsc-app",
        pr_number=7,
        merge_commit_sha="abc123def4567890",
        default_branch="main",
        installation_id=42,
        source_pr_url="https://github.com/koredeBNB/mock-bsc-app/pull/7",
        correlation_id="request-1",
    )
    original_doc = """# Validators

The response includes validator_id, status, and commission_rate.

Example response:
{
  "validator_id": "validator-1",
  "status": "active",
  "commission_rate": 0.05
}
"""
    hallucinated_doc = """# Validators

The response includes validator_id, status, commission_rate, and voting_power.

Example response:
{
  "validator_id": "validator-1",
  "status": "active",
  "commission_rate": -0.05,
  "voting_power": 1000
}
"""
    github = InMemoryGitHubClient(
        docsite_repo=settings.docsite_repo,
        source_prs={
            job.identity: SourcePullRequestFixture(
                repo=job.source_repo,
                number=job.pr_number,
                merge_commit_sha=job.merge_commit_sha,
                url=job.source_pr_url,
                diff='+        "voting_power": 1000,',
                changed_files=[ChangedFile(path="src/mock_bsc_app/validators.py", status="modified")],
            )
        },
        doc_files={"docs/validators.md": original_doc},
    )
    metrics = MemoryMetrics()
    logger = MemoryLogger()
    worker = DocUpdateWorker(
        settings=settings,
        github=github,
        ai=FixedAIClient(
            AIUpdateResult(
                status="updates",
                summary="Added voting_power.",
                updates=[
                    DocUpdate(
                        path="docs/validators.md",
                        content=hallucinated_doc,
                        rationale="API added voting_power.",
                        confidence=0.95,
                    )
                ],
                usage=AIUsage(model="test", prompt_version="test"),
            )
        ),
        validator=MockMkDocsValidator(),
        logger=logger,
        metrics=metrics,
    )

    result = worker.run(job)

    assert result.status == "safety_rejected"
    assert "0.05" in result.message
    assert "-0.05" in result.message
    assert github.pull_requests == []
    assert github.doc_files["docs/validators.md"] == original_doc
    assert metrics.counters["job.safety_rejected"] == 1
    assert "ai_update_safety_rejected" in logger.as_json_lines()


def test_worker_rejects_invalid_ai_output_safely() -> None:
    worker, job, github, _, _ = make_worker_context(diff="+ API changed", invalid_output=True)

    with pytest.raises(PermanentJobError):
        worker.run(job)

    assert github.pull_requests == []


def test_low_confidence_ai_output_does_not_create_pr() -> None:
    worker, job, github, _, metrics = make_worker_context(diff="+ API changed", low_confidence=True)

    result = worker.run(job)

    assert result.status == "low_confidence"
    assert github.pull_requests == []
    assert metrics.counters["job.low_confidence"] == 1


def test_worker_logs_prompt_and_model_audit_metadata() -> None:
    worker, job, _, logger, _ = make_worker_context(diff="+ API endpoint changed")

    worker.run(job)

    logs = logger.as_json_lines()
    assert "mock-rule-based" in logs
    assert "prototype-v1" in logs
    assert "token_usage" in logs
    assert job.identity in logs


def test_pr_body_contains_source_traceability_and_reviewer_checklist() -> None:
    job = DocUpdateJob(
        source_repo="bnb-chain/mock-bsc-app",
        pr_number=7,
        merge_commit_sha="abc123",
        default_branch="main",
        installation_id=42,
        source_pr_url="https://github.com/bnb-chain/mock-bsc-app/pull/7",
        correlation_id="request-1",
    )

    body = build_pr_body(job=job, summary="Docs updated.", changed_files=["docs/api.md"])

    assert "bnb-chain/mock-bsc-app" in body
    assert "https://github.com/bnb-chain/mock-bsc-app/pull/7" in body
    assert "abc123" in body
    assert "`docs/api.md`" in body
    assert "Reviewer Checklist" in body
