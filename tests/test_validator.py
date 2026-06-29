from __future__ import annotations

from docsite_updater.models import DocUpdate
from docsite_updater.validator import MockMkDocsValidator
from tests.test_worker import make_worker_context


def test_mock_mkdocs_validator_passes_valid_updates() -> None:
    validator = MockMkDocsValidator()
    update = DocUpdate(path="docs/api.md", content="# API\n\nValid markdown.", rationale="API changed")

    result = validator.validate([update])

    assert result.ok is True
    assert "passed" in result.output


def test_mock_mkdocs_validator_fails_broken_updates() -> None:
    validator = MockMkDocsValidator()
    update = DocUpdate(path="docs/api.md", content="# API\n\nBROKEN_MKDOCS", rationale="API changed")

    result = validator.validate([update])

    assert result.ok is False
    assert "mkdocs build failed" in result.output


def test_worker_skips_pr_creation_when_mkdocs_validation_fails() -> None:
    worker, job, github, logger, metrics = make_worker_context(diff="+ API endpoint changed", validator_passes=False)

    result = worker.run(job)

    assert result.status == "validation_failed"
    assert github.branches == set()
    assert github.pull_requests == []
    assert metrics.counters["job.validation_failed"] == 1
    assert "mkdocs_validation_failed" in logger.as_json_lines()


def test_worker_runs_validation_before_creating_branch() -> None:
    worker, job, github, _, _ = make_worker_context(diff="+ API endpoint changed")

    result = worker.run(job)

    assert result.status == "created_pr"
    assert github.branches == {"ai-docs/bnb-chain-mock-bsc-app-7-abc123de"}
