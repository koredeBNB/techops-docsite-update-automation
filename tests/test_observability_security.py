from __future__ import annotations

import json

from docsite_updater.config import Settings
from docsite_updater.github import InMemoryGitHubClient, UnauthorizedRepositoryError
from tests.conftest import signed_headers
from tests.test_worker import make_worker_context


def test_worker_structured_logs_include_required_fields() -> None:
    worker, job, _, logger, _ = make_worker_context(diff="+ API endpoint changed")

    worker.run(job)

    completed = [record for record in logger.records if record["message"] == "worker_completed"][-1]
    assert completed["request_id"] == "request-1"
    assert completed["job_id"] == job.identity
    assert completed["source_repo"] == "bnb-chain/mock-bsc-app"
    assert completed["pr_number"] == 7
    assert completed["merge_commit_sha"] == "abc123def4567890"
    assert completed["operation"] == "doc_update_worker"
    assert completed["status"] == "created_pr"
    assert "duration" in completed


def test_logs_do_not_contain_secrets(settings: Settings, logger) -> None:
    logger.info(
        "secret_test",
        github_token="ghs_secret",
        webhook_secret=settings.github_webhook_secret,
        ai_api_key=settings.ai_api_key,
        safe_value="visible",
    )

    logs = logger.as_json_lines()
    assert "ghs_secret" not in logs
    assert settings.github_webhook_secret not in logs
    assert settings.ai_api_key not in logs
    assert "visible" in logs


def test_metrics_are_emitted_for_webhook_and_worker(test_client, settings: Settings, metrics, merged_pr_payload: dict) -> None:
    body = json.dumps(merged_pr_payload)

    response = test_client.post(
        "/webhooks/github",
        content=body,
        headers=signed_headers(merged_pr_payload, settings.github_webhook_secret),
    )
    worker, job, _, _, worker_metrics = make_worker_context(diff="+ API endpoint changed")
    worker.run(job)

    assert response.status_code == 200
    assert metrics.counters["webhook.received"] == 1
    assert metrics.counters["webhook.enqueued"] == 1
    assert worker_metrics.counters["docs_pr.created"] == 1
    assert worker_metrics.observations["github.latency_ms"]
    assert worker_metrics.observations["ai.latency_ms"]


def test_external_timeout_is_configured(settings: Settings) -> None:
    assert settings.external_timeout_seconds > 0
    assert settings.external_timeout_seconds <= 30


def test_authorization_prevents_writing_to_non_docsite_repo() -> None:
    github = InMemoryGitHubClient(docsite_repo="bnb-chain/mock-bnb-docsite", allowed_write_repo="bnb-chain/mock-bsc-app")

    try:
        github.create_docsite_branch("ai-docs/should-not-write")
    except UnauthorizedRepositoryError as exc:
        assert "docsite" in str(exc)
    else:
        raise AssertionError("Expected UnauthorizedRepositoryError")
