from __future__ import annotations

import json

from fastapi.testclient import TestClient

from docsite_updater.config import Settings
from docsite_updater.observability import MemoryLogger, MemoryMetrics
from docsite_updater.queue import InMemoryJobQueue
from docsite_updater.webhook import create_app

from .conftest import clone_payload, signed_headers


def post_payload(client: TestClient, payload: dict, settings: Settings, event: str = "pull_request"):
    body = json.dumps(payload)
    return client.post(
        "/webhooks/github",
        content=body,
        headers=signed_headers(payload, settings.github_webhook_secret, event=event),
    )


def test_rejects_missing_signature(test_client: TestClient, merged_pr_payload: dict) -> None:
    response = test_client.post("/webhooks/github", json=merged_pr_payload, headers={"x-github-event": "pull_request"})

    assert response.status_code == 401


def test_rejects_invalid_signature(test_client: TestClient, merged_pr_payload: dict) -> None:
    response = test_client.post(
        "/webhooks/github",
        json=merged_pr_payload,
        headers={"x-github-event": "pull_request", "x-hub-signature-256": "sha256=bad"},
    )

    assert response.status_code == 401


def test_accepts_valid_merged_default_branch_pr(
    test_client: TestClient,
    settings: Settings,
    queue: InMemoryJobQueue,
    merged_pr_payload: dict,
) -> None:
    response = post_payload(test_client, merged_pr_payload, settings)

    assert response.status_code == 200
    assert response.json()["status"] == "enqueued"
    assert len(queue.jobs) == 1
    assert queue.jobs[0].job.identity == "bnb-chain/mock-bsc-app#7@abc123def4567890"


def test_ignores_non_pull_request_event(test_client: TestClient, settings: Settings, queue: InMemoryJobQueue, merged_pr_payload: dict) -> None:
    response = post_payload(test_client, merged_pr_payload, settings, event="push")

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert queue.jobs == []


def test_ignores_closed_unmerged_pr(test_client: TestClient, settings: Settings, queue: InMemoryJobQueue, merged_pr_payload: dict) -> None:
    payload = clone_payload(merged_pr_payload)
    payload["pull_request"]["merged"] = False

    response = post_payload(test_client, payload, settings)

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert queue.jobs == []


def test_ignores_non_default_branch_pr(test_client: TestClient, settings: Settings, queue: InMemoryJobQueue, merged_pr_payload: dict) -> None:
    payload = clone_payload(merged_pr_payload)
    payload["pull_request"]["base"]["ref"] = "release"

    response = post_payload(test_client, payload, settings)

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert queue.jobs == []


def test_duplicate_delivery_does_not_duplicate_job(
    test_client: TestClient,
    settings: Settings,
    queue: InMemoryJobQueue,
    merged_pr_payload: dict,
) -> None:
    first = post_payload(test_client, merged_pr_payload, settings)
    second = post_payload(test_client, merged_pr_payload, settings)

    assert first.json()["status"] == "enqueued"
    assert second.json()["status"] == "duplicate"
    assert len(queue.jobs) == 1


def test_webhook_does_not_run_ai_inline(
    test_client: TestClient,
    settings: Settings,
    queue: InMemoryJobQueue,
    merged_pr_payload: dict,
) -> None:
    response = post_payload(test_client, merged_pr_payload, settings)

    assert response.status_code == 200
    assert len(queue.jobs) == 1
    assert queue.jobs[0].status == "queued"


def test_webhook_can_schedule_worker_handler_for_real_processing(settings: Settings, merged_pr_payload: dict) -> None:
    logger = MemoryLogger()
    metrics = MemoryMetrics()
    queue = InMemoryJobQueue(max_retries=3, logger=logger, metrics=metrics)
    handled: list[str] = []

    def handler(job) -> None:
        handled.append(job.identity)

    app = create_app(settings=settings, queue=queue, logger=logger, metrics=metrics, job_handler=handler)
    client = TestClient(app)

    response = post_payload(client, merged_pr_payload, settings)

    assert response.status_code == 200
    assert response.json()["status"] == "enqueued"
    assert handled == ["bnb-chain/mock-bsc-app#7@abc123def4567890"]
    assert queue.jobs == []
