from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docsite_updater.config import Settings
from docsite_updater.observability import MemoryLogger, MemoryMetrics
from docsite_updater.queue import InMemoryJobQueue
from docsite_updater.security import sign_body
from docsite_updater.webhook import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        github_app_id="123",
        github_private_key="super-private-key",
        github_webhook_secret="test-webhook-secret",
        docsite_repo="bnb-chain/mock-bnb-docsite",
        ai_api_key="ai-secret-key",
        reviewer_logins=("docs-reviewer",),
    )


@pytest.fixture
def logger() -> MemoryLogger:
    return MemoryLogger()


@pytest.fixture
def metrics() -> MemoryMetrics:
    return MemoryMetrics()


@pytest.fixture
def queue(settings: Settings, logger: MemoryLogger, metrics: MemoryMetrics) -> InMemoryJobQueue:
    return InMemoryJobQueue(max_retries=settings.max_retries, logger=logger, metrics=metrics)


@pytest.fixture
def test_client(settings: Settings, queue: InMemoryJobQueue, logger: MemoryLogger, metrics: MemoryMetrics) -> TestClient:
    app = create_app(settings=settings, queue=queue, logger=logger, metrics=metrics)
    return TestClient(app)


@pytest.fixture
def merged_pr_payload() -> dict:
    payload_path = Path(__file__).parent / "fixtures" / "pull_request_closed_merged.json"
    return json.loads(payload_path.read_text())


def signed_headers(payload: dict, secret: str, event: str = "pull_request") -> dict[str, str]:
    body = json.dumps(payload).encode("utf-8")
    return {
        "x-github-event": event,
        "x-hub-signature-256": sign_body(body, secret),
        "content-type": "application/json",
    }


def clone_payload(payload: dict) -> dict:
    return copy.deepcopy(payload)
