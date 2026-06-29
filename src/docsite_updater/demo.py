from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from .ai import RuleBasedAIClient
from .config import Settings
from .github import InMemoryGitHubClient, SourcePullRequestFixture
from .models import ChangedFile
from .observability import MemoryLogger, MemoryMetrics
from .queue import InMemoryJobQueue
from .security import sign_body
from .validator import MockMkDocsValidator
from .webhook import create_app
from .worker import DocUpdateWorker


def demo_payload(*, merge_commit_sha: str = "abc123def4567890") -> dict[str, Any]:
    return {
        "action": "closed",
        "installation": {"id": 42},
        "repository": {"full_name": "bnb-chain/mock-bsc-app", "default_branch": "main"},
        "pull_request": {
            "number": 7,
            "merged": True,
            "merge_commit_sha": merge_commit_sha,
            "html_url": "https://github.com/bnb-chain/mock-bsc-app/pull/7",
            "base": {"ref": "main"},
        },
    }


def run_demo(*, diff: str = "+ API endpoint get_validator now returns validator status") -> dict[str, Any]:
    settings = Settings(
        github_app_id="123",
        github_private_key="private",
        github_webhook_secret="demo-webhook-secret",
        docsite_repo="bnb-chain/mock-bnb-docsite",
        ai_api_key="demo-ai-key",
        reviewer_logins=("docs-reviewer",),
    )
    logger = MemoryLogger()
    metrics = MemoryMetrics()
    queue = InMemoryJobQueue(max_retries=3, logger=logger, metrics=metrics)
    app = create_app(settings=settings, queue=queue, logger=logger, metrics=metrics)
    client = TestClient(app)

    payload = demo_payload()
    body = json.dumps(payload)
    webhook_response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "x-github-event": "pull_request",
            "x-hub-signature-256": sign_body(body.encode("utf-8"), settings.github_webhook_secret),
            "content-type": "application/json",
        },
    )

    job = queue.jobs[0].job
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
        doc_files={"docs/api.md": "# API\n\nValidator API reference."},
    )
    worker = DocUpdateWorker(
        settings=settings,
        github=github,
        ai=RuleBasedAIClient(),
        validator=MockMkDocsValidator(),
        logger=logger,
        metrics=metrics,
    )

    queue.process_all(worker.run)

    return {
        "webhook_status": webhook_response.json()["status"],
        "jobs_remaining": len(queue.jobs),
        "dead_letters": len(queue.dead_letters),
        "pull_requests": github.pull_requests,
        "doc_files": github.doc_files,
        "metrics": metrics,
        "logs": logger.records,
    }
