from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from .config import Settings
from .models import DocUpdateJob, Logger, Metrics
from .queue import InMemoryJobQueue
from .security import verify_github_signature


def create_app(
    *,
    settings: Settings,
    queue: InMemoryJobQueue,
    logger: Logger,
    metrics: Metrics,
    job_handler: Optional[Callable[[DocUpdateJob], object]] = None,
) -> FastAPI:
    app = FastAPI(title="AI Docsite Update GitHub App")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhooks/github")
    async def github_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_github_event: Optional[str] = Header(default=None),
        x_hub_signature_256: Optional[str] = Header(default=None),
    ) -> dict[str, str]:
        body = await request.body()
        metrics.increment("webhook.received")
        if not verify_github_signature(body, x_hub_signature_256, settings.github_webhook_secret):
            metrics.increment("webhook.rejected")
            logger.warning("webhook_rejected", operation="github_webhook", status="invalid_signature")
            raise HTTPException(status_code=401, detail="Invalid GitHub signature")

        payload = json.loads(body.decode("utf-8"))
        job = build_job_from_payload(event=x_github_event, payload=payload)
        if not job:
            metrics.increment("webhook.skipped")
            return {"status": "skipped"}

        enqueued = queue.enqueue(job)
        metrics.increment("webhook.enqueued" if enqueued else "webhook.duplicate")
        logger.info(
            "webhook_enqueued" if enqueued else "webhook_duplicate",
            request_id=job.correlation_id,
            job_id=job.identity,
            source_repo=job.source_repo,
            pr_number=job.pr_number,
            merge_commit_sha=job.merge_commit_sha,
            operation="github_webhook",
            status="enqueued" if enqueued else "duplicate",
        )
        if enqueued and job_handler:
            background_tasks.add_task(queue.process_all, job_handler)
        return {"status": "enqueued" if enqueued else "duplicate", "job_id": job.identity}

    return app


def build_job_from_payload(*, event: Optional[str], payload: dict[str, Any]) -> Optional[DocUpdateJob]:
    if event != "pull_request":
        return None
    if payload.get("action") != "closed":
        return None

    pull_request = payload.get("pull_request") or {}
    repository = payload.get("repository") or {}
    installation = payload.get("installation") or {}

    if not pull_request.get("merged"):
        return None

    base_ref = ((pull_request.get("base") or {}).get("ref")) or ""
    default_branch = repository.get("default_branch") or "main"
    if base_ref != default_branch:
        return None

    merge_commit_sha = pull_request.get("merge_commit_sha")
    repo_full_name = repository.get("full_name")
    pr_number = pull_request.get("number")
    installation_id = installation.get("id")
    source_pr_url = pull_request.get("html_url")
    if not all([merge_commit_sha, repo_full_name, pr_number, installation_id, source_pr_url]):
        return None

    return DocUpdateJob(
        source_repo=str(repo_full_name),
        pr_number=int(pr_number),
        merge_commit_sha=str(merge_commit_sha),
        default_branch=str(default_branch),
        installation_id=int(installation_id),
        source_pr_url=str(source_pr_url),
        correlation_id=str(uuid.uuid4()),
    )
