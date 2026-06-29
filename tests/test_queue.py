from __future__ import annotations

from docsite_updater.models import DocUpdateJob
from docsite_updater.observability import MemoryLogger, MemoryMetrics
from docsite_updater.queue import InMemoryJobQueue, PermanentJobError, TransientJobError


def make_job(sha: str = "abc123") -> DocUpdateJob:
    return DocUpdateJob(
        source_repo="bnb-chain/mock-bsc-app",
        pr_number=7,
        merge_commit_sha=sha,
        default_branch="main",
        installation_id=42,
        source_pr_url="https://github.com/bnb-chain/mock-bsc-app/pull/7",
        correlation_id="request-1",
    )


def test_job_identity_uses_repo_pr_and_merge_commit() -> None:
    assert make_job().identity == "bnb-chain/mock-bsc-app#7@abc123"


def test_duplicate_jobs_are_not_enqueued() -> None:
    queue = InMemoryJobQueue()
    job = make_job()

    assert queue.enqueue(job) is True
    assert queue.enqueue(job) is False
    assert len(queue.jobs) == 1


def test_job_payload_carries_required_trace_fields() -> None:
    job = make_job()

    assert job.source_repo
    assert job.pr_number == 7
    assert job.default_branch == "main"
    assert job.merge_commit_sha == "abc123"
    assert job.installation_id == 42
    assert job.correlation_id == "request-1"


def test_transient_failure_is_retried_then_succeeds() -> None:
    metrics = MemoryMetrics()
    queue = InMemoryJobQueue(max_retries=3, metrics=metrics)
    queue.enqueue(make_job())
    attempts = {"count": 0}

    def handler(_: DocUpdateJob) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TransientJobError("temporary github outage")

    queue.process_all(handler)
    assert len(queue.jobs) == 1
    assert queue.jobs[0].status == "queued"

    queue.process_all(handler)
    assert queue.jobs == []
    assert metrics.counters["job.retry"] == 1
    assert metrics.counters["job.success"] == 1


def test_transient_failure_moves_to_dead_letter_after_retry_limit() -> None:
    logger = MemoryLogger()
    metrics = MemoryMetrics()
    queue = InMemoryJobQueue(max_retries=2, logger=logger, metrics=metrics)
    job = make_job()
    queue.enqueue(job)

    def handler(_: DocUpdateJob) -> None:
        raise TransientJobError("still unavailable")

    queue.process_all(handler)
    queue.process_all(handler)

    assert queue.jobs == []
    assert job.identity in queue.dead_letters
    assert queue.dead_letters[job.identity].status == "dead_letter"
    assert metrics.counters["job.dead_letter"] == 1
    assert "still unavailable" in queue.dead_letters[job.identity].errors[-1]


def test_permanent_failure_moves_to_dead_letter_immediately() -> None:
    queue = InMemoryJobQueue(max_retries=3)
    job = make_job()
    queue.enqueue(job)

    def handler(_: DocUpdateJob) -> None:
        raise PermanentJobError("invalid ai output")

    queue.process_all(handler)

    assert queue.jobs == []
    assert job.identity in queue.dead_letters
    assert queue.dead_letters[job.identity].attempts == 1
