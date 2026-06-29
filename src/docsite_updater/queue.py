from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .models import DocUpdateJob, Logger, Metrics


class TransientJobError(Exception):
    """Retryable job failure."""


class PermanentJobError(Exception):
    """Non-retryable job failure."""


@dataclass
class QueuedJob:
    job: DocUpdateJob
    attempts: int = 0
    status: str = "queued"
    errors: list[str] = field(default_factory=list)


class InMemoryJobQueue:
    def __init__(self, *, max_retries: int = 3, logger: Logger | None = None, metrics: Metrics | None = None) -> None:
        self.max_retries = max_retries
        self.logger = logger
        self.metrics = metrics
        self._jobs: dict[str, QueuedJob] = {}
        self.dead_letters: dict[str, QueuedJob] = {}

    @property
    def jobs(self) -> list[QueuedJob]:
        return list(self._jobs.values())

    def enqueue(self, job: DocUpdateJob) -> bool:
        if job.identity in self._jobs or job.identity in self.dead_letters:
            if self.metrics:
                self.metrics.increment("queue.duplicate")
            return False
        self._jobs[job.identity] = QueuedJob(job=job)
        if self.metrics:
            self.metrics.increment("queue.enqueued")
            self.metrics.observe("queue.depth", len(self._jobs))
        return True

    def process_all(self, handler: Callable[[DocUpdateJob], object]) -> None:
        for identity in list(self._jobs.keys()):
            queued = self._jobs[identity]
            try:
                queued.attempts += 1
                queued.status = "running"
                handler(queued.job)
                queued.status = "completed"
                if self.metrics:
                    self.metrics.increment("job.success")
                del self._jobs[identity]
            except TransientJobError as exc:
                queued.errors.append(str(exc))
                if queued.attempts >= self.max_retries:
                    self._dead_letter(identity, queued, "retry_exhausted")
                else:
                    queued.status = "queued"
                    if self.metrics:
                        self.metrics.increment("job.retry")
                    self._backoff_seconds(queued.attempts)
            except PermanentJobError as exc:
                queued.errors.append(str(exc))
                self._dead_letter(identity, queued, "permanent_failure")

    def _dead_letter(self, identity: str, queued: QueuedJob, reason: str) -> None:
        queued.status = "dead_letter"
        self.dead_letters[identity] = queued
        del self._jobs[identity]
        if self.metrics:
            self.metrics.increment("job.dead_letter")
        if self.logger:
            self.logger.error("job_dead_lettered", job_id=identity, reason=reason)

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        jitter = random.uniform(0, 0.1)
        return min(30.0, (2**attempt) + jitter)
