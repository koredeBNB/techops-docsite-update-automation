from __future__ import annotations

from .ai import create_ai_client
from .config import Settings
from .github import create_github_client
from .observability import MemoryLogger, MemoryMetrics
from .queue import InMemoryJobQueue
from .validator import MockMkDocsValidator
from .webhook import create_app
from .worker import DocUpdateWorker

settings = Settings.from_env()
logger = MemoryLogger()
metrics = MemoryMetrics()
queue = InMemoryJobQueue(max_retries=settings.max_retries, logger=logger, metrics=metrics)
github = create_github_client(settings)
ai = create_ai_client(settings)
validator = MockMkDocsValidator()
worker = DocUpdateWorker(settings=settings, github=github, ai=ai, validator=validator, logger=logger, metrics=metrics)

app = create_app(settings=settings, queue=queue, logger=logger, metrics=metrics, job_handler=worker.run)
