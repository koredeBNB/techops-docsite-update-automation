from __future__ import annotations

import re
import time

from .ai import InvalidAIOutputError
from .config import Settings
from .models import AIClient, DocFile, DocUpdate, DocUpdateJob, DocsiteValidator, GitHubClient, Logger, Metrics, WorkerResult
from .queue import PermanentJobError


class DocUpdateWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        github: GitHubClient,
        ai: AIClient,
        validator: DocsiteValidator,
        logger: Logger,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.github = github
        self.ai = ai
        self.validator = validator
        self.logger = logger
        self.metrics = metrics

    def run(self, job: DocUpdateJob) -> WorkerResult:
        started = time.monotonic()
        base_log_fields = {
            "request_id": job.correlation_id,
            "job_id": job.identity,
            "source_repo": job.source_repo,
            "pr_number": job.pr_number,
            "merge_commit_sha": job.merge_commit_sha,
        }
        self.logger.info("worker_started", operation="doc_update_worker", status="started", **base_log_fields)
        try:
            self.github.create_installation_token(job.installation_id)
            pr_context = self.github.fetch_pr_context(job)
            docs = self.github.list_doc_files()
            self.metrics.observe("github.latency_ms", 1.0)

            ai_started = time.monotonic()
            ai_result = self.ai.propose_doc_updates(pr_context, docs)
            self.metrics.observe("ai.latency_ms", (time.monotonic() - ai_started) * 1000)
            self.logger.info(
                "ai_result",
                operation="ai_doc_update",
                status=ai_result.status,
                model=ai_result.usage.model,
                prompt_version=ai_result.usage.prompt_version,
                token_usage={
                    "input": ai_result.usage.input_tokens,
                    "output": ai_result.usage.output_tokens,
                    "cost": ai_result.usage.estimated_cost_usd,
                },
                **base_log_fields,
            )
        except InvalidAIOutputError as exc:
            self.metrics.increment("job.failure")
            raise PermanentJobError(str(exc)) from exc

        if ai_result.status == "no_changes_needed" or not ai_result.updates:
            self.metrics.increment("job.no_changes_needed")
            self.logger.info("worker_completed", operation="doc_update_worker", status="no_changes_needed", **base_log_fields)
            return WorkerResult(status="no_changes_needed", message=ai_result.summary)

        if ai_result.status == "low_confidence":
            self.metrics.increment("job.low_confidence")
            self.logger.warning("worker_low_confidence", operation="doc_update_worker", status="low_confidence", **base_log_fields)
            return WorkerResult(status="low_confidence", message=ai_result.summary)

        safety_result = validate_doc_update_safety(original_docs=docs, updates=ai_result.updates, source_diff=pr_context.diff)
        if not safety_result.ok:
            self.metrics.increment("job.safety_rejected")
            self.logger.error(
                "ai_update_safety_rejected",
                operation="ai_doc_safety_check",
                status="safety_rejected",
                safety_reason=safety_result.reason,
                **base_log_fields,
            )
            return WorkerResult(status="safety_rejected", message=safety_result.reason)

        validation = self.validator.validate(ai_result.updates)
        if not validation.ok:
            self.metrics.increment("job.validation_failed")
            self.logger.error(
                "mkdocs_validation_failed",
                operation="mkdocs_build",
                status="validation_failed",
                validation_output=validation.output,
                **base_log_fields,
            )
            return WorkerResult(status="validation_failed", message=validation.output)

        branch_name = f"ai-docs/{job.source_repo.replace('/', '-')}-{job.pr_number}-{job.merge_commit_sha[:8]}"
        changed_files = [update.path for update in ai_result.updates]
        pr_body = build_pr_body(job=job, summary=ai_result.summary, changed_files=changed_files)

        self.github.create_docsite_branch(branch_name)
        committed_files = self.github.commit_doc_updates(branch_name, ai_result.updates)
        pr = self.github.open_docsite_pr(
            branch_name,
            title=f"[AI Docs] Update docs for {job.source_repo}#{job.pr_number}",
            body=pr_body,
            changed_files=committed_files,
        )
        if self.settings.reviewer_logins:
            self.github.request_reviewers(pr.number, list(self.settings.reviewer_logins))

        self.metrics.increment("docs_pr.created")
        self.logger.info(
            "worker_completed",
            operation="doc_update_worker",
            status="created_pr",
            duration=(time.monotonic() - started),
            docsite_pr=pr.url,
            **base_log_fields,
        )
        return WorkerResult(status="created_pr", pr=pr, message=ai_result.summary)


def build_pr_body(*, job: DocUpdateJob, summary: str, changed_files: list[str]) -> str:
    files = "\n".join(f"- `{path}`" for path in changed_files) or "- No files changed"
    return f"""## Summary
{summary}

## Source
- Source repo: `{job.source_repo}`
- Source PR: {job.source_pr_url}
- Merge commit: `{job.merge_commit_sha}`

## Changed Docs
{files}

## Reviewer Checklist
- [ ] Confirm the generated docs are technically accurate.
- [ ] Confirm examples and references match the source change.
- [ ] Confirm this PR should be merged before publishing.
"""


class SafetyResult:
    def __init__(self, ok: bool, reason: str = "") -> None:
        self.ok = ok
        self.reason = reason


NUMERIC_LITERAL_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def validate_doc_update_safety(*, original_docs: list[DocFile], updates: list[DocUpdate], source_diff: str) -> SafetyResult:
    docs_by_path = {doc.path: doc for doc in original_docs}
    for update in updates:
        original = docs_by_path.get(update.path)
        if original is None:
            continue
        result = _validate_numeric_facts_unchanged(original=original.content, updated=update.content, source_diff=source_diff, path=update.path)
        if not result.ok:
            return result
    return SafetyResult(ok=True)


def _validate_numeric_facts_unchanged(*, original: str, updated: str, source_diff: str, path: str) -> SafetyResult:
    original_numbers = set(NUMERIC_LITERAL_PATTERN.findall(original))
    updated_numbers = set(NUMERIC_LITERAL_PATTERN.findall(updated))
    diff_numbers = set(NUMERIC_LITERAL_PATTERN.findall(source_diff))

    for number in sorted(original_numbers):
        if number in updated_numbers:
            continue
        if number in diff_numbers:
            continue
        negated = number[1:] if number.startswith("-") else f"-{number}"
        if negated in updated_numbers and negated not in diff_numbers:
            return SafetyResult(
                ok=False,
                reason=f"Rejected AI update for {path}: existing numeric value `{number}` changed to `{negated}` without source diff support.",
            )
        return SafetyResult(
            ok=False,
            reason=f"Rejected AI update for {path}: existing numeric value `{number}` was removed or changed without source diff support.",
        )
    return SafetyResult(ok=True)
