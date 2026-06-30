from __future__ import annotations

import re
import time

from .ai import InvalidAIOutputError
from .config import Settings
from .models import (
    AIClient,
    CreatedPullRequest,
    DocFile,
    DocUpdate,
    DocUpdateJob,
    DocsiteValidator,
    GitHubClient,
    Logger,
    Metrics,
    RepoRole,
    WorkerResult,
)
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
            playground_files = self.github.list_playground_files()
            candidate_files = [*docs, *playground_files]
            self.metrics.observe("github.latency_ms", 1.0)

            ai_started = time.monotonic()
            ai_result = self.ai.propose_doc_updates(pr_context, candidate_files)
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

        safety_result = validate_doc_update_safety(
            original_docs=candidate_files,
            updates=ai_result.updates,
            source_diff=pr_context.diff,
        )
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

        updates_by_role = group_updates_by_role(ai_result.updates)
        validation = self.validator.validate(updates_by_role["docsite"])
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

        playground_validation = self.validator.validate_playground(updates_by_role["playground"])
        if not playground_validation.ok:
            self.metrics.increment("job.validation_failed")
            self.logger.error(
                "playground_validation_failed",
                operation="playground_build",
                status="validation_failed",
                validation_output=playground_validation.output,
                **base_log_fields,
            )
            return WorkerResult(status="validation_failed", message=playground_validation.output)

        created_prs: list[CreatedPullRequest] = []
        for repo_role in ("docsite", "playground"):
            role_updates = updates_by_role[repo_role]
            if not role_updates:
                continue
            pr = self._open_update_pr(
                repo_role=repo_role,
                job=job,
                summary=ai_result.summary,
                updates=role_updates,
            )
            if pr:
                created_prs.append(pr)

        if not created_prs:
            self.metrics.increment("job.no_changes_needed")
            self.logger.info("worker_completed", operation="doc_update_worker", status="no_changes_needed", **base_log_fields)
            return WorkerResult(status="no_changes_needed", message="AI proposed no material file changes.")

        self.metrics.increment("docs_pr.created", len([pr for pr in created_prs if pr.repo_role == "docsite"]))
        self.metrics.increment("playground_pr.created", len([pr for pr in created_prs if pr.repo_role == "playground"]))
        self.logger.info(
            "worker_completed",
            operation="doc_update_worker",
            status="created_pr",
            duration=(time.monotonic() - started),
            pull_requests=[pr.url for pr in created_prs],
            **base_log_fields,
        )
        return WorkerResult(status="created_pr", pr=created_prs[0], prs=created_prs, message=ai_result.summary)

    def _open_update_pr(
        self,
        *,
        repo_role: RepoRole,
        job: DocUpdateJob,
        summary: str,
        updates: list[DocUpdate],
    ) -> CreatedPullRequest | None:
        branch_prefix = "ai-docs" if repo_role == "docsite" else "ai-playground"
        branch_name = f"{branch_prefix}/{job.source_repo.replace('/', '-')}-{job.pr_number}-{job.merge_commit_sha[:8]}"
        changed_files = [update.path for update in updates]
        pr_body = build_pr_body(job=job, summary=summary, changed_files=changed_files, repo_role=repo_role)

        self.github.create_repository_branch(repo_role, branch_name)
        committed_files = self.github.commit_repository_updates(repo_role, branch_name, updates)
        if not committed_files:
            return None
        title_subject = "docs" if repo_role == "docsite" else "playground"
        pr = self.github.open_repository_pr(
            repo_role,
            branch_name,
            title=f"[AI {title_subject.title()}] Update {title_subject} for {job.source_repo}#{job.pr_number}",
            body=pr_body,
            changed_files=committed_files,
        )
        if self.settings.reviewer_logins:
            self.github.request_repository_reviewers(repo_role, pr.number, list(self.settings.reviewer_logins))
        return pr


def build_pr_body(*, job: DocUpdateJob, summary: str, changed_files: list[str], repo_role: RepoRole = "docsite") -> str:
    files = "\n".join(f"- `{path}`" for path in changed_files) or "- No files changed"
    target = "MkDocs documentation" if repo_role == "docsite" else "interactive playground"
    return f"""## Summary
{summary}

## Target
- {target}

## Source
- Source repo: `{job.source_repo}`
- Source PR: {job.source_pr_url}
- Merge commit: `{job.merge_commit_sha}`

## Changed Files
{files}

## Reviewer Checklist
- [ ] Confirm the generated updates are technically accurate.
- [ ] Confirm examples and references match the source change.
- [ ] Confirm this PR should be merged before publishing.
"""


def group_updates_by_role(updates: list[DocUpdate]) -> dict[RepoRole, list[DocUpdate]]:
    grouped: dict[RepoRole, list[DocUpdate]] = {"docsite": [], "playground": []}
    for update in updates:
        grouped[update.repo_role].append(update)
    return grouped


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
