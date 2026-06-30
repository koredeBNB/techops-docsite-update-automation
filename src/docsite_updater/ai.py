from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .models import AIUpdateResult, AIUsage, DocFile, DocUpdate, PullRequestContext, RepoRole


class InvalidAIOutputError(Exception):
    pass


@dataclass
class RuleBasedAIClient:
    model: str = "mock-rule-based"
    prompt_version: str = "prototype-v1"
    low_confidence: bool = False
    invalid_output: bool = False

    def propose_doc_updates(self, pr_context: PullRequestContext, docs: list[DocFile]) -> AIUpdateResult:
        if self.invalid_output:
            raise InvalidAIOutputError("AI output could not be parsed")

        usage = AIUsage(
            model=self.model,
            prompt_version=self.prompt_version,
            input_tokens=len(pr_context.diff.split()) + sum(len(doc.content.split()) for doc in docs),
            output_tokens=64,
            estimated_cost_usd=0.001,
        )

        if "NO_DOC_CHANGE" in pr_context.diff:
            return AIUpdateResult(status="no_changes_needed", summary="No documentation update needed.", usage=usage)

        updates: list[DocUpdate] = []
        lower_diff = pr_context.diff.lower()
        for doc in docs:
            if "api" in lower_diff and ("api" in doc.path.lower() or "api" in doc.content.lower()):
                section = "Automated Playground Update" if doc.repo_role == "playground" else "Automated Update"
                updates.append(
                    DocUpdate(
                        path=doc.path,
                        content=f"{doc.content.rstrip()}\n\n## {section}\n\nUpdated based on {pr_context.source_repo} PR #{pr_context.pr_number}.\n",
                        rationale="API-related source changes affect this documentation page.",
                        confidence=0.4 if self.low_confidence else 0.92,
                        repo_role=doc.repo_role,
                    )
                )

        if not updates:
            return AIUpdateResult(status="no_changes_needed", summary="No directly relevant docs were found.", usage=usage)

        if self.low_confidence:
            return AIUpdateResult(status="low_confidence", summary="Possible documentation update found with low confidence.", updates=updates, usage=usage)

        return AIUpdateResult(status="updates", summary="Documentation updates are needed for API changes.", updates=updates, usage=usage)


@dataclass
class OpenRouterAIClient:
    api_key: str
    model: str = "deepseek/deepseek-v3.2"
    base_url: str = "https://openrouter.ai/api/v1"
    prompt_version: str = "openrouter-doc-update-v1"
    timeout_seconds: float = 10.0
    http_referer: str = "https://github.com/koredeBNB/automated-docsite-prototype"
    app_title: str = "AI Docsite Update Prototype"
    client: httpx.Client | None = None

    def propose_doc_updates(self, pr_context: PullRequestContext, docs: list[DocFile]) -> AIUpdateResult:
        if not self.api_key:
            raise InvalidAIOutputError("OPENROUTER_API_KEY is required when AI_PROVIDER=openrouter")

        response = self._client().post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_title,
            },
            json={
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(pr_context, docs)},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = _extract_message_content(payload)
        parsed = _parse_ai_json(content)
        return _result_from_openrouter_payload(parsed, payload, self.model, self.prompt_version)

    def _client(self) -> httpx.Client:
        if self.client is not None:
            return self.client
        return httpx.Client(timeout=self.timeout_seconds, trust_env=False)


def create_ai_client(settings: Settings):
    if settings.ai_provider.lower() == "openrouter":
        return OpenRouterAIClient(
            api_key=settings.openrouter_api_key or settings.ai_api_key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.external_timeout_seconds,
            http_referer=settings.openrouter_http_referer,
            app_title=settings.openrouter_app_title,
        )
    return RuleBasedAIClient()


def system_prompt() -> str:
    return """You update documentation and interactive playground files from source code diffs.

Rules:
- Be conservative. Only update files directly affected by the source diff.
- Preserve the existing style of each target repository.
- Return no_changes_needed when no docs or playground files need changes.
- Return low_confidence if the update is plausible but uncertain.
- Never invent APIs, fields, behavior, or examples not supported by the diff.
- Do not modify existing documented values, examples, numbers, defaults, or claims unless the source diff explicitly changes them.
- When adding a new response field, add only that new field and keep all existing example values exactly unchanged.
- Use repo_role "docsite" for MkDocs documentation updates.
- Use repo_role "playground" for interactive playground updates.
- Playground updates may touch any relevant existing text/code file from the provided playground repository files, but must be small, source-diff-supported, and directly related.
- Do not update workflows, secrets, generated build output, binary assets, lockfiles, or dependency files unless the source diff explicitly requires it.
- Output only valid JSON with this schema:
{
  "status": "updates | no_changes_needed | low_confidence",
  "summary": "short explanation",
  "updates": [
    {
      "repo_role": "docsite | playground",
      "path": "docs/example.md",
      "content": "full updated markdown content",
      "rationale": "why this file changed",
      "confidence": 0.92
    }
  ]
}
"""


def user_prompt(pr_context: PullRequestContext, docs: list[DocFile]) -> str:
    changed_files = "\n".join(f"- {item.path} ({item.status})" for item in pr_context.changed_files)
    files_payload = "\n\n".join(f"--- FILE: {doc.repo_role}:{doc.path} ---\n{doc.content}" for doc in docs)
    return f"""Source repository: {pr_context.source_repo}
Source PR: #{pr_context.pr_number}
Source PR URL: {pr_context.source_pr_url}
Merge commit: {pr_context.merge_commit_sha}

Changed files:
{changed_files}

Source diff:
```diff
{pr_context.diff}
```

Documentation files:
{files_payload}
"""


def _extract_message_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidAIOutputError("OpenRouter response did not include message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise InvalidAIOutputError("OpenRouter response content was empty")
    return content


def _parse_ai_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidAIOutputError("AI output was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise InvalidAIOutputError("AI output must be a JSON object")
    return parsed


def _result_from_openrouter_payload(parsed: dict[str, Any], raw_payload: dict[str, Any], model: str, prompt_version: str) -> AIUpdateResult:
    status = parsed.get("status")
    if status not in {"updates", "no_changes_needed", "low_confidence"}:
        raise InvalidAIOutputError("AI output status is invalid")

    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise InvalidAIOutputError("AI output summary is required")

    raw_updates = parsed.get("updates", [])
    if raw_updates is None:
        raw_updates = []
    if not isinstance(raw_updates, list):
        raise InvalidAIOutputError("AI output updates must be a list")

    updates = [_doc_update_from_payload(item) for item in raw_updates]
    if status == "updates" and not updates:
        raise InvalidAIOutputError("AI output with updates status must include at least one update")

    usage_payload = raw_payload.get("usage") or {}
    usage = AIUsage(
        model=model,
        prompt_version=prompt_version,
        input_tokens=int(usage_payload.get("prompt_tokens") or 0),
        output_tokens=int(usage_payload.get("completion_tokens") or 0),
        estimated_cost_usd=float(usage_payload.get("cost") or 0.0),
    )
    return AIUpdateResult(status=status, summary=summary, updates=updates, usage=usage)


def _doc_update_from_payload(item: Any) -> DocUpdate:
    if not isinstance(item, dict):
        raise InvalidAIOutputError("Each update must be an object")
    path = item.get("path")
    content = item.get("content")
    rationale = item.get("rationale")
    confidence = item.get("confidence", 0.0)
    repo_role = item.get("repo_role", "docsite")
    if repo_role not in {"docsite", "playground"}:
        raise InvalidAIOutputError("Each update repo_role must be docsite or playground")
    typed_repo_role = repo_role  # helps older type checkers narrow below
    if not isinstance(path, str):
        raise InvalidAIOutputError("Each update path must be a string")
    if typed_repo_role == "docsite" and not path.startswith("docs/"):
        raise InvalidAIOutputError("Docsite update paths must be docs/ paths")
    if typed_repo_role == "playground" and not _is_safe_playground_update_path(path):
        raise InvalidAIOutputError("Playground update path is not safe to edit")
    if not isinstance(content, str) or not content.strip():
        raise InvalidAIOutputError("Each update content must be non-empty")
    if not isinstance(rationale, str) or not rationale.strip():
        raise InvalidAIOutputError("Each update rationale must be non-empty")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise InvalidAIOutputError("Each update confidence must be numeric") from exc
    if confidence_value < 0 or confidence_value > 1:
        raise InvalidAIOutputError("Each update confidence must be between 0 and 1")
    return DocUpdate(
        path=path,
        content=content,
        rationale=rationale,
        confidence=confidence_value,
        repo_role=typed_repo_role,
    )


def _is_safe_playground_update_path(path: str) -> bool:
    lowered = path.lower()
    parts = lowered.split("/")
    if path.startswith("/") or ".." in parts:
        return False
    if any(part in {".git", "node_modules", "dist", "build", "coverage"} for part in parts):
        return False
    if lowered.startswith((".env", ".github/")) or "/.env" in lowered:
        return False
    if lowered.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".lock")):
        return False
    return True
