from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt

from .models import ChangedFile, CreatedPullRequest, DocFile, DocUpdate, DocUpdateJob, PullRequestContext


class UnauthorizedRepositoryError(Exception):
    pass


class GitHubClientError(Exception):
    pass


@dataclass
class SourcePullRequestFixture:
    repo: str
    number: int
    merge_commit_sha: str
    url: str
    diff: str
    changed_files: list[ChangedFile]


@dataclass
class InMemoryGitHubClient:
    docsite_repo: str
    source_prs: dict[str, SourcePullRequestFixture] = field(default_factory=dict)
    doc_files: dict[str, str] = field(default_factory=dict)
    branches: set[str] = field(default_factory=set)
    pull_requests: list[CreatedPullRequest] = field(default_factory=list)
    requested_reviewers: dict[int, list[str]] = field(default_factory=dict)
    installation_tokens: list[int] = field(default_factory=list)
    allowed_write_repo: str | None = None

    def __post_init__(self) -> None:
        if self.allowed_write_repo is None:
            self.allowed_write_repo = self.docsite_repo

    def create_installation_token(self, installation_id: int) -> str:
        self.installation_tokens.append(installation_id)
        return f"installation-token-{installation_id}"

    def fetch_pr_context(self, job: DocUpdateJob) -> PullRequestContext:
        fixture = self.source_prs[job.identity]
        return PullRequestContext(
            source_repo=fixture.repo,
            pr_number=fixture.number,
            merge_commit_sha=fixture.merge_commit_sha,
            source_pr_url=fixture.url,
            diff=fixture.diff,
            changed_files=fixture.changed_files,
        )

    def list_doc_files(self) -> list[DocFile]:
        return [DocFile(path=path, content=content) for path, content in sorted(self.doc_files.items())]

    def create_docsite_branch(self, branch_name: str) -> None:
        self._assert_can_write_docsite()
        self.branches.add(branch_name)

    def commit_doc_updates(self, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        self._assert_can_write_docsite()
        if branch_name not in self.branches:
            raise ValueError(f"Branch does not exist: {branch_name}")
        changed: list[str] = []
        for update in updates:
            previous = self.doc_files.get(update.path)
            if previous != update.content:
                self.doc_files[update.path] = update.content
                changed.append(update.path)
        return changed

    def open_docsite_pr(self, branch_name: str, title: str, body: str, changed_files: list[str]) -> CreatedPullRequest:
        self._assert_can_write_docsite()
        pr = CreatedPullRequest(
            url=f"https://github.com/{self.docsite_repo}/pull/{len(self.pull_requests) + 1}",
            number=len(self.pull_requests) + 1,
            branch=branch_name,
            changed_files=changed_files,
        )
        self.pull_requests.append(pr)
        return pr

    def request_reviewers(self, pr_number: int, reviewers: list[str]) -> None:
        self.requested_reviewers[pr_number] = reviewers

    def _assert_can_write_docsite(self) -> None:
        if self.allowed_write_repo != self.docsite_repo:
            raise UnauthorizedRepositoryError("Writes are restricted to the configured docsite repository")


@dataclass
class GitHubAppClient:
    app_id: str
    private_key: str
    docsite_repo: str
    api_base_url: str = "https://api.github.com"
    timeout_seconds: float = 10.0
    default_branch: str = "main"
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._installation_token: str | None = None
        self._docsite_default_branch: str | None = None

    def create_installation_token(self, installation_id: int) -> str:
        response = self._client().post(
            self._url(f"/app/installations/{installation_id}/access_tokens"),
            headers=self._app_headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        token = response.json().get("token")
        if not token:
            raise GitHubClientError("GitHub installation token response did not include token")
        self._installation_token = str(token)
        return self._installation_token

    def fetch_pr_context(self, job: DocUpdateJob) -> PullRequestContext:
        owner, repo = _split_repo(job.source_repo)
        pr = self._get_json(f"/repos/{owner}/{repo}/pulls/{job.pr_number}")
        files_payload = self._get_json(f"/repos/{owner}/{repo}/pulls/{job.pr_number}/files")
        diff_response = self._client().get(
            self._url(f"/repos/{owner}/{repo}/pulls/{job.pr_number}"),
            headers={**self._installation_headers(), "Accept": "application/vnd.github.v3.diff"},
            timeout=self.timeout_seconds,
        )
        diff_response.raise_for_status()
        changed_files = [
            ChangedFile(path=str(item.get("filename")), status=str(item.get("status")))
            for item in files_payload
            if item.get("filename")
        ]
        return PullRequestContext(
            source_repo=job.source_repo,
            pr_number=job.pr_number,
            merge_commit_sha=str(pr.get("merge_commit_sha") or job.merge_commit_sha),
            source_pr_url=str(pr.get("html_url") or job.source_pr_url),
            diff=diff_response.text,
            changed_files=changed_files,
        )

    def list_doc_files(self) -> list[DocFile]:
        owner, repo = _split_repo(self.docsite_repo)
        branch = self._docsite_branch()
        tree = self._get_json(f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1").get("tree", [])
        docs: list[DocFile] = []
        for item in tree:
            path = item.get("path")
            if item.get("type") != "blob" or not isinstance(path, str):
                continue
            if not _is_doc_file(path):
                continue
            docs.append(DocFile(path=path, content=self._get_file_content(owner=owner, repo=repo, path=path, ref=branch)[0]))
        return docs

    def create_docsite_branch(self, branch_name: str) -> None:
        owner, repo = _split_repo(self.docsite_repo)
        branch = self._docsite_branch()
        ref = self._get_json(f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        sha = ((ref.get("object") or {}).get("sha"))
        if not sha:
            raise GitHubClientError("Could not resolve docsite default branch SHA")
        response = self._client().post(
            self._url(f"/repos/{owner}/{repo}/git/refs"),
            headers=self._installation_headers(),
            json={"ref": f"refs/heads/{branch_name}", "sha": sha},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 422:
            return
        response.raise_for_status()

    def commit_doc_updates(self, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        owner, repo = _split_repo(self.docsite_repo)
        changed_files: list[str] = []
        for update in updates:
            current_content, sha = self._get_file_content(owner=owner, repo=repo, path=update.path, ref=branch_name)
            if current_content == update.content:
                continue
            payload = {
                "message": f"Update {update.path} from AI docsite updater",
                "content": base64.b64encode(update.content.encode("utf-8")).decode("ascii"),
                "branch": branch_name,
                "sha": sha,
            }
            response = self._client().put(
                self._url(f"/repos/{owner}/{repo}/contents/{update.path}"),
                headers=self._installation_headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            changed_files.append(update.path)
        return changed_files

    def open_docsite_pr(self, branch_name: str, title: str, body: str, changed_files: list[str]) -> CreatedPullRequest:
        owner, repo = _split_repo(self.docsite_repo)
        response = self._client().post(
            self._url(f"/repos/{owner}/{repo}/pulls"),
            headers=self._installation_headers(),
            json={"title": title, "body": body, "head": branch_name, "base": self._docsite_branch()},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return CreatedPullRequest(
            url=str(payload.get("html_url")),
            number=int(payload.get("number")),
            branch=branch_name,
            changed_files=changed_files,
        )

    def request_reviewers(self, pr_number: int, reviewers: list[str]) -> None:
        if not reviewers:
            return
        owner, repo = _split_repo(self.docsite_repo)
        response = self._client().post(
            self._url(f"/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers"),
            headers=self._installation_headers(),
            json={"reviewers": reviewers},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _docsite_branch(self) -> str:
        if self._docsite_default_branch:
            return self._docsite_default_branch
        owner, repo = _split_repo(self.docsite_repo)
        repo_payload = self._get_json(f"/repos/{owner}/{repo}")
        self._docsite_default_branch = str(repo_payload.get("default_branch") or self.default_branch)
        return self._docsite_default_branch

    def _get_file_content(self, *, owner: str, repo: str, path: str, ref: str) -> tuple[str, str]:
        payload = self._get_json(f"/repos/{owner}/{repo}/contents/{path}?ref={ref}")
        encoded = payload.get("content")
        sha = payload.get("sha")
        if not isinstance(encoded, str) or not isinstance(sha, str):
            raise GitHubClientError(f"GitHub content response for {path} was missing content or sha")
        content = base64.b64decode(encoded.encode("ascii"), validate=False).decode("utf-8")
        return content, sha

    def _get_json(self, path: str) -> Any:
        response = self._client().get(self._url(path), headers=self._installation_headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _installation_headers(self) -> dict[str, str]:
        if not self._installation_token:
            raise GitHubClientError("Installation token has not been created")
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._installation_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _app_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._app_jwt()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _app_jwt(self) -> str:
        now = int(time.time())
        private_key = self.private_key.replace("\\n", "\n")
        return jwt.encode({"iat": now - 60, "exp": now + 540, "iss": self.app_id}, private_key, algorithm="RS256")

    def _client(self) -> httpx.Client:
        if self.client is not None:
            return self.client
        return httpx.Client(timeout=self.timeout_seconds, trust_env=False)

    def _url(self, path: str) -> str:
        return f"{self.api_base_url.rstrip('/')}/{path.lstrip('/')}"


def create_github_client(settings) -> InMemoryGitHubClient | GitHubAppClient:
    if settings.github_client.lower() == "app":
        return GitHubAppClient(
            app_id=settings.github_app_id,
            private_key=settings.github_private_key,
            docsite_repo=settings.docsite_repo,
            api_base_url=settings.github_api_base_url,
            timeout_seconds=settings.external_timeout_seconds,
        )
    return InMemoryGitHubClient(docsite_repo=settings.docsite_repo)


def _split_repo(full_name: str) -> tuple[str, str]:
    parts = full_name.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise GitHubClientError(f"Invalid repository full name: {full_name}")
    return parts[0], parts[1]


def _is_doc_file(path: str) -> bool:
    return path.startswith("docs/") and path.lower().endswith((".md", ".mdx", ".rst", ".txt", ".adoc"))
