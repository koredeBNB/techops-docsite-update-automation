from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt

from .models import ChangedFile, CreatedPullRequest, DocFile, DocUpdate, DocUpdateJob, PullRequestContext, RepoRole


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
    playground_repo: str = ""
    source_prs: dict[str, SourcePullRequestFixture] = field(default_factory=dict)
    doc_files: dict[str, str] = field(default_factory=dict)
    playground_files: dict[str, str] = field(default_factory=dict)
    branches: set[str] = field(default_factory=set)
    pull_requests: list[CreatedPullRequest] = field(default_factory=list)
    pr_diffs: dict[tuple[RepoRole, int], str] = field(default_factory=dict)
    pr_comments: dict[tuple[RepoRole, int], list[str]] = field(default_factory=dict)
    requested_reviewers: dict[int, list[str]] = field(default_factory=dict)
    installation_tokens: list[int] = field(default_factory=list)
    allowed_write_repo: str | None = None

    def __post_init__(self) -> None:
        # By default the in-memory client can write to the configured target
        # repositories. Tests can set allowed_write_repo to simulate a
        # restricted GitHub App installation.
        pass

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
        return [
            DocFile(path=path, content=content, repo_role="docsite")
            for path, content in sorted(self.doc_files.items())
            if _is_doc_file(path)
        ]

    def list_playground_files(self) -> list[DocFile]:
        if not self.playground_repo:
            return []
        return [
            DocFile(path=path, content=content, repo_role="playground")
            for path, content in sorted(self.playground_files.items())
            if _is_playground_file(path)
        ]

    def create_docsite_branch(self, branch_name: str) -> None:
        self.create_repository_branch("docsite", branch_name)

    def commit_doc_updates(self, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        return self.commit_repository_updates("docsite", branch_name, updates)

    def open_docsite_pr(self, branch_name: str, title: str, body: str, changed_files: list[str]) -> CreatedPullRequest:
        return self.open_repository_pr("docsite", branch_name, title, body, changed_files)

    def request_reviewers(self, pr_number: int, reviewers: list[str]) -> None:
        self.request_repository_reviewers("docsite", pr_number, reviewers)

    def create_repository_branch(self, repo_role: RepoRole, branch_name: str) -> None:
        self._assert_can_write_repo(repo_role)
        self.branches.add(self._branch_key(repo_role, branch_name))

    def commit_repository_updates(self, repo_role: RepoRole, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        self._assert_can_write_repo(repo_role)
        if self._branch_key(repo_role, branch_name) not in self.branches:
            raise ValueError(f"Branch does not exist: {branch_name}")
        files = self._files_for_role(repo_role)
        changed: list[str] = []
        for update in updates:
            previous = files.get(update.path)
            if previous != update.content:
                files[update.path] = update.content
                changed.append(update.path)
        return changed

    def open_repository_pr(
        self,
        repo_role: RepoRole,
        branch_name: str,
        title: str,
        body: str,
        changed_files: list[str],
    ) -> CreatedPullRequest:
        self._assert_can_write_repo(repo_role)
        repo = self._repo_for_role(repo_role)
        pr = CreatedPullRequest(
            url=f"https://github.com/{repo}/pull/{len(self.pull_requests) + 1}",
            number=len(self.pull_requests) + 1,
            branch=branch_name,
            changed_files=changed_files,
            repo_role=repo_role,
            repo=repo,
        )
        self.pull_requests.append(pr)
        self.pr_diffs[(repo_role, pr.number)] = "\n".join(f"diff -- {path}" for path in changed_files)
        return pr

    def request_repository_reviewers(self, repo_role: RepoRole, pr_number: int, reviewers: list[str]) -> None:
        self._assert_can_write_repo(repo_role)
        self.requested_reviewers[pr_number] = reviewers

    def fetch_repository_pr_diff(self, repo_role: RepoRole, pr_number: int) -> str:
        self._assert_can_write_repo(repo_role)
        return self.pr_diffs.get((repo_role, pr_number), "")

    def comment_on_repository_pr(self, repo_role: RepoRole, pr_number: int, body: str) -> None:
        self._assert_can_write_repo(repo_role)
        self.pr_comments.setdefault((repo_role, pr_number), []).append(body)

    def _assert_can_write_docsite(self) -> None:
        self._assert_can_write_repo("docsite")

    def _assert_can_write_repo(self, repo_role: RepoRole) -> None:
        repo = self._repo_for_role(repo_role)
        if not repo:
            raise UnauthorizedRepositoryError(f"{repo_role} repository is not configured")
        if self.allowed_write_repo is not None and self.allowed_write_repo != repo:
            raise UnauthorizedRepositoryError(f"Writes are restricted to the configured {repo_role} repository")

    def _repo_for_role(self, repo_role: RepoRole) -> str:
        return self.docsite_repo if repo_role == "docsite" else self.playground_repo

    def _files_for_role(self, repo_role: RepoRole) -> dict[str, str]:
        return self.doc_files if repo_role == "docsite" else self.playground_files

    @staticmethod
    def _branch_key(repo_role: RepoRole, branch_name: str) -> str:
        return branch_name if repo_role == "docsite" else f"{repo_role}:{branch_name}"


@dataclass
class GitHubAppClient:
    app_id: str
    private_key: str
    docsite_repo: str
    playground_repo: str = ""
    api_base_url: str = "https://api.github.com"
    timeout_seconds: float = 10.0
    default_branch: str = "main"
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._installation_token: str | None = None
        self._default_branches: dict[str, str] = {}

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
        return self._list_repository_files("docsite")

    def list_playground_files(self) -> list[DocFile]:
        if not self.playground_repo:
            return []
        return self._list_repository_files("playground")

    def create_docsite_branch(self, branch_name: str) -> None:
        self.create_repository_branch("docsite", branch_name)

    def create_repository_branch(self, repo_role: RepoRole, branch_name: str) -> None:
        owner, repo = _split_repo(self._repo_for_role(repo_role))
        branch = self._default_branch_for_repo(repo_role)
        ref = self._get_json(f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        sha = ((ref.get("object") or {}).get("sha"))
        if not sha:
            raise GitHubClientError(f"Could not resolve {repo_role} default branch SHA")
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
        return self.commit_repository_updates("docsite", branch_name, updates)

    def commit_repository_updates(self, repo_role: RepoRole, branch_name: str, updates: list[DocUpdate]) -> list[str]:
        owner, repo = _split_repo(self._repo_for_role(repo_role))
        changed_files: list[str] = []
        for update in updates:
            current_content, sha = self._get_file_content_or_missing(
                owner=owner,
                repo=repo,
                path=update.path,
                ref=branch_name,
            )
            if current_content == update.content:
                continue
            payload = {
                "message": f"Update {update.path} from AI docsite updater",
                "content": base64.b64encode(update.content.encode("utf-8")).decode("ascii"),
                "branch": branch_name,
            }
            if sha:
                payload["sha"] = sha
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
        return self.open_repository_pr("docsite", branch_name, title, body, changed_files)

    def open_repository_pr(
        self,
        repo_role: RepoRole,
        branch_name: str,
        title: str,
        body: str,
        changed_files: list[str],
    ) -> CreatedPullRequest:
        repo_full_name = self._repo_for_role(repo_role)
        owner, repo = _split_repo(repo_full_name)
        response = self._client().post(
            self._url(f"/repos/{owner}/{repo}/pulls"),
            headers=self._installation_headers(),
            json={"title": title, "body": body, "head": branch_name, "base": self._default_branch_for_repo(repo_role)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return CreatedPullRequest(
            url=str(payload.get("html_url")),
            number=int(payload.get("number")),
            branch=branch_name,
            changed_files=changed_files,
            repo_role=repo_role,
            repo=repo_full_name,
        )

    def request_reviewers(self, pr_number: int, reviewers: list[str]) -> None:
        self.request_repository_reviewers("docsite", pr_number, reviewers)

    def request_repository_reviewers(self, repo_role: RepoRole, pr_number: int, reviewers: list[str]) -> None:
        if not reviewers:
            return
        owner, repo = _split_repo(self._repo_for_role(repo_role))
        response = self._client().post(
            self._url(f"/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers"),
            headers=self._installation_headers(),
            json={"reviewers": reviewers},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def fetch_repository_pr_diff(self, repo_role: RepoRole, pr_number: int) -> str:
        owner, repo = _split_repo(self._repo_for_role(repo_role))
        response = self._client().get(
            self._url(f"/repos/{owner}/{repo}/pulls/{pr_number}"),
            headers={**self._installation_headers(), "Accept": "application/vnd.github.v3.diff"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.text

    def comment_on_repository_pr(self, repo_role: RepoRole, pr_number: int, body: str) -> None:
        owner, repo = _split_repo(self._repo_for_role(repo_role))
        response = self._client().post(
            self._url(f"/repos/{owner}/{repo}/issues/{pr_number}/comments"),
            headers=self._installation_headers(),
            json={"body": body},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _docsite_branch(self) -> str:
        return self._default_branch_for_repo("docsite")

    def _default_branch_for_repo(self, repo_role: RepoRole) -> str:
        repo_full_name = self._repo_for_role(repo_role)
        if repo_full_name in self._default_branches:
            return self._default_branches[repo_full_name]
        owner, repo = _split_repo(repo_full_name)
        repo_payload = self._get_json(f"/repos/{owner}/{repo}")
        branch = str(repo_payload.get("default_branch") or self.default_branch)
        self._default_branches[repo_full_name] = branch
        return branch

    def _list_repository_files(self, repo_role: RepoRole) -> list[DocFile]:
        repo_full_name = self._repo_for_role(repo_role)
        owner, repo = _split_repo(repo_full_name)
        branch = self._default_branch_for_repo(repo_role)
        tree = self._get_json(f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1").get("tree", [])
        files: list[DocFile] = []
        for item in tree:
            path = item.get("path")
            if item.get("type") != "blob" or not isinstance(path, str):
                continue
            if repo_role == "docsite" and not _is_doc_file(path):
                continue
            if repo_role == "playground" and not _is_playground_file(path):
                continue
            try:
                content = self._get_file_content(owner=owner, repo=repo, path=path, ref=branch)[0]
            except UnicodeDecodeError:
                continue
            files.append(DocFile(path=path, content=content, repo_role=repo_role))
        return files

    def _repo_for_role(self, repo_role: RepoRole) -> str:
        repo = self.docsite_repo if repo_role == "docsite" else self.playground_repo
        if not repo:
            raise GitHubClientError(f"{repo_role} repository is not configured")
        return repo

    def _get_file_content(self, *, owner: str, repo: str, path: str, ref: str) -> tuple[str, str]:
        payload = self._get_json(f"/repos/{owner}/{repo}/contents/{path}?ref={ref}")
        encoded = payload.get("content")
        sha = payload.get("sha")
        if not isinstance(encoded, str) or not isinstance(sha, str):
            raise GitHubClientError(f"GitHub content response for {path} was missing content or sha")
        content = base64.b64decode(encoded.encode("ascii"), validate=False).decode("utf-8")
        return content, sha

    def _get_file_content_or_missing(self, *, owner: str, repo: str, path: str, ref: str) -> tuple[str, str | None]:
        try:
            return self._get_file_content(owner=owner, repo=repo, path=path, ref=ref)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return "", None
            raise

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
            playground_repo=settings.playground_repo,
            api_base_url=settings.github_api_base_url,
            timeout_seconds=settings.external_timeout_seconds,
        )
    return InMemoryGitHubClient(docsite_repo=settings.docsite_repo, playground_repo=settings.playground_repo)


def _split_repo(full_name: str) -> tuple[str, str]:
    parts = full_name.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise GitHubClientError(f"Invalid repository full name: {full_name}")
    return parts[0], parts[1]


def _is_doc_file(path: str) -> bool:
    lowered = path.lower()
    if lowered in {"mkdocs.yml", "mkdocs.yaml"}:
        return True
    return path.startswith("docs/") and lowered.endswith((".md", ".mdx", ".rst", ".txt", ".adoc"))


def _is_playground_file(path: str) -> bool:
    lowered = path.lower()
    parts = lowered.split("/")
    if any(part in {".git", "node_modules", "dist", "build", "coverage"} for part in parts):
        return False
    if lowered.startswith((".env", ".github/")) or "/.env" in lowered:
        return False
    if lowered.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".lock")):
        return False
    return lowered.endswith((
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".md",
        ".css",
        ".html",
        ".yml",
        ".yaml",
        ".svg",
    ))
