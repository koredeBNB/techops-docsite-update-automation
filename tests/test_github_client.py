from __future__ import annotations

import base64
import json

import httpx
import pytest

from docsite_updater.config import Settings
from docsite_updater.github import GitHubAppClient, InMemoryGitHubClient, SourcePullRequestFixture, UnauthorizedRepositoryError, create_github_client
from docsite_updater.models import ChangedFile, DocUpdate
from tests.test_queue import make_job


def make_github() -> InMemoryGitHubClient:
    job = make_job()
    return InMemoryGitHubClient(
        docsite_repo="bnb-chain/mock-bnb-docsite",
        source_prs={
            job.identity: SourcePullRequestFixture(
                repo=job.source_repo,
                number=job.pr_number,
                merge_commit_sha=job.merge_commit_sha,
                url=job.source_pr_url,
                diff="diff --git a/src/api.py b/src/api.py\n+def get_validator(): pass",
                changed_files=[ChangedFile(path="src/api.py", status="modified")],
            )
        },
        doc_files={"docs/api.md": "# API\n\nOld API docs."},
    )


def test_creates_installation_token_not_pat() -> None:
    github = make_github()

    token = github.create_installation_token(42)

    assert token == "installation-token-42"
    assert github.installation_tokens == [42]


def test_fetches_pr_metadata_files_and_diff() -> None:
    github = make_github()

    context = github.fetch_pr_context(make_job())

    assert context.source_repo == "bnb-chain/mock-bsc-app"
    assert context.pr_number == 7
    assert context.diff.startswith("diff --git")
    assert context.changed_files[0].path == "src/api.py"


def test_lists_candidate_doc_files() -> None:
    github = make_github()

    docs = github.list_doc_files()

    assert len(docs) == 1
    assert docs[0].path == "docs/api.md"


def test_lists_mkdocs_config_as_candidate_docsite_file() -> None:
    github = InMemoryGitHubClient(
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        doc_files={
            "docs/index.md": "# Home\n",
            "mkdocs.yml": "nav:\n  - Home: index.md\n",
        },
    )

    docs = github.list_doc_files()

    assert [doc.path for doc in docs] == ["docs/index.md", "mkdocs.yml"]


def test_creates_branch_commits_changes_and_opens_pr() -> None:
    github = make_github()
    update = DocUpdate(path="docs/api.md", content="# API\n\nNew API docs.", rationale="API changed")

    github.create_docsite_branch("ai-docs/test")
    changed_files = github.commit_doc_updates("ai-docs/test", [update])
    pr = github.open_docsite_pr("ai-docs/test", "Update docs", "Source PR: test", changed_files)

    assert changed_files == ["docs/api.md"]
    assert github.doc_files["docs/api.md"] == "# API\n\nNew API docs."
    assert pr.url == "https://github.com/bnb-chain/mock-bnb-docsite/pull/1"
    assert pr.changed_files == ["docs/api.md"]


def test_in_memory_client_lists_and_updates_playground_files() -> None:
    github = InMemoryGitHubClient(
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        playground_repo="koredeBNB/techops-docsite-interactive-playground",
        playground_files={
            "src/validatorStatus.ts": "export const response = {}\n",
            "dist/assets/index.js": "generated",
            ".github/workflows/deploy.yml": "name: deploy",
        },
    )

    files = github.list_playground_files()
    update = DocUpdate(
        path="src/validatorStatus.ts",
        content="export const response = { reward_rate: 0.12 }\n",
        rationale="Playground mirrors API response.",
        repo_role="playground",
    )
    github.create_repository_branch("playground", "ai-playground/test")
    changed_files = github.commit_repository_updates("playground", "ai-playground/test", [update])
    pr = github.open_repository_pr("playground", "ai-playground/test", "Update playground", "Body", changed_files)

    assert [file.path for file in files] == ["src/validatorStatus.ts"]
    assert github.playground_files["src/validatorStatus.ts"] == "export const response = { reward_rate: 0.12 }\n"
    assert pr.url == "https://github.com/koredeBNB/techops-docsite-interactive-playground/pull/1"
    assert pr.repo_role == "playground"


def test_request_reviewers_records_reviewers() -> None:
    github = make_github()

    github.request_reviewers(1, ["docs-reviewer"])

    assert github.requested_reviewers[1] == ["docs-reviewer"]


def test_in_memory_client_fetches_diff_and_records_pr_comments() -> None:
    github = make_github()
    github.create_docsite_branch("ai-docs/test")
    changed = github.commit_doc_updates(
        "ai-docs/test",
        [DocUpdate(path="docs/api.md", content="# API\n\nNew API docs.", rationale="API changed")],
    )
    pr = github.open_docsite_pr("ai-docs/test", "Update docs", "Body", changed)

    diff = github.fetch_repository_pr_diff("docsite", pr.number)
    github.comment_on_repository_pr("docsite", pr.number, "AI review comment")

    assert "docs/api.md" in diff
    assert github.pr_comments[("docsite", pr.number)] == ["AI review comment"]


def test_write_operations_are_restricted_to_docsite_repo() -> None:
    github = make_github()
    github.allowed_write_repo = "bnb-chain/mock-bsc-app"

    with pytest.raises(UnauthorizedRepositoryError):
        github.create_docsite_branch("ai-docs/test")


def test_factory_selects_real_github_app_client() -> None:
    settings = Settings(
        github_app_id="123",
        github_private_key="private",
        github_webhook_secret="webhook",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        playground_repo="koredeBNB/techops-docsite-interactive-playground",
        ai_api_key="ai-key",
        github_client="app",
    )

    client = create_github_client(settings)

    assert isinstance(client, GitHubAppClient)
    assert client.docsite_repo == "koredeBNB/mock-mkdocs-repo"
    assert client.playground_repo == "koredeBNB/techops-docsite-interactive-playground"


def test_github_app_client_default_http_client_ignores_proxy_env() -> None:
    client = GitHubAppClient(app_id="123", private_key="private", docsite_repo="koredeBNB/mock-mkdocs-repo")

    http_client = client._client()

    assert http_client._trust_env is False
    http_client.close()


def test_github_app_client_uses_installation_token_and_fetches_pr_context(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url), request.headers.get("authorization")))
        if request.method == "POST" and request.url.path == "/app/installations/42/access_tokens":
            return httpx.Response(201, json={"token": "installation-token"})
        if request.method == "GET" and request.url.path == "/repos/bnb-chain/mock-bsc-app/pulls/7":
            if request.headers.get("accept") == "application/vnd.github.v3.diff":
                return httpx.Response(200, text="diff --git a/src/api.py b/src/api.py\n+ voting_power")
            return httpx.Response(
                200,
                json={
                    "merge_commit_sha": "abc123",
                    "html_url": "https://github.com/bnb-chain/mock-bsc-app/pull/7",
                },
            )
        if request.method == "GET" and request.url.path == "/repos/bnb-chain/mock-bsc-app/pulls/7/files":
            return httpx.Response(200, json=[{"filename": "src/api.py", "status": "modified"}])
        return httpx.Response(404, json={"message": "not found"})

    client = GitHubAppClient(
        app_id="123",
        private_key="private",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(client, "_app_jwt", lambda: "app-jwt")

    token = client.create_installation_token(42)
    context = client.fetch_pr_context(make_job())

    assert token == "installation-token"
    assert context.diff.startswith("diff --git")
    assert context.changed_files[0].path == "src/api.py"
    assert requests[0] == ("POST", "https://api.github.com/app/installations/42/access_tokens", "Bearer app-jwt")
    assert all(auth == "Bearer installation-token" for _, _, auth in requests[1:])


def test_github_app_client_lists_docs_commits_and_opens_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_content = "# Validators\n\nOld docs."
    encoded_doc = base64.b64encode(doc_content.encode("utf-8")).decode("ascii")
    seen: dict[str, object] = {"put_payload": None, "pull_payload": None, "review_payload": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/app/installations/42/access_tokens":
            return httpx.Response(201, json={"token": "installation-token"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/git/trees/main":
            return httpx.Response(
                200,
                json={"tree": [{"type": "blob", "path": "docs/validators.md"}, {"type": "blob", "path": "mkdocs.yml"}]},
            )
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/contents/docs/validators.md":
            return httpx.Response(200, json={"content": encoded_doc, "sha": "file-sha"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/contents/mkdocs.yml":
            return httpx.Response(200, json={"content": base64.b64encode(b"nav: []\n").decode("ascii"), "sha": "mkdocs-sha"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/git/ref/heads/main":
            return httpx.Response(200, json={"object": {"sha": "branch-sha"}})
        if request.method == "POST" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/git/refs":
            return httpx.Response(201, json={"ref": "refs/heads/ai-docs/test"})
        if request.method == "PUT" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/contents/docs/validators.md":
            seen["put_payload"] = json.loads(request.content)
            return httpx.Response(200, json={"content": {"sha": "new-file-sha"}})
        if request.method == "POST" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/pulls":
            seen["pull_payload"] = json.loads(request.content)
            return httpx.Response(201, json={"html_url": "https://github.com/koredeBNB/mock-mkdocs-repo/pull/3", "number": 3})
        if request.method == "POST" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/pulls/3/requested_reviewers":
            seen["review_payload"] = json.loads(request.content)
            return httpx.Response(201, json={})
        return httpx.Response(404, json={"message": f"not found: {request.method} {request.url.path}"})

    client = GitHubAppClient(
        app_id="123",
        private_key="private",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(client, "_app_jwt", lambda: "app-jwt")
    client.create_installation_token(42)

    docs = client.list_doc_files()
    client.create_docsite_branch("ai-docs/test")
    changed = client.commit_doc_updates(
        "ai-docs/test",
        [DocUpdate(path="docs/validators.md", content="# Validators\n\nNew docs.", rationale="API changed")],
    )
    pr = client.open_docsite_pr("ai-docs/test", "Update docs", "Body", changed)
    client.request_reviewers(pr.number, ["koredeBNB"])

    assert [doc.path for doc in docs] == ["docs/validators.md", "mkdocs.yml"]
    assert docs[0].content == doc_content
    assert changed == ["docs/validators.md"]
    assert seen["put_payload"]["branch"] == "ai-docs/test"
    assert seen["put_payload"]["sha"] == "file-sha"
    assert seen["pull_payload"] == {"title": "Update docs", "body": "Body", "head": "ai-docs/test", "base": "main"}
    assert seen["review_payload"] == {"reviewers": ["koredeBNB"]}
    assert pr.url == "https://github.com/koredeBNB/mock-mkdocs-repo/pull/3"


def test_github_app_client_fetches_pr_diff_and_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {"comment_payload": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/app/installations/42/access_tokens":
            return httpx.Response(201, json={"token": "installation-token"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/pulls/3":
            if request.headers.get("accept") == "application/vnd.github.v3.diff":
                return httpx.Response(200, text="diff --git a/docs/index.md b/docs/index.md")
            return httpx.Response(200, json={"html_url": "https://github.com/koredeBNB/mock-mkdocs-repo/pull/3"})
        if request.method == "POST" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/issues/3/comments":
            seen["comment_payload"] = json.loads(request.content)
            return httpx.Response(201, json={"id": 99})
        return httpx.Response(404, json={"message": f"not found: {request.method} {request.url.path}"})

    client = GitHubAppClient(
        app_id="123",
        private_key="private",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(client, "_app_jwt", lambda: "app-jwt")
    client.create_installation_token(42)

    diff = client.fetch_repository_pr_diff("docsite", 3)
    client.comment_on_repository_pr("docsite", 3, "AI review comment")

    assert "docs/index.md" in diff
    assert seen["comment_payload"] == {"body": "AI review comment"}


def test_github_app_client_can_create_new_files(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {"put_payload": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/app/installations/42/access_tokens":
            return httpx.Response(201, json={"token": "installation-token"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/git/ref/heads/main":
            return httpx.Response(200, json={"object": {"sha": "branch-sha"}})
        if request.method == "POST" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/git/refs":
            return httpx.Response(201, json={"ref": "refs/heads/ai-docs/test"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/contents/docs/gas-fees.md":
            return httpx.Response(404, json={"message": "Not Found"})
        if request.method == "PUT" and request.url.path == "/repos/koredeBNB/mock-mkdocs-repo/contents/docs/gas-fees.md":
            seen["put_payload"] = json.loads(request.content)
            return httpx.Response(201, json={"content": {"sha": "new-file-sha"}})
        return httpx.Response(404, json={"message": f"not found: {request.method} {request.url.path}"})

    client = GitHubAppClient(
        app_id="123",
        private_key="private",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(client, "_app_jwt", lambda: "app-jwt")
    client.create_installation_token(42)

    client.create_docsite_branch("ai-docs/test")
    changed = client.commit_doc_updates(
        "ai-docs/test",
        [DocUpdate(path="docs/gas-fees.md", content="# Gas Fees\n", rationale="New gas fees docs")],
    )

    assert changed == ["docs/gas-fees.md"]
    assert seen["put_payload"]["branch"] == "ai-docs/test"
    assert "sha" not in seen["put_payload"]


def test_github_app_client_lists_playground_files(monkeypatch: pytest.MonkeyPatch) -> None:
    ts_content = "export const response = {}\n"
    encoded_ts = base64.b64encode(ts_content.encode("utf-8")).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/app/installations/42/access_tokens":
            return httpx.Response(201, json={"token": "installation-token"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/techops-docsite-interactive-playground":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/techops-docsite-interactive-playground/git/trees/main":
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {"type": "blob", "path": "src/validatorStatus.ts"},
                        {"type": "blob", "path": "dist/assets/index.js"},
                    ]
                },
            )
        if request.method == "GET" and request.url.path == "/repos/koredeBNB/techops-docsite-interactive-playground/contents/src/validatorStatus.ts":
            return httpx.Response(200, json={"content": encoded_ts, "sha": "file-sha"})
        return httpx.Response(404, json={"message": f"not found: {request.method} {request.url.path}"})

    client = GitHubAppClient(
        app_id="123",
        private_key="private",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        playground_repo="koredeBNB/techops-docsite-interactive-playground",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(client, "_app_jwt", lambda: "app-jwt")
    client.create_installation_token(42)

    files = client.list_playground_files()

    assert len(files) == 1
    assert files[0].repo_role == "playground"
    assert files[0].path == "src/validatorStatus.ts"
    assert files[0].content == ts_content
