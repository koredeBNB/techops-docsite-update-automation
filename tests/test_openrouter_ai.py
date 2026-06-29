from __future__ import annotations

import json

import httpx
import pytest

from docsite_updater.ai import InvalidAIOutputError, OpenRouterAIClient, create_ai_client, system_prompt
from docsite_updater.config import Settings
from docsite_updater.models import ChangedFile, DocFile, PullRequestContext


def make_pr_context() -> PullRequestContext:
    return PullRequestContext(
        source_repo="koredeBNB/mock-bsc-app",
        pr_number=7,
        merge_commit_sha="abc123",
        source_pr_url="https://github.com/koredeBNB/mock-bsc-app/pull/7",
        diff="+        \"voting_power\": 1000,",
        changed_files=[ChangedFile(path="src/mock_bsc_app/validators.py", status="modified")],
    )


def test_factory_selects_openrouter_deepseek_client() -> None:
    settings = Settings(
        github_app_id="123",
        github_private_key="private",
        github_webhook_secret="webhook",
        docsite_repo="koredeBNB/mock-mkdocs-repo",
        ai_api_key="",
        ai_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="deepseek/deepseek-v3.2",
    )

    client = create_ai_client(settings)

    assert isinstance(client, OpenRouterAIClient)
    assert client.model == "deepseek/deepseek-v3.2"


def test_openrouter_default_http_client_ignores_proxy_env() -> None:
    client = OpenRouterAIClient(api_key="or-key")

    http_client = client._client()

    assert http_client._trust_env is False
    http_client.close()


def test_system_prompt_forbids_unrelated_value_mutations() -> None:
    prompt = system_prompt()

    assert "Do not modify existing documented values" in prompt
    assert "keep all existing example values exactly unchanged" in prompt


def test_openrouter_client_posts_strict_json_request_and_parses_updates() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["referer"] = request.headers["http-referer"]
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "updates",
                                    "summary": "Added voting power docs.",
                                    "updates": [
                                        {
                                            "path": "docs/validators.md",
                                            "content": "# Validators\n\nDocuments voting power.",
                                            "rationale": "The API now returns voting_power.",
                                            "confidence": 0.93,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.002},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenRouterAIClient(
        api_key="or-key",
        model="deepseek/deepseek-v3.2",
        base_url="https://openrouter.ai/api/v1",
        http_referer="https://example.test",
        client=http_client,
    )

    result = client.propose_doc_updates(make_pr_context(), [DocFile(path="docs/validators.md", content="# Validators")])

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer or-key"
    assert captured["referer"] == "https://example.test"
    assert captured["body"]["model"] == "deepseek/deepseek-v3.2"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert result.status == "updates"
    assert result.updates[0].path == "docs/validators.md"
    assert result.usage.model == "deepseek/deepseek-v3.2"
    assert result.usage.input_tokens == 100


def test_openrouter_client_returns_no_changes_needed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"status":"no_changes_needed","summary":"Docs are already current.","updates":[]}'}}
                ]
            },
        )

    client = OpenRouterAIClient(api_key="or-key", client=httpx.Client(transport=httpx.MockTransport(handler)))

    result = client.propose_doc_updates(make_pr_context(), [DocFile(path="docs/validators.md", content="# Validators")])

    assert result.status == "no_changes_needed"
    assert result.updates == []


def test_openrouter_client_rejects_invalid_json() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    client = OpenRouterAIClient(api_key="or-key", client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(InvalidAIOutputError):
        client.propose_doc_updates(make_pr_context(), [DocFile(path="docs/validators.md", content="# Validators")])


def test_openrouter_client_requires_api_key() -> None:
    client = OpenRouterAIClient(api_key="")

    with pytest.raises(InvalidAIOutputError):
        client.propose_doc_updates(make_pr_context(), [])
