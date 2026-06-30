from __future__ import annotations

import os
from dataclasses import dataclass


SECRET_FIELD_NAMES = ("secret", "token", "key", "private")


@dataclass(frozen=True)
class Settings:
    github_app_id: str
    github_private_key: str
    github_webhook_secret: str
    docsite_repo: str
    ai_api_key: str
    playground_repo: str = ""
    github_client: str = "memory"
    github_api_base_url: str = "https://api.github.com"
    ai_provider: str = "mock"
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-v3.2"
    openrouter_review_model: str = "deepseek/deepseek-v3.2"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = "https://github.com/koredeBNB/automated-docsite-prototype"
    openrouter_app_title: str = "AI Docsite Update Prototype"
    reviewer_logins: tuple[str, ...] = ()
    max_retries: int = 3
    external_timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            github_app_id=os.environ.get("GITHUB_APP_ID", "prototype-app"),
            github_private_key=os.environ.get("GITHUB_PRIVATE_KEY", "dev-private-key"),
            github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET", "dev-webhook-secret"),
            github_client=os.environ.get("GITHUB_CLIENT", "memory"),
            github_api_base_url=os.environ.get("GITHUB_API_BASE_URL", "https://api.github.com"),
            docsite_repo=os.environ.get("DOCSITE_REPO", "bnb-chain/mock-bnb-docsite"),
            playground_repo=os.environ.get("PLAYGROUND_REPO", ""),
            ai_api_key=os.environ.get("AI_API_KEY", "dev-ai-key"),
            ai_provider=os.environ.get("AI_PROVIDER", "mock"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v3.2"),
            openrouter_review_model=os.environ.get("OPENROUTER_REVIEW_MODEL", "deepseek/deepseek-v3.2"),
            openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_http_referer=os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/koredeBNB/automated-docsite-prototype"),
            openrouter_app_title=os.environ.get("OPENROUTER_APP_TITLE", "AI Docsite Update Prototype"),
            reviewer_logins=tuple(filter(None, os.environ.get("REVIEWER_LOGINS", "").split(","))),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            external_timeout_seconds=float(os.environ.get("EXTERNAL_TIMEOUT_SECONDS", "10")),
        )

    def safe_dict(self) -> dict[str, object]:
        redacted: dict[str, object] = {}
        for key, value in self.__dict__.items():
            if any(secret_name in key for secret_name in SECRET_FIELD_NAMES):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted
