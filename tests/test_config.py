from __future__ import annotations

from docsite_updater.config import Settings
from docsite_updater.observability import MemoryLogger


def test_settings_safe_dict_redacts_secrets(settings: Settings) -> None:
    safe = settings.safe_dict()

    assert safe["github_webhook_secret"] == "[REDACTED]"
    assert safe["github_private_key"] == "[REDACTED]"
    assert safe["ai_api_key"] == "[REDACTED]"
    assert safe["docsite_repo"] == "bnb-chain/mock-bnb-docsite"


def test_logger_redacts_secret_fields(settings: Settings) -> None:
    logger = MemoryLogger()

    logger.info("config_loaded", **settings.safe_dict())

    logs = logger.as_json_lines()
    assert "super-private-key" not in logs
    assert "test-webhook-secret" not in logs
    assert "ai-secret-key" not in logs
    assert "[REDACTED]" in logs
