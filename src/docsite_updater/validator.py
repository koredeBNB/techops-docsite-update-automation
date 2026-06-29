from __future__ import annotations

from .models import DocUpdate, ValidationResult


class MockMkDocsValidator:
    def __init__(self, *, should_pass: bool = True) -> None:
        self.should_pass = should_pass
        self.calls: list[list[DocUpdate]] = []

    def validate(self, updates: list[DocUpdate]) -> ValidationResult:
        self.calls.append(updates)
        if not self.should_pass:
            return ValidationResult(ok=False, output="mkdocs build failed: broken markdown")
        for update in updates:
            if "BROKEN_MKDOCS" in update.content:
                return ValidationResult(ok=False, output="mkdocs build failed: broken markdown")
        return ValidationResult(ok=True, output="mkdocs build passed")
