from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from .config import SECRET_FIELD_NAMES


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(secret_name in key.lower() for secret_name in SECRET_FIELD_NAMES) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class MemoryLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def info(self, message: str, **fields: object) -> None:
        self._log("info", message, fields)

    def warning(self, message: str, **fields: object) -> None:
        self._log("warning", message, fields)

    def error(self, message: str, **fields: object) -> None:
        self._log("error", message, fields)

    def _log(self, level: str, message: str, fields: dict[str, object]) -> None:
        record = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **redact(fields),
        }
        self.records.append(record)

    def as_json_lines(self) -> str:
        return "\n".join(json.dumps(record, sort_keys=True) for record in self.records)


class MemoryMetrics:
    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.observations: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.observations[name].append(value)
