from __future__ import annotations

import hmac
from hashlib import sha256


def sign_body(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    return f"sha256={digest}"


def verify_github_signature(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature:
        return False
    expected = sign_body(body, secret)
    return hmac.compare_digest(expected, signature)
