"""Shared HTTP helpers: a simple rate limiter and retrying POST for the local/free providers."""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx


class ProviderError(RuntimeError):
    """Raised when a model provider cannot be reached or returns an error."""


class RateLimiter:
    """Allow at most ``per_minute`` calls per minute by spacing calls evenly."""

    def __init__(self, per_minute: float | None) -> None:
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    """POST JSON with retries on connection errors, 429 and 5xx responses."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if response.status_code == 429 or response.status_code >= 500:
                last_error = ProviderError(
                    f"{response.status_code} from {url}: {response.text[:200]}"
                )
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < retries:
                    time.sleep(min(int(retry_after), 60))
                    continue
            elif response.status_code >= 400:
                raise ProviderError(f"{response.status_code} from {url}: {response.text[:500]}")
            else:
                return response.json()
        if attempt < retries:
            time.sleep(min(2**attempt, 20))
    raise ProviderError(f"Request to {url} failed after {retries + 1} attempts: {last_error}")
