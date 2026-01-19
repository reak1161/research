"""Thin OpenAI client with basic rate limiting and retries.

使い方:
    client = OpenAIClient(model="gpt-4o-mini", min_interval=1.0)
    result = client.generate_json(prompt, expect_json=True)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OpenAIClient:
    """A small wrapper around OpenAI Chat Completions with rate limiting."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        max_retries: int = 3,
        min_interval: float = 1.0,
        temperature: float = 0.2,
        dry_run: bool = False,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.temperature = temperature
        self.dry_run = dry_run
        self._last_call: float = 0.0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        wait = self.min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def generate_json(self, prompt: str, expect_json: bool = True) -> Optional[Dict[str, Any]]:
        """Send prompt and return parsed JSON if possible."""
        if self.dry_run:
            print("[info] LLM dry-run. Prompt would be sent:\n", prompt)
            return None

        self._throttle()
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": 200,
                    "messages": [
                        {"role": "system", "content": "Return JSON only." if expect_json else "Reply briefly."},
                        {"role": "user", "content": prompt},
                    ],
                }
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"].strip()
                if expect_json:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        last_err = RuntimeError("Failed to parse JSON from response")
                        continue
                return {"text": text}
            except (urllib.error.URLError, KeyError, ValueError) as e:  # pragma: no cover - network errors
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.min_interval)
                    continue
                break
        if last_err:
            raise last_err
        return None
