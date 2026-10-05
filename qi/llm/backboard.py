"""Minimal client for the Backboard API (https://docs.backboard.io), used for auto-interp labels and feature
categorization. Backboard routes to Anthropic models; each call here is stateless (memory off, fresh thread).

Env: BACKBOARD_API_KEY (required), BACKBOARD_MODEL (model_name), BACKBOARD_PROVIDER (default "anthropic").
"""
from __future__ import annotations

import json
import os
import time
from typing import Protocol

BASE_URL = "https://app.backboard.io/api"


class LLMClient(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...


class BackboardClient:
    def __init__(self, api_key: str | None = None, model_name: str | None = None, llm_provider: str | None = None,
                 base_url: str = BASE_URL, timeout: float = 120.0, max_retries: int = 4, session=None):
        self.api_key = api_key or os.environ.get("BACKBOARD_API_KEY")
        if not self.api_key:
            raise RuntimeError("BACKBOARD_API_KEY is not set (put it in .env)")
        self.model_name = model_name or os.environ.get("BACKBOARD_MODEL")
        self.llm_provider = llm_provider or os.environ.get("BACKBOARD_PROVIDER", "anthropic")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        """One stateless completion. A system instruction is prepended to the content since the message endpoint
        takes a single content field; memory is off and no thread_id is passed, so nothing persists."""
        content = f"{system.strip()}\n\n{prompt}" if system else prompt
        body = {"content": content, "memory": "off", "send_to_llm": True, "web_search": "off",
                "llm_provider": self.llm_provider}
        if self.model_name:
            body["model_name"] = self.model_name
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            r = self.session.post(f"{self.base_url}/threads/messages", json=body, headers=headers, timeout=self.timeout)
            self.calls += 1
            if r.status_code == 200:
                data = r.json()
                return data["content"] if isinstance(data, dict) and "content" in data else json.dumps(data)
            if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Backboard error {r.status_code}: {r.text[:300]}")
        raise RuntimeError("unreachable")
