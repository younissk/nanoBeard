"""Thin OpenAI-compatible chat client for the eval gates.

Everything here talks to a llama-server (`make serve`, or spawned on demand),
which is the same surface the chat playground and rejection sampling use. That
means one prompt-rendering path for the whole repo rather than an eval harness
with its own subtly different idea of the chat template.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class ChatClient:
    base_url: str
    system: str | None = None
    # Qwen3 thinks by default. Thinking is great for accuracy and terrible for a
    # 512-token budget, so it is opt-in and reported alongside the score.
    enable_thinking: bool = False
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 180.0
    retries: int = 3

    def chat(self, user: str, tools: list[dict] | None = None) -> Reply:
        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.load(r)
                msg = body["choices"][0]["message"]
                return Reply(
                    content=(msg.get("content") or "").strip(),
                    tool_calls=msg.get("tool_calls") or [],
                )
            except (urllib.error.URLError, TimeoutError, OSError, KeyError) as e:
                if attempt == self.retries - 1:
                    return Reply(content="", error=f"{type(e).__name__}: {e}")
        return Reply(content="", error="unreachable")

    def map(self, items: list, fn, workers: int = 4) -> list:
        """Run `fn(item)` concurrently. llama-server handles parallel slots, and
        a 200-problem eval is otherwise minutes of waiting on one request."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))
