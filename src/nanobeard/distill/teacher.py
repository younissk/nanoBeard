"""Kimi (Moonshot) client for generating SFT data.

Three things about this API cost real money to learn, so they are encoded here:

1. **kimi-k2.6 is a reasoning model.** It returns `reasoning_content` alongside
   `content`, and the reasoning is billed. Measured on a one-line arithmetic
   question: 336 of 387 completion tokens were reasoning. Budget accordingly.

2. **max_tokens too low returns an EMPTY string, not an error.** The reasoning
   eats the whole allowance and `content` comes back "" with
   `finish_reason: stop`. At max_tokens=300 the math case produced nothing and
   still charged for 300 tokens. `MIN_MAX_TOKENS` guards against re-learning this.

3. **The allowed temperature depends on whether thinking is on.** With thinking
   the API accepts only `temperature=1`; with `thinking: {"type": "disabled"}`
   it accepts only `0.6`. Both are 400s otherwise, and the error message is the
   only place this is documented.

Thinking is OFF by default. Measured on one chat turn: 285 completion tokens
with thinking (84% of them reasoning, billed and discarded) versus 45 without —
roughly 6x the cost for output that was, if anything, less characterful. Turn it
back on per-call for anything where the format is fiddly enough to be worth it.

Every call's usage is accumulated so a run can report what it actually spent
rather than what it estimated.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from nanobeard.env import load_env

BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k2.6"
# Reasoning tokens come out of max_tokens. Below roughly this, `content` starts
# coming back empty for anything that needs a moment's thought. Only enforced
# when thinking is on; without it, replies are short and this would be silly.
MIN_MAX_TOKENS = 900
# The API permits exactly one temperature per mode. Not a default — a constraint.
TEMP_THINKING = 1.0
TEMP_NO_THINKING = 0.6


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    empty_content: int = 0
    errors: int = 0

    def add(self, u: dict) -> None:
        self.calls += 1
        self.prompt_tokens += u.get("prompt_tokens", 0)
        self.completion_tokens += u.get("completion_tokens", 0)
        self.reasoning_tokens += (
            u.get("completion_tokens_details", {}) or {}
        ).get("reasoning_tokens", 0)

    def summary(self) -> str:
        r = self.reasoning_tokens
        c = self.completion_tokens or 1
        return (
            f"{self.calls} calls | {self.prompt_tokens:,} in / {self.completion_tokens:,} out "
            f"({r:,} of those reasoning = {r / c:.0%}) | "
            f"{self.empty_content} empty, {self.errors} errors"
        )


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    reasoning: str = ""
    error: str | None = None


@dataclass
class Teacher:
    model: str = DEFAULT_MODEL
    max_tokens: int = 1400
    thinking: bool = False
    timeout: float = 180.0
    retries: int = 3
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        load_env()
        key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        if not key:
            raise RuntimeError("Set KIMI_API_KEY in envs/.env")
        self._key = key
        if self.thinking and self.max_tokens < MIN_MAX_TOKENS:
            raise ValueError(
                f"max_tokens={self.max_tokens} is below {MIN_MAX_TOKENS}: reasoning "
                f"tokens would consume the allowance and content would come back empty"
            )

    def ask(
        self,
        system: str,
        user: str,
        tools: list[dict] | None = None,
        thinking: bool | None = None,
    ) -> Reply:
        think = self.thinking if thinking is None else thinking
        payload: dict = {
            "model": self.model,
            "temperature": TEMP_THINKING if think else TEMP_NO_THINKING,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if not think:
            payload["thinking"] = {"type": "disabled"}
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.load(r)
                break
            except urllib.error.HTTPError as e:
                # Never echo the request: the Authorization header carries the key.
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self.usage.errors += 1
                return Reply(content="", error=f"HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self.usage.errors += 1
                return Reply(content="", error=type(e).__name__)
        else:
            self.usage.errors += 1
            return Reply(content="", error="retries exhausted")

        self.usage.add(body.get("usage", {}))
        msg = body["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        calls = msg.get("tool_calls") or []
        if not content and not calls:
            self.usage.empty_content += 1
        return Reply(content=content, tool_calls=calls, reasoning=msg.get("reasoning_content") or "")


def balance() -> float | None:
    """Remaining credit, for measuring what a run actually cost."""
    load_env()
    key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    if not key:
        return None
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/users/me/balance", headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return float(json.load(r)["data"]["available_balance"])
    except Exception:
        return None
