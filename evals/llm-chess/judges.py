"""Judges — compare two responses to the same prompt, return winner + reason.

Each judge carries an explicit `version` string. The version is baked into
match records and judge-call cache keys, so two judge releases never get
their verdicts averaged together by accident.

Verdicts may be "A", "B", or "tie". Position bias (A-vs-B slot preference)
is handled in `arena.py` via an A/B swap rather than in the judge itself.
"""

from __future__ import annotations

import json as _json
import random
from dataclasses import dataclass, field
from typing import Protocol

JUDGE_RUBRIC = """You are judging two LLM responses (A and B) to the same prompt.

Ignore length. Ignore formatting. Judge correctness and helpfulness only.

Rubric — apply each, then pick the overall winner:
1. Correctness — are claims factually right?
2. Relevance — does it address the prompt asked?
3. Completeness — are the key points covered?
4. Hallucination — penalise invented facts, fake citations, made-up names.

If the two responses are equivalent in substance, reply with "tie".

Reply with strict JSON, no prose, no markdown:
{"winner": "A" or "B" or "tie", "reason": "<one short sentence>"}
"""


def render_judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return (
        f"{JUDGE_RUBRIC}\n"
        f"---\nPrompt:\n{prompt}\n\n"
        f"---\nResponse A:\n{response_a}\n\n"
        f"---\nResponse B:\n{response_b}\n"
    )


@dataclass
class Verdict:
    winner: str  # "A", "B", or "tie"
    reason: str


class Judge(Protocol):
    name: str
    version: str

    def judge(self, prompt: str, response_a: str, response_b: str) -> Verdict: ...


@dataclass
class RandomJudge:
    """Coin-flip judge — picks A or B uniformly, no reason. Baseline for sanity.

    Combined with the arena's A/B swap, ~50% of random verdicts collapse to
    ties, so this judge produces a slow random walk in Elo rather than a fast
    one. A real judge that can't beat this is no judge.
    """

    name: str = "random"
    version: str = "v1"
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def judge(self, prompt: str, response_a: str, response_b: str) -> Verdict:
        return Verdict(winner=self._rng.choice(["A", "B"]), reason="")


# Pinned OpenAI snapshot. `gpt-4o` (no date) is a moving alias — OpenAI rolls
# it forward and the same eval run can silently change behaviour. Always pin.
GPT_4O_PIN = "gpt-4o-2024-11-20"


@dataclass
class OpenAIJudge:
    """OpenAI-backed judge. Sends the rubric, parses JSON {winner, reason}.

    `model` is the snapshot ID actually sent to the API. `version` is what
    gets recorded in matches and used as the judge cache key — keep them
    aligned unless you have a reason not to.
    """

    name: str = "gpt-4o-judge"
    model: str = GPT_4O_PIN
    version: str = ""  # filled from `model` in __post_init__ if blank
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.version:
            self.version = self.model

    def judge(self, prompt: str, response_a: str, response_b: str) -> Verdict:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": render_judge_prompt(prompt, response_a, response_b)}
            ],
        )
        text = resp.choices[0].message.content or "{}"
        data = _json.loads(text)
        winner = data.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        return Verdict(winner=winner, reason=str(data.get("reason", "")))


REGISTRY: dict[str, Judge] = {}


def register(judge: Judge) -> None:
    REGISTRY[judge.name] = judge


register(RandomJudge())
register(OpenAIJudge())
