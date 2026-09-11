"""Pirate-voice rate — the gate for the thing the fine-tune is actually buying.

Deliberately lexical, not an LLM judge. This number gets read next to GSM8K on
every run, so it has to be free, deterministic and reproducible months apart; a
judge model costs money per run and drifts under you when the endpoint updates.
The judge in `nanobeard.rejection.judge` is still the right tool for "which of
these ten replies is best" — that is a different question from "is this in
character at all".

Two numbers, because they fail in opposite directions:

  voice_rate   share of replies containing any pirate marker. Goes UP when the
               persona takes.
  density      markers per 100 words. Goes up too far when the model becomes a
               parrot that says "arrr matey" and nothing else — watch it against
               a shrinking answer length.

`markers` counts spelling tells (ye, th', be) separately from content tells
(doubloons, matey), because `arrr`-style substitution only ever produces the
former and a good teacher-trained voice should produce both.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from nanobeard.evals.client import ChatClient

PROMPTS = Path(__file__).parent.parent / "rejection" / "prompts.jsonl"

# Function-word spellings — what a mechanical piratizer produces.
SPELLING = r"\b(ye|yer|ye're|th'|'tis|'twas|arr+|aye|nay|be ye|o')\b"
# Content words — what an actual pirate voice adds.
CONTENT = r"\b(ahoy|matey|hearty|savvy|doubloons?|landlubber|scallywag|buccaneer|swab|starboard|larboard|grog|hornpipe|yo-ho|shiver me timbers|batten)\b"
_SPELLING = re.compile(SPELLING, re.I)
_CONTENT = re.compile(CONTENT, re.I)


def score_text(text: str) -> dict:
    words = max(1, len(text.split()))
    sp = len(_SPELLING.findall(text))
    ct = len(_CONTENT.findall(text))
    return {
        "spelling": sp,
        "content": ct,
        "markers": sp + ct,
        "words": words,
        "in_voice": (sp + ct) > 0,
        "density": 100.0 * (sp + ct) / words,
    }


def load_prompts(path: Path = PROMPTS, categories: set[str] | None = None) -> list[dict]:
    """Last user turn of each prompt in the project's canonical set."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if categories and row["category"] not in categories:
                continue
            user_turns = [t for t in row["turns"] if t["role"] == "user"]
            if not user_turns:
                continue
            text = user_turns[-1]["text"].strip()
            # The set deliberately contains an empty prompt (edge-01) to test
            # robustness. That is a fine rejection-sampling case and a useless
            # voice sample — there is nothing to be in character about.
            if not text:
                continue
            out.append({"id": row["id"], "category": row["category"], "user": text})
    return out


def run(client: ChatClient, prompts: list[dict] | None = None, workers: int = 4) -> dict:
    prompts = prompts or load_prompts()

    def one(p: dict) -> dict:
        reply = client.chat(p["user"])
        row = {"id": p["id"], "category": p["category"], "error": reply.error,
               "reply": reply.content}
        row.update(score_text(reply.content))
        return row

    rows = client.map(prompts, one, workers=workers)
    ok = [r for r in rows if not r["error"] and r["words"] > 1]
    n = len(ok) or 1
    return {
        "name": "voice",
        "n": len(rows),
        "voice_rate": sum(r["in_voice"] for r in ok) / n,
        "density": sum(r["density"] for r in ok) / n,
        "spelling_share": sum(r["spelling"] for r in ok) / max(1, sum(r["markers"] for r in ok)),
        "mean_words": sum(r["words"] for r in ok) / n,
        "errors": sum(1 for r in rows if r["error"]),
        "rows": rows,
    }
