"""GSM8K exact-match — the reasoning gate.

Grade-school word problems with a single numeric answer, so grading is exact
and needs no judge model. That is the whole reason this benchmark is the one
worth wiring first: it is cheap, deterministic, and it moves when reasoning
moves.

Answer extraction is the part that quietly ruins these harnesses. A model that
says "she sold 72 clips" is correct; one that says "48 + 24 = 72 clips" is also
correct; one that trails off mid-sentence is not. We take the LAST number in the
reply, after preferring an explicit `#### N`, because models put the final
answer last and intermediate arithmetic earlier.

Hold gsm8k TEST out of every training mix, forever. MetaMathQA and friends are
augmented from gsm8k TRAIN, so they are safe; the moment test leaks, this number
stops meaning anything and you will not be able to tell.
"""

from __future__ import annotations

import re

from nanobeard.evals.client import ChatClient

DATASET, CONFIG, SPLIT = "openai/gsm8k", "main", "test"
# Fixed slice so runs are comparable. Seeded shuffle, not the first N, because
# the dataset has mild ordering structure.
DEFAULT_N = 200
SEED = 1337

INSTRUCTION = (
    "Solve the problem. Show your working briefly, then give the final numeric "
    "answer on its own last line in the form: #### <number>"
)
_FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def normalize(raw: str) -> str | None:
    """`$1,234.00` and `1234` are the same answer; make them the same string."""
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    # Integers are the common case; don't let 72 != 72.0 cost a point.
    return str(int(f)) if f == int(f) else str(f)


def extract_answer(text: str) -> str | None:
    """Prefer the `#### N` marker; otherwise the last number in the reply."""
    marked = _FINAL.findall(text)
    if marked:
        return normalize(marked[-1])
    numbers = _NUMBER.findall(text)
    return normalize(numbers[-1]) if numbers else None


def gold_answer(answer_field: str) -> str | None:
    """GSM8K reference answers always end with `#### N`."""
    return extract_answer(answer_field)


def load_problems(n: int = DEFAULT_N, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG, split=SPLIT).shuffle(seed=seed)
    n = min(n, len(ds))
    return [{"question": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])} for i in range(n)]


def run(client: ChatClient, problems: list[dict], workers: int = 4) -> dict:
    def solve(p: dict) -> dict:
        reply = client.chat(f"{p['question']}\n\n{INSTRUCTION}")
        got = extract_answer(reply.content)
        return {
            "question": p["question"],
            "gold": p["gold"],
            "got": got,
            "correct": got is not None and got == p["gold"],
            "no_answer": got is None,
            "error": reply.error,
            "reply": reply.content,
        }

    rows = client.map(problems, solve, workers=workers)
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "name": "gsm8k",
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "no_answer": sum(r["no_answer"] for r in rows),
        "errors": sum(1 for r in rows if r["error"]),
        "rows": rows,
    }
