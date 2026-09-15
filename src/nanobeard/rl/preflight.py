"""Check everything an RL run needs, before paying for a GPU.

Nine Vast launches were spent on this project discovering problems that a
thirty-second local check would have caught. This is that check: corpus, model,
protocol, reward, and the credentials needed to get results back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nanobeard.rl.corpus import DEFAULT_INDEX


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{f' — {detail}' if detail else ''}")
    return ok


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from nanobeard.env import load_env

    load_env()
    results = []
    print("preflight\n")

    # --- corpus ---
    path = Path(args.index)
    if path.exists():
        from nanobeard.rl.corpus import load

        index, questions = load(path)
        results.append(check("search index", len(index) > 1000,
                             f"{len(index):,} paragraphs, {len(questions):,} questions"))
        q = questions[0]
        titles = [index.docs[i].title for i, _ in index.search(q.question, k=5)]
        results.append(check("BM25 returns results", bool(titles), f"top: {titles[0]}"))
    else:
        results.append(check("search index", False, "missing — run `make rl-corpus`"))
        sys.exit(1)

    # --- the protocol, end to end, with a scripted policy ---
    from nanobeard.rl import rewards as R
    from nanobeard.rl.env import run_episodes

    scripted = iter([["<search>test</search>", "<answer>yes</answer>"]])
    plan = next(scripted)
    turn = {"i": 0}

    def fake(bodies):
        out = [plan[min(turn["i"], len(plan) - 1)]] * len(bodies)
        turn["i"] += 1
        return out

    ep = run_episodes([questions[0]], index, fake, max_searches=1)[0]
    results.append(check("episode protocol", ep.answer is not None and ep.did_search,
                         f"steps={[s.kind for s in ep.steps]}"))
    results.append(check("generated spans exclude results",
                         "<results>" not in "".join(
                             ep.transcript[a:b] for a, b in ep.generated_spans)))

    # --- reward cannot be gamed by retrieval alone ---
    cheat = R.compute("wrong", "right", did_search=True,
                      retrieved_titles=["A", "B"], gold_titles=["A", "B"])
    honest = R.compute("right", "right", did_search=True, retrieved_titles=[],
                       gold_titles=["A", "B"])
    results.append(check("answering beats retrieving", honest.total > cheat.total,
                         f"{honest.total:.2f} vs {cheat.total:.2f}"))

    # --- training deps and credentials ---
    try:
        import peft  # noqa: F401
        import torch
        import transformers  # noqa: F401

        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        results.append(check("torch + peft + transformers", True, f"device={dev}"))
        if dev == "cpu":
            print("         (cpu works but a real run wants a GPU)")
    except ImportError as e:
        results.append(check("torch + peft + transformers", False,
                             f"{e} — `uv sync --group finetune`"))

    tok = os.getenv("HF_TOKEN")
    results.append(check("HF_TOKEN set", bool(tok and tok != "none"),
                         "needed for --push-to-hub, the only reliable way off a rented box"))

    print()
    if all(results):
        print("ready. suggested first run:")
        print("  make rl-train RL_STEPS=150 RL_GROUP=8")
    else:
        print("not ready — fix the FAIL lines above")
        sys.exit(1)


if __name__ == "__main__":
    main()
