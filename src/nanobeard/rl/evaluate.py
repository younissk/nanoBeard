"""Measure search ability on questions the policy never trained on.

    uv run python -m nanobeard.rl.evaluate --model <gguf> --label search-v3

Serves the GGUF through llama-server and plays the same `<search>`/`<answer>`
episodes the RL loop used, scoring with the same reward. Two things make the
number honest:

* **Held-out questions.** Training used the first N of the validation split, so
  evaluation starts after them. Re-scoring training questions would mostly
  measure memorisation.
* **The same environment.** The reward, the BM25 index and the episode protocol
  are imported, not reimplemented, so a change to any of them cannot quietly
  make the eval disagree with the thing it is evaluating.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nanobeard.rl import rewards as R
from nanobeard.rl.env import STOP_STRINGS, SYSTEM, run_episode

OUT_DIR = Path("runs/evals")


def make_generate(base_url: str, max_tokens: int, temperature: float):
    def generate(body: str) -> str:
        payload = {
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": body}],
            "max_tokens": max_tokens, "temperature": temperature,
            "stop": list(STOP_STRINGS),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
        except (urllib.error.URLError, TimeoutError, OSError):
            return ""
        out = d["choices"][0]["message"].get("content") or ""
        # llama-server strips the stop string; put it back so the parser sees a
        # closed tag rather than treating a good action as malformed.
        if d["choices"][0].get("finish_reason") == "stop":
            for s in STOP_STRINGS:
                tag = s[2:-1]
                if f"<{tag}>" in out and s not in out:
                    out += s
        return out

    return generate


def run(base_url: str, index, questions, *, workers: int = 6, max_tokens: int = 160,
        temperature: float = 0.3, max_searches: int = 2) -> dict:
    generate = make_generate(base_url, max_tokens, temperature)

    def one(q):
        ep = run_episode(q, index, generate, max_searches=max_searches)
        rw = R.compute(ep.answer, q.answer, did_search=ep.did_search,
                       retrieved_titles=ep.retrieved_titles, gold_titles=q.gold_titles)
        return {"em": rw.answer_em, "format": rw.format_ok, "searched": rw.searched,
                "recall": rw.breakdown["retrieval_recall"],
                "f1": rw.breakdown["token_f1"], "n_searches": ep.n_searches}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, questions))
    n = len(rows) or 1
    return {
        "n": len(rows),
        "exact_match": sum(r["em"] for r in rows) / n,
        "token_f1": sum(r["f1"] for r in rows) / n,
        "gave_answer": sum(r["format"] for r in rows) / n,
        "searched": sum(r["searched"] for r in rows) / n,
        "retrieval_recall": sum(r["recall"] for r in rows) / n,
        "searches_per_episode": sum(r["n_searches"] for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="GGUF to serve")
    ap.add_argument("--server", help="Already-running llama-server URL")
    ap.add_argument("--label", required=True)
    ap.add_argument("--index", default="data/search/hotpot_bm25.pkl")
    ap.add_argument("--skip", type=int, default=2000,
                    help="Questions used for training; evaluation starts after them")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Match the training temperature. Sampling colder than the "
                         "policy was trained at changes its behaviour, not just its "
                         "variance: at 0.3 it stopped searching on 43% of questions.")
    ap.add_argument("--llama-port", type=int, default=8941)
    ap.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()

    from nanobeard.chat.server import LlamaServer
    from nanobeard.rl.corpus import load as load_corpus

    index, questions = load_corpus(Path(args.index))
    held_out = questions[args.skip:args.skip + args.n]
    if not held_out:
        raise SystemExit(
            f"no held-out questions: index has {len(questions)}, --skip is {args.skip}")
    print(f"{args.label}: {len(held_out)} held-out questions "
          f"(index {len(index):,} paragraphs, skipping the first {args.skip})")

    llama = None
    try:
        if args.server:
            base = args.server
        else:
            llama = LlamaServer(port=args.llama_port, ctx=args.ctx, slots=args.workers)
            print(f"loading {args.model} …")
            llama.start(args.model)
            base = llama.url
        res = run(base, index, held_out, workers=args.workers,
                  temperature=args.temperature)
    finally:
        if llama is not None:
            llama.stop()

    res.update({"label": args.label, "model": args.model or args.server, "skip": args.skip})
    print()
    for k in ("exact_match", "token_f1", "gave_answer", "searched",
              "retrieval_recall", "searches_per_episode"):
        v = res[k]
        print(f"  {k:<22}{v:>7.1%}" if v <= 1 and k != "searches_per_episode"
              else f"  {k:<22}{v:>7.2f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"search-{args.label}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
