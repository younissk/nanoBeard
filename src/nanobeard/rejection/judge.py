"""Ask a cheap hosted model to pick its favourite among the candidate replies.

Two ways to run it:

    # 1. Live, for the viewer's "ask qwen" button:
    uv run python -m nanobeard.rejection.judge --serve

    # 2. Batch, bake every verdict into the HTML up front:
    uv run python -m nanobeard.rejection.judge --all \
        --in runs/rejection/frigate-360m-q8.jsonl

WHY A SERVER AND NOT A fetch() STRAIGHT FROM THE PAGE
-----------------------------------------------------
The viewer is a single self-contained HTML file opened over file://. Calling
OpenRouter from it would mean writing OPENROUTER_API_KEY into that file — a
live credential sitting in a 400 KB artifact that is easy to share, copy into
a bug report, or drop in a public dir. So the key stays here, in a process
bound to 127.0.0.1, and the page only ever talks to localhost.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from nanobeard.env import require
from nanobeard.rejection.generate import load_prompts  # noqa: F401  (re-exported for CLI use)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
# $0.048/$0.193 per M tokens — ~$0.005 to judge all 172 prompts.
# NOT the nominally-cheaper qwen3.7-flash: that one is a *reasoning* model, and
# it spends the whole token budget on hidden reasoning, returning
# content=None with finish_reason=length. Use an -instruct model here.
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

SYSTEM = """You are grading replies from a tiny 358M-parameter pirate chatbot.
It is a toy model: expect broken grammar, repetition and invented facts.

Pick the ONE reply that is the best response to the user's last message.
Judge in this order:
  1. Does it actually respond to what was asked?
  2. Is it factually acceptable? A confident wrong answer is WORSE than a vague one.
  3. Is it coherent — not looping the same phrase over and over?
  4. Does it keep a pirate voice?

Do not reward length. Do not reward pirate slang stuffed into a non-answer.
If every reply is bad, still pick the least bad one and say so.

Reply with JSON only: {"pick": <1-based index>, "reason": "<one short sentence>"}"""


def build_user_msg(turns: list[dict], candidates: list[str]) -> str:
    convo = "\n".join(
        f"{'User' if t['role'] == 'user' else 'Pirate'}: {t['text']}" for t in turns
    )
    numbered = "\n".join(
        f"{i}. {c.strip() or '(empty)'}" for i, c in enumerate(candidates, 1)
    )
    return f"Conversation so far:\n{convo}\n\nCandidate replies:\n{numbered}"


def _parse_verdict(text: str, n: int) -> dict:
    """Models wander outside the JSON fence; recover rather than crash a batch."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"pick": None, "reason": "judge returned unparseable output"}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"pick": None, "reason": "judge returned unparseable output"}
    try:
        pick = int(data.get("pick"))
    except (TypeError, ValueError):
        return {"pick": None, "reason": str(data.get("reason", ""))[:300]}
    if not 1 <= pick <= n:
        return {"pick": None, "reason": f"judge picked out-of-range index {pick}"}
    return {"pick": pick, "reason": str(data.get("reason", ""))[:300]}


def _judge_once(turns: list[dict], candidates: list[str], *, api_key: str, model: str,
                timeout: float, retries: int, order: list[int]) -> dict:
    """One call. `order` maps presented slot -> original index."""
    shown = [candidates[i] for i in order]
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_msg(turns, shown)},
        ],
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/younissk/pirate_llm",
            "X-Title": "nanoBeard rejection sampling",
        },
    )
    body = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.load(r)
            break
        except urllib.error.HTTPError as e:
            # Upstream providers throttle per-model; back off and retry.
            # Never echo the request headers — they carry the key.
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return {"pick": None, "reason": f"OpenRouter HTTP {e.code}", "error": True}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return {"pick": None, "reason": type(e).__name__, "error": True}

    if body is None:
        # Every retry raised and the loop fell through without returning.
        return {"pick": None, "reason": "no response from OpenRouter", "error": True}

    choice = body["choices"][0]
    content = choice["message"].get("content")
    if not content:
        why = ("model spent its budget on reasoning and returned no content — "
               "use an -instruct model" if choice.get("finish_reason") == "length"
               else "model returned empty content")
        return {"pick": None, "reason": why, "error": True, "model": model}

    out = _parse_verdict(content, len(shown))
    if out["pick"] is not None:
        out["pick"] = order[out["pick"] - 1] + 1   # presented slot -> original index
    out["model"] = model
    return out


def judge(turns: list[dict], candidates: list[str], *, api_key: str,
          model: str = DEFAULT_MODEL, timeout: float = 90.0, retries: int = 3,
          votes: int = 1, seed: int | None = None) -> dict:
    """Judge with the candidate order shuffled per call.

    Measured on the 10-sample run: presenting candidates in fixed order gave a
    significant primacy effect (chi-square 22.5, df=9 -> p<0.01; slot 1 won 31
    times against ~17 expected). Slot index carries no quality signal here --
    it is just the generation seed -- so a fixed order turns that bias into a
    systematic preference for particular seeds. Shuffling per call converts it
    into noise. With votes>1 we shuffle differently each round and take the
    majority, which suppresses it further.
    """
    rng = random.Random(seed)
    n = len(candidates)
    results = []
    for _ in range(max(1, votes)):
        order = list(range(n))
        rng.shuffle(order)
        r = _judge_once(turns, candidates, api_key=api_key, model=model,
                        timeout=timeout, retries=retries, order=order)
        results.append(r)
        if r.get("error"):
            break

    picks = [r["pick"] for r in results if r.get("pick") is not None]
    if not picks:
        return results[0]

    winner, count = collections.Counter(picks).most_common(1)[0]
    best = next(r for r in results if r["pick"] == winner)
    out = dict(best)
    out["pick"] = winner
    if len(results) > 1:
        out["votes"] = f"{count}/{len(results)}"
        out["all_picks"] = picks
    return out


# ---------------------------------------------------------------- serve mode
def make_handler(api_key: str, model: str, votes: int = 1):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, obj: dict) -> None:
            blob = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            # The page is opened over file://, whose Origin is literally "null".
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(blob)

        def do_OPTIONS(self):  # noqa: N802
            self._send(204, {})

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "model": model, "votes": votes})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/judge":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_000_000:
                return self._send(413, {"error": "payload too large"})
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                turns, cands = req["turns"], req["candidates"]
            except (json.JSONDecodeError, KeyError):
                return self._send(400, {"error": "expected {turns, candidates}"})
            self._send(200, judge(turns, cands, api_key=api_key, model=model, votes=votes))

        def log_message(self, fmt, *args):
            print(f"  judge {fmt % args}")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="Run the local judge endpoint")
    ap.add_argument("--all", action="store_true", help="Batch-judge every prompt")
    ap.add_argument("--in", dest="inp", help="Generation JSONL (for --all)")
    ap.add_argument("--out", default=None, help="Verdict JSON (default: <in>.judged.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="Judge only the first N prompts")
    ap.add_argument("--votes", type=int, default=1,
                    help="Judge each prompt N times with different candidate orders, take the majority")
    args = ap.parse_args()

    api_key = require("OPENROUTER_API_KEY")

    if args.serve:
        srv = HTTPServer(("127.0.0.1", args.port), make_handler(api_key, args.model, args.votes))
        print(f"judge ready on http://127.0.0.1:{args.port}  model={args.model}")
        print("the viewer's 'ask qwen' button talks to this. ctrl-c to stop.")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        return

    if not args.all or not args.inp:
        raise SystemExit("use --serve, or --all --in <generations.jsonl>")

    from nanobeard.rejection.viewer import build_payload, load

    inp = Path(args.inp)
    payload = build_payload(load(inp), inp.stem)
    prompts = payload["prompts"]
    if args.limit:
        prompts = prompts[: args.limit]
    out_path = Path(args.out) if args.out else inp.with_suffix(".judged.json")

    def run(p):
        v = judge(p["turns"], [s["text"] for s in p["samples"]],
                  api_key=api_key, model=args.model, votes=args.votes,
                  seed=hash(p["id"]) & 0xFFFF)   # deterministic shuffle per prompt
        return p["id"], v

    print(f"judging {len(prompts)} prompts with {args.model} "
          f"({args.votes} vote{'s' if args.votes > 1 else ''}, shuffled order) ...")
    verdicts, failures = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (pid, v) in enumerate(pool.map(run, prompts), 1):
            verdicts[pid] = v
            if v.get("pick") is None:
                failures += 1
            if i % 25 == 0:
                print(f"  {i}/{len(prompts)}")

    out_path.write_text(json.dumps(
        {"model": args.model, "votes": args.votes, "verdicts": verdicts}, indent=2))
    print(f"Wrote {len(verdicts)} verdicts -> {out_path}  ({failures} without a pick)")


if __name__ == "__main__":
    main()
