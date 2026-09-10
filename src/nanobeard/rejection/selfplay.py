"""Self-play: Qwen drives a conversation, nanoBeard answers by rejection sampling.

Per turn:
  1. nanoBeard generates N candidate replies (llama-server, one seed each).
  2. Qwen picks its favourite  (temperature 0, candidate order shuffled).
  3. Qwen writes the next user message, in character (high temperature).
Stops when Qwen says the conversation has run its course, but never before
MIN_TURNS and never after MAX_TURNS.

    uv run python -m nanobeard.rejection.selfplay \
        --out runs/selfplay/frigate-360m.jsonl --conversations 100 --n 20

Two Qwen calls per turn rather than one combined call: picking wants
temperature 0, speaking wants ~1.0, and a single call cannot have both. The
speaking call sees the chosen reply in the transcript, so nothing is lost.

Context: each llama-server slot gets 512 tokens (the model's trained length),
so the prompt is trimmed from the oldest turns each round — a 15-turn
transcript does not fit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from nanobeard.env import require
from nanobeard.rejection.generate import complete, render_prompt
from nanobeard.rejection.judge import DEFAULT_MODEL, judge
from nanobeard.rejection.personas import SITUATIONS

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MIN_TURNS, MAX_TURNS = 5, 15
# Leave room for the reply inside the slot's 512-token window.
PROMPT_TOKEN_BUDGET = 360

SPEAKER_SYSTEM = """You are role-playing a HUMAN chatting with a pirate-themed chatbot.

Situation: {situation}

Rules:
- Write ONLY your next message. No narration, no quotes, no name prefix.
- Plain everyday English. You are NOT a pirate. Never use pirate slang.
- Keep it short — one or two sentences, like a real chat message.
- React to what the pirate actually just said. Follow up on it, push back on
  it, or move the conversation somewhere new. Do not ignore it.
- The pirate is a very small model: it rambles, repeats itself and invents
  facts. Do not correct it or comment on it being an AI. Stay in the chat.
- Vary yourself. Do not open every message the same way.

{wind}
When the conversation has genuinely run its course, reply with exactly: <END>"""

OPENER_SYSTEM = """You are role-playing a HUMAN starting a chat with a pirate-themed chatbot.

Situation: {situation}

Write ONLY the opening message — one or two sentences, plain English, no
pirate slang, no narration, no quotes. Make it specific to the situation
rather than a generic greeting."""


def _post(payload: dict, api_key: str, timeout: float = 90.0, retries: int = 3) -> dict | None:
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "HTTP-Referer": "https://github.com/younissk/pirate_llm",
                 "X-Title": "nanoBeard self-play"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None


def _clean(text: str) -> str:
    """Strip role prefixes / quotes the speaker model sometimes adds anyway."""
    t = (text or "").strip()
    t = re.sub(r"^(user|you|human)\s*:\s*", "", t, flags=re.I)
    if len(t) > 1 and t[0] in '"“' and t[-1] in '"”':
        t = t[1:-1].strip()
    return t


def speak(turns: list[dict], situation: str, *, api_key: str, model: str,
          temperature: float, opening: bool, wind: str = "") -> str | None:
    """Next human message, or None when the speaker calls the conversation done."""
    system = (OPENER_SYSTEM.format(situation=situation) if opening
              else SPEAKER_SYSTEM.format(situation=situation, wind=wind))
    msgs = [{"role": "system", "content": system}]
    if opening:
        msgs.append({"role": "user", "content": "Begin."})
    else:
        # The human's own lines are 'assistant' here: this model IS the human.
        for t in turns:
            msgs.append({"role": "assistant" if t["role"] == "user" else "user",
                         "content": t["text"]})
    body = _post({"model": model, "temperature": temperature, "top_p": 0.95,
                  "max_tokens": 120, "messages": msgs}, api_key)
    if not body:
        return None
    out = _clean(body["choices"][0]["message"].get("content") or "")
    if not out or out.upper().startswith("<END>"):
        return None
    return out


def fit_prompt(server: str, turns: list[dict], budget: int = PROMPT_TOKEN_BUDGET) -> str:
    """Render the transcript, dropping oldest turns until it fits the slot."""
    keep = list(turns)
    while keep:
        prompt = render_prompt(keep)
        try:
            req = urllib.request.Request(
                f"{server}/tokenize", data=json.dumps({"content": prompt}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                n = len(json.load(r)["tokens"])
        except (urllib.error.URLError, TimeoutError, OSError, KeyError):
            n = len(prompt) // 3          # conservative fallback
        if n <= budget or len(keep) <= 2:
            return prompt
        keep = keep[2:]                    # drop one user+bot exchange
    return render_prompt(turns[-2:])


def run_conversation(idx: int, *, api_key: str, server: str, judge_model: str,
                     speaker_model: str, speaker_temp: float, n: int,
                     gen_params: dict, rng: random.Random) -> dict | None:
    situation = rng.choice(SITUATIONS)
    # Draw a target length up front. Left to its own judgement the speaker
    # essentially never says <END>, so every conversation ran to the 15-turn
    # cap and the last few exchanges decayed into "yeah, I agree" filler.
    # A per-conversation target gives the dataset real length variety and
    # lets each one close while it is still going somewhere.
    target = rng.randint(MIN_TURNS, MAX_TURNS)
    turns: list[dict] = []
    records = []

    first = speak([], situation, api_key=api_key, model=speaker_model,
                  temperature=speaker_temp, opening=True)
    if not first:
        return None
    turns.append({"role": "user", "text": first})

    for turn_i in range(MAX_TURNS):
        prompt = fit_prompt(server, turns)
        cands, seeds = [], []
        with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
            futs = {pool.submit(complete, server, prompt, seed=rng.randrange(1 << 30),
                                **gen_params): k for k in range(n)}
            for f in as_completed(futs):
                try:
                    d = f.result()
                except Exception:
                    continue
                cands.append(d.get("content", ""))
                seeds.append(d.get("seed"))
        cands = [c for c in cands if c.strip()]
        if not cands:
            break

        v = judge(turns, cands, api_key=api_key, model=judge_model,
                  seed=rng.randrange(1 << 30))
        pick = v.get("pick")
        if pick is None:
            break
        reply = cands[pick - 1].strip()
        turns.append({"role": "bot", "text": reply})
        records.append({"turn": turn_i + 1, "n_candidates": len(cands),
                        "pick": pick, "reason": v.get("reason", ""),
                        "candidates": cands})

        exchanges = sum(1 for t in turns if t["role"] == "bot")
        if exchanges >= min(target, MAX_TURNS):
            break

        left = target - exchanges
        if left <= 1:
            wind = "This should be your LAST message. Close the conversation warmly and naturally."
        elif left <= 3:
            wind = "The conversation is nearing its end. Start steering it toward a natural close."
        else:
            wind = "There is plenty of conversation left. Keep it moving and open up something new."

        nxt = speak(turns, situation, api_key=api_key, model=speaker_model,
                    temperature=speaker_temp, opening=False, wind=wind)
        if nxt is None:
            if exchanges >= MIN_TURNS:
                break
            nxt = speak(turns, situation, api_key=api_key, model=speaker_model,
                        temperature=speaker_temp, opening=False,
                        wind="Keep the conversation going — do not end it yet.")
            if nxt is None:
                break
        turns.append({"role": "user", "text": nxt})

    exchanges = sum(1 for t in turns if t["role"] == "bot")
    if exchanges < MIN_TURNS:
        return None
    if turns and turns[-1]["role"] == "user":
        turns.pop()                        # never end on an unanswered human line
    return {"id": f"conv-{idx:04d}", "situation": situation,
            "target_exchanges": target, "turns": turns,
            "n_exchanges": sum(1 for t in turns if t["role"] == "bot"),
            "judge_model": judge_model, "speaker_model": speaker_model,
            "speaker_temperature": speaker_temp, "gen_params": gen_params,
            "turn_details": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conversations", type=int, default=100)
    ap.add_argument("--n", type=int, default=20, help="nanoBeard candidates per turn")
    ap.add_argument("--server", default="http://127.0.0.1:8899")
    ap.add_argument("--judge-model", default=DEFAULT_MODEL)
    ap.add_argument("--speaker-model", default=DEFAULT_MODEL)
    ap.add_argument("--speaker-temp", type=float, default=1.1)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--n-predict", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4, help="Conversations in flight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    api_key = require("OPENROUTER_API_KEY")
    gen_params = {"temperature": args.temperature, "top_p": args.top_p,
                  "top_k": args.top_k, "n_predict": args.n_predict}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have: set[str] = set()
    if args.resume and out.is_file():
        with out.open() as f:
            for line in f:
                if line.strip():
                    have.add(json.loads(line)["id"])
        print(f"resuming: {len(have)} conversations already done")

    todo = [i for i in range(args.conversations) if f"conv-{i:04d}" not in have]
    print(f"{len(todo)} conversations to run · {args.n} candidates/turn · "
          f"{MIN_TURNS}-{MAX_TURNS} exchanges · speaker temp {args.speaker_temp}")

    lock, t0 = Lock(), time.time()
    done = dropped = 0
    fh = out.open("a" if args.resume else "w")

    def work(i: int):
        return run_conversation(
            i, api_key=api_key, server=args.server, judge_model=args.judge_model,
            speaker_model=args.speaker_model, speaker_temp=args.speaker_temp,
            n=args.n, gen_params=gen_params, rng=random.Random(args.seed * 100000 + i))

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for conv in pool.map(work, todo):
                with lock:
                    if conv is None:
                        dropped += 1
                    else:
                        fh.write(json.dumps(conv, ensure_ascii=False) + "\n")
                        fh.flush()
                        done += 1
                    n = done + dropped
                    if n % 5 == 0:
                        el = time.time() - t0
                        print(f"  {n}/{len(todo)}  kept={done} dropped={dropped}  "
                              f"({el/max(n,1):.0f}s each, ~{(len(todo)-n)*el/max(n,1)/60:.0f}m left)")
    finally:
        fh.close()

    print(f"\nWrote {done} conversations -> {out}  ({dropped} dropped as too short)")


if __name__ == "__main__":
    main()
