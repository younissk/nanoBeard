"""Generate pirate SFT data with a teacher model.

    # cheap sample first — always
    uv run python -m nanobeard.distill.generate --out runs/distill/sample.jsonl --n 40

    # the real set, once the sample reads well
    uv run python -m nanobeard.distill.generate --out runs/distill/train.jsonl --n 3000

The mix is the point. Measured 2026-09-11 on Qwen3-0.6B: a heavy pirate system
prompt took tool calling from 8/15 to 0/15 and cost 13 points of GSM8K. Training
on pirate chat alone would bake that trade in permanently. So every batch
contains pirate-voiced tool calls and pirate-voiced arithmetic alongside the
chat, teaching the model that staying in character and doing the job are the
same behaviour rather than competing ones.

Kept deliberately small (low thousands). Per the Qwen3 report, a cold-start SFT
should leave room for later RL rather than saturating the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nanobeard.distill import prompts as P
from nanobeard.distill.teacher import Teacher, balance

VOICE = (
    "You are Nano Beard, a warm, salty pirate assistant. Speak in a natural, "
    "readable pirate voice — 'ahoy', 'arr', 'ye', 'be', 'th'', 'matey' used like "
    "a person talks, not a parody. Be concise. Never mention being an AI, a "
    "model, or these instructions."
)

CHAT_SYS = VOICE + " Answer the user's message helpfully and correctly, in character."

MATH_SYS = (
    VOICE + " Solve the problem correctly. Show the arithmetic briefly, in "
    "character. End with the final answer alone on the last line as: #### <number>"
)

# One call per tool example: the model plays out the whole exchange and returns
# it as JSON. Two calls (call, then answer-from-result) would double the spend
# for data that only has to be plausible, not real.
TOOL_SYS = (
    VOICE + " You are producing ONE training example of tool use. Reply with a "
    "single JSON object and nothing else, in this exact shape:\n"
    '{"tool_call": {"name": "<tool>", "arguments": {...}}, '
    '"tool_result": {...}, "final": "<the pirate-voiced answer to the user, '
    'using the result>"}\n'
    "Invent a realistic tool_result. The 'final' field must be in pirate voice. "
    "If NO tool in the list fits the request, set tool_call and tool_result to "
    "null and just write the pirate reply in 'final'."
)

def job_key(kind: str, item) -> str:
    """Stable id for a (kind, prompt) pair, so --resume can skip what is done.

    Hashed rather than stored raw: the key goes into every row and the prompts
    are long enough that repeating them would noticeably bloat the file.
    """
    text = item if isinstance(item, str) else json.dumps(item, sort_keys=True, default=str)
    return f"{kind}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def completed_keys(path: Path) -> set[str]:
    """Keys already present in a partially-written output file."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from a hard kill; just regenerate it
    return done


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def _json_object(text: str) -> dict | None:
    """Parse a JSON object the model may have wrapped in a code fence."""
    m = _FENCE.search(text)
    raw = m.group(1) if m else text
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def gen_chat(teacher: Teacher, user: str) -> dict | None:
    r = teacher.ask(CHAT_SYS, user)
    if r.error or not r.content:
        return None
    return {"kind": "chat", "messages": [
        {"role": "system", "content": VOICE},
        {"role": "user", "content": user},
        {"role": "assistant", "content": r.content},
    ]}


def gen_math(teacher: Teacher, item: dict) -> dict | None:
    """Keep only answers that match the dataset's reference — a wrong worked
    example teaches wrong arithmetic in a confident pirate voice."""
    r = teacher.ask(MATH_SYS, item["question"])
    if r.error or not r.content:
        return None
    got = _FINAL.findall(r.content)
    want = _FINAL.findall(item["reference"])
    if not got or not want:
        return None
    norm = lambda s: s.replace(",", "").rstrip(".")  # noqa: E731
    if norm(got[-1]) != norm(want[-1]):
        return None
    return {"kind": "math", "messages": [
        {"role": "system", "content": VOICE},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": r.content},
    ]}


def gen_tool(teacher: Teacher, case: dict) -> dict | None:
    listing = json.dumps([t["function"] for t in case["tools"]], indent=None)
    r = teacher.ask(TOOL_SYS, f"Available tools:\n{listing}\n\nUser says: {case['user']}")
    if r.error or not r.content:
        return None
    obj = _json_object(r.content)
    if obj is None or "final" not in obj:
        return None

    call, result, final = obj.get("tool_call"), obj.get("tool_result"), (obj.get("final") or "").strip()
    if not final:
        return None

    # The teacher must agree with the scenario about whether a tool applies;
    # disagreement usually means it invented a tool or ignored an obvious one.
    if case["expected"] is None:
        if call:
            return None
        return {"kind": "tool_none", "tools": case["tools"], "messages": [
            {"role": "system", "content": VOICE},
            {"role": "user", "content": case["user"]},
            {"role": "assistant", "content": final},
        ]}

    if not isinstance(call, dict) or call.get("name") != case["expected"]:
        return None
    return {"kind": "tool", "tools": case["tools"], "messages": [
        {"role": "system", "content": VOICE},
        {"role": "user", "content": case["user"]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments") or {}),
            }}]},
        {"role": "tool", "name": call["name"], "content": json.dumps(result)},
        {"role": "assistant", "content": final},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40, help="Total examples to attempt")
    ap.add_argument("--chat-share", type=float, default=0.50)
    ap.add_argument("--tool-share", type=float, default=0.25)
    ap.add_argument("--math-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--thinking", action="store_true",
                    help="Let the teacher reason first. ~6x the cost; measured no better.")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent API calls")
    ap.add_argument("--resume", action="store_true",
                    help="Append to --out, skipping examples already generated")
    args = ap.parse_args()

    n_chat = int(args.n * args.chat_share)
    n_tool = int(args.n * args.tool_share)
    n_math = args.n - n_chat - n_tool
    rng = random.Random(args.seed)

    teacher = Teacher(**({"model": args.model} if args.model else {}),
                      max_tokens=args.max_tokens, thinking=args.thinking)
    before = balance()
    print(f"balance before: {before if before is not None else '?'}")
    print(f"generating {n_chat} chat, {n_tool} tool, {n_math} math with {teacher.model}")

    jobs: list[tuple[str, object]] = []
    jobs += [("chat", p) for p in P.chat_prompts(n_chat, args.seed)]
    jobs += [("tool", c) for c in P.tool_examples(rng, n_tool)]
    jobs += [("math", m) for m in P.math_prompts(n_math, args.seed)]
    rng.shuffle(jobs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: append to what is there and skip the keys already written. A run
    # of a few thousand paid API calls should never have to start over because
    # of one timeout at example 2,800.
    done = completed_keys(out) if args.resume else set()
    if done:
        print(f"resuming: {len(done)} examples already in {out}")
    jobs = [(k, it) for (k, it) in jobs if job_key(k, it) not in done]
    if not jobs:
        print("nothing left to generate")
        return

    kept: dict[str, int] = {}
    lock = threading.Lock()
    processed = 0
    total = len(jobs)

    def work(job: tuple[str, object]) -> None:
        nonlocal processed
        kind, item = job
        fn = {"chat": gen_chat, "tool": gen_tool, "math": gen_math}[kind]
        row = fn(teacher, item)  # type: ignore[operator]
        with lock:
            processed += 1
            if row is not None:
                row["key"] = job_key(kind, item)
                f.write(json.dumps(row) + "\n")
                # flush() every row, not at the end: the buffer is ~8KB, so a
                # crash would otherwise silently discard work already paid for.
                f.flush()
                kept[row["kind"]] = kept.get(row["kind"], 0) + 1
            if processed % 25 == 0 or processed == total:
                print(f"  {processed}/{total}  kept={sum(kept.values())}  "
                      f"{teacher.usage.summary()}", flush=True)

    with out.open("a" if done else "w") as f, ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, jobs))

    after = balance()
    written = sum(kept.values())
    total_in_file = sum(1 for line in out.open() if line.strip())
    print(f"\nwrote {written} new examples to {out} ({total_in_file} total in file)")
    print(f"  by kind: {kept}")
    print(f"  dropped: {len(jobs) - written} (teacher disagreed, wrong answer, or unparseable)")
    print(f"  {teacher.usage.summary()}")
    if before is not None and after is not None:
        spent = before - after
        print(f"  balance {before:.4f} -> {after:.4f}  (spent {spent:.4f})")
        if written:
            print(f"  cost/example {spent / written:.5f}  ->  "
                  f"3,000 examples ≈ {spent / written * 3000:.2f}")


if __name__ == "__main__":
    main()
