"""Convert public function-calling datasets into pirate-persona training rows.

Why this exists: LoRA v2 could call a tool correctly *when it called at all* —
4/4 right — but only called on 4 of 12 prompts. The cause was diversity, not
volume. Our teacher-generated set was 1,534 examples built from **8 distinct
tools and 16 distinct user sentences**, each repeated ~96 times. The model
memorised those sentences instead of learning "a tool fits here", so eval
scenarios using different tools got no call.

Hermes' function-calling sets give **1,024 distinct tools across 1,090 queries**
in the same `<tool_call>{"name":..,"arguments":{..}}` shape Qwen3 emits. Nothing
needs a teacher: the assistant turn is JSON, which has no accent, so we keep
their tools/query/call verbatim and simply put *our* pirate system prompt in
front. That is exactly the pairing the model is failing at — persona present,
tool still called — and it costs nothing.

ToolACE is deliberately not used: its calls are `[Some Tool(a=1)]` rather than
JSON, and tool names contain spaces, so converting it reliably is more risk than
the extra rows are worth.
"""

from __future__ import annotations

import json
import re

CONFIGS = ("func_calling_singleturn", "func_calling", "glaive_func_calling")
DATASET = "NousResearch/hermes-function-calling-v1"
_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _first_user(conversations) -> str | None:
    for c in conversations:
        if c.get("from") in ("human", "user"):
            return (c.get("value") or "").strip() or None
    return None


def _first_call(conversations) -> dict | None:
    """The first well-formed tool call in the assistant's turns."""
    for c in conversations:
        if c.get("from") not in ("gpt", "assistant"):
            continue
        m = _CALL.search(c.get("value") or "")
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            return obj
    return None


def _tools(raw) -> list[dict] | None:
    """Normalise the tools column to OpenAI function format."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for t in raw:
        if not isinstance(t, dict):
            return None
        # Some rows are already {"type":"function","function":{...}}, others are bare.
        out.append(t if t.get("type") == "function" else {"type": "function", "function": t})
    return out


def convert_row(row: dict, system: str) -> dict | None:
    tools = _tools(row.get("tools"))
    user = _first_user(row.get("conversations") or [])
    call = _first_call(row.get("conversations") or [])
    if not (tools and user and call):
        return None
    # The call must name a tool that was actually offered, or the example
    # teaches the model to invent tools.
    offered = {t["function"].get("name") for t in tools}
    if call["name"] not in offered:
        return None
    return {
        "kind": "tool",
        "source": "hermes",
        "tools": tools,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments") or {}),
                }}]},
        ],
    }


def build(system: str, configs=CONFIGS, limit: int | None = None, token: str | None = None):
    """Rows ready to concatenate with the teacher-generated set."""
    from datasets import load_dataset

    seen_queries: set[str] = set()
    out: list[dict] = []
    for cfg in configs:
        ds = load_dataset(DATASET, cfg, split="train", token=token)
        for row in ds:
            r = convert_row(row, system)
            if r is None:
                continue
            q = r["messages"][1]["content"]
            # Deduplicate across configs: glaive and func_calling overlap, and a
            # repeated query is the exact failure mode being fixed.
            if q in seen_queries:
                continue
            seen_queries.add(q)
            out.append(r)
            if limit and len(out) >= limit:
                return out
    return out


def main() -> None:
    import argparse
    import os
    from pathlib import Path

    from nanobeard.distill.generate import VOICE
    from nanobeard.env import load_env

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/distill/tools_public.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_env()
    rows = build(VOICE, limit=args.limit, token=os.getenv("HF_TOKEN"))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    names = {r["messages"][2]["tool_calls"][0]["function"]["name"] for r in rows}
    queries = {r["messages"][1]["content"] for r in rows}
    print(f"wrote {len(rows)} rows to {p}")
    print(f"  distinct tools:   {len(names)}")
    print(f"  distinct queries: {len(queries)}")


if __name__ == "__main__":
    main()
