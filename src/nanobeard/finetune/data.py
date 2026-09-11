"""Turn teacher-generated JSONL into masked training tensors.

The masking is the part worth reading. We want loss on what the *assistant*
says and nothing else — training on the user turns teaches the model to write
the user's next message, and training on the `<tool_response>` turns teaches it
to hallucinate tool output instead of waiting for one.

TRL can do this via `assistant_only_loss`, but only when the chat template
carries `{% generation %}` markers. Qwen3's does not (checked 2026-09-11), so
the masking is built here.

The obvious implementation — render `messages[:i]` and `messages[:i+1]`, treat
the difference as turn i — **does not work on Qwen3**, and fails silently. Two
reasons, both found by inspecting decoded spans rather than by reading the
template:

1. Token counts from separately-rendered strings do not index into the full
   render. `<|im_end|>\n` tokenizes differently at the end of a string than when
   followed by `<|im_start|>`, so spans slid forward by a token and supervised
   the opening of the *next* turn. On a tool example that trained the model to
   write the `<tool_response>` itself instead of waiting for one.
2. The template is not append-only. Qwen3 keeps the `<think>` block only on the
   final assistant turn and strips it from earlier ones, so `messages[:i+1]`
   rendered alone is not a prefix of the full conversation. A strict prefix
   check therefore skipped every tool call — the exact turns this fine-tune
   exists to teach.

So spans are located in the full render itself: find each assistant header, take
everything up to and including the following end-of-turn marker. Both markers are
derived from the tokenizer rather than hardcoded, and the number of spans found
is checked against the number of assistant messages — a mismatch drops the
example instead of mislabelling it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IGNORE_INDEX = -100


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    kind: str

    @property
    def n_supervised(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render(tok, messages: list[dict], tools=None, add_generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def turn_markers(tok) -> tuple[str, str]:
    """(assistant header, end-of-turn) strings for this tokenizer's template.

    Derived by diffing a render with and without the generation prompt, so this
    works for any ChatML-ish template rather than only Qwen3. Any thinking block
    the generation prompt tacks on is trimmed: we want the role marker, not the
    model's opening move.
    """
    probe = [{"role": "user", "content": "x"}]
    base = _render(tok, probe)
    with_gen = _render(tok, probe, add_generation_prompt=True)
    header = with_gen[len(base):] if with_gen.startswith(base) else "<|im_start|>assistant\n"
    if "<think>" in header:
        header = header[: header.index("<think>")]
    end = tok.eos_token or "<|im_end|>"
    if end not in base:
        end = "<|im_end|>"
    return header, end


def assistant_spans(full_text: str, header: str, end: str) -> list[tuple[int, int]]:
    """Character spans of every assistant turn, header excluded, end marker kept.

    The end marker is supervised on purpose: stopping is a behaviour the model
    has to learn, and a model that never emits it runs to the token limit.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        h = full_text.find(header, pos)
        if h == -1:
            break
        start = h + len(header)
        stop = full_text.find(end, start)
        if stop == -1:
            break
        stop += len(end)
        spans.append((start, stop))
        pos = stop
    return spans


def build_example(tok, row: dict, max_len: int = 2048) -> Example | None:
    """Tokenize one conversation, supervising only the assistant turns."""
    messages = row["messages"]
    tools = row.get("tools")
    header, end = turn_markers(tok)

    full_text = _render(tok, messages, tools)
    spans = assistant_spans(full_text, header, end)
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    # A mismatch means the template did something this code does not model.
    # Dropping one example is cheap; mislabelling it is not.
    if len(spans) != n_assistant:
        return None

    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    labels = [IGNORE_INDEX] * len(input_ids)
    for lo, hi in spans:
        for t, (a, b) in enumerate(offsets):
            if b > a and a >= lo and b <= hi:
                labels[t] = input_ids[t]

    if not any(x != IGNORE_INDEX for x in labels):
        return None
    if len(input_ids) > max_len:
        return None  # truncating mid-answer teaches the model to stop mid-answer
    return Example(input_ids=input_ids, labels=labels, kind=row.get("kind", "?"))


def build_dataset(tok, rows: list[dict], max_len: int = 2048) -> list[Example]:
    out = []
    for row in rows:
        ex = build_example(tok, row, max_len)
        if ex is not None:
            out.append(ex)
    return out


def split(examples: list[Example], val_frac: float = 0.05, seed: int = 1337):
    """Deterministic split, stratified by kind so val covers every capability."""
    import random

    rng = random.Random(seed)
    by_kind: dict[str, list[Example]] = {}
    for ex in examples:
        by_kind.setdefault(ex.kind, []).append(ex)
    train, val = [], []
    for kind in sorted(by_kind):
        items = by_kind[kind][:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac)) if len(items) > 1 else 0
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train)
    return train, val


def describe(examples: list[Example]) -> str:
    by: dict[str, list[Example]] = {}
    for ex in examples:
        by.setdefault(ex.kind, []).append(ex)
    lines = [f"{'kind':<12}{'n':>6}{'mean tok':>10}{'supervised':>12}{'sup %':>8}"]
    lines.append("-" * 48)
    for kind in sorted(by):
        v = by[kind]
        toks = sum(len(e.input_ids) for e in v) / len(v)
        sup = sum(e.n_supervised for e in v) / len(v)
        lines.append(f"{kind:<12}{len(v):>6}{toks:>10.0f}{sup:>12.0f}{sup / toks:>7.0%}")
    tot_t = sum(len(e.input_ids) for e in examples)
    tot_s = sum(e.n_supervised for e in examples)
    lines.append(f"{'TOTAL':<12}{len(examples):>6}{tot_t / max(1, len(examples)):>10.0f}"
                 f"{tot_s / max(1, len(examples)):>12.0f}{tot_s / max(1, tot_t):>7.0%}")
    return "\n".join(lines)


class Collator:
    """Pad a batch to its longest member; padding is never supervised."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[Example]) -> dict:
        import torch

        width = max(len(f.input_ids) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = width - len(f.input_ids)
            input_ids.append(f.input_ids + [self.pad_token_id] * pad)
            labels.append(f.labels + [IGNORE_INDEX] * pad)
            attn.append([1] * len(f.input_ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
