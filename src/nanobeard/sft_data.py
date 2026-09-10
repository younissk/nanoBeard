"""SFT data for the pirate *chatbot*.

Format is a minimal plaintext chat transcript — no markdown, no special
tokens (the BPE vocab is frozen):

    User: <plain english>
    Pirate: <pirate reply><eos>
    User: <next message>
    Pirate: <reply><eos>

The user always speaks plain English; the bot ("Pirate") always replies in
pirate-speak. Loss is masked on everything except the bot reply spans (+eos),
so the model learns *when and how* to answer, never to parrot the user.

Sources (both English in, pirate out):
  - TeeZee/dolly-15k-pirate-speech  -> single-turn instruction following.
    Responses are re-piratized through `arrr` for a consistent voice (the
    upstream translations are uneven, e.g. answers like "Tope").
  - Estwld/empathetic_dialogues_llm -> multi-turn chit-chat. This is what
    teaches turn-taking and short-range memory; the assistant turns are
    piratized, the user turns left as plain English.

Set SFT_LIMIT=<n> to cap examples per source for a fast local smoke run.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from arrr import translate
from datasets import load_dataset
from tokenizers import Tokenizer

from nanobeard.config import Config
from nanobeard.env import load_env

load_env()

USER_PREFIX = "User: "
BOT_PREFIX = "Pirate: "
TURN_SEP = "\n"
IGNORE_INDEX = -100


@dataclass
class Turn:
    role: str  # "user" | "bot"
    text: str


# A conversation is an ordered list of Turns, starting (usually) with a user turn.
Conversation = list[Turn]


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]
    truncated: bool = False


def _rows(raw: Any) -> Iterable[dict[str, Any]]:
    """Iterate a HF split as plain dicts.

    `load_dataset(..., split=...)` is typed as a union wide enough that a type
    checker cannot tell a row from a column, so narrow it once here rather than
    casting at every field access.
    """
    return raw


# --------------------------------------------------------------------------
# Source loaders -> Conversations (bot text still plain English here;
# piratization happens once, in batch, afterwards).
# --------------------------------------------------------------------------
def _dolly_conversations(limit: int | None) -> list[Conversation]:
    raw = load_dataset("TeeZee/dolly-15k-pirate-speech", split="train")
    convs: list[Conversation] = []
    for row in _rows(raw):
        instr = (row["instruction"] or "").strip()
        ctx = (row.get("context") or "").strip()
        resp = (row["response"] or "").strip()
        if not instr or not resp:
            continue
        user_text = f"{instr}\n{ctx}" if ctx else instr
        convs.append([Turn("user", user_text), Turn("bot", resp)])
        if limit and len(convs) >= limit:
            break
    return convs


def _empathetic_conversations(limit: int | None) -> list[Conversation]:
    raw = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
    convs: list[Conversation] = []
    for row in _rows(raw):
        turns: Conversation = []
        for msg in row["conversations"]:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            role = "bot" if msg.get("role") == "assistant" else "user"
            turns.append(Turn(role, content))
        # Need at least one user->bot exchange to be useful.
        if any(t.role == "bot" for t in turns) and len(turns) >= 2:
            convs.append(turns)
        if limit and len(convs) >= limit:
            break
    return convs


def _piratize_bot_turns(convs: list[Conversation]) -> None:
    """Translate every bot turn to pirate-speak in place.

    `arrr.translate` is a deterministic local regex translator — ~65k short
    strings finish in a few seconds, so this runs single-process on purpose.
    Do NOT reintroduce a multiprocessing Pool: vast.ai containers report the
    *host* core count via cpu_count(), so Pool(cpu_count()-1) forks ~32 workers
    and exhausts the container's process limit ("fork: Resource temporarily
    unavailable") before training even starts.
    """
    for conv in convs:
        for turn in conv:
            if turn.role == "bot":
                turn.text = translate(turn.text)


# --------------------------------------------------------------------------
# Encoding: conversation -> (input_ids, labels) with per-turn masking.
# --------------------------------------------------------------------------
def _enc(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def encode_conversation(
    conv: Conversation,
    tokenizer: Tokenizer,
    block_size: int,
    eos_id: int,
) -> SFTExample | None:
    """Render a conversation to ids/labels.

    User turns and the "Pirate: " cue are masked (IGNORE_INDEX); only the bot
    reply tokens + the turn-ending eos carry loss. Over-long conversations are
    LEFT-truncated (keep the most recent turns) — this mirrors the rolling
    memory window used at inference. Returns None if nothing is left to learn.
    """
    ids: list[int] = []
    labels: list[int] = []
    for i, turn in enumerate(conv):
        sep = "" if i == 0 else TURN_SEP
        if turn.role == "user":
            seg = _enc(tokenizer, sep + USER_PREFIX + turn.text)
            ids += seg
            labels += [IGNORE_INDEX] * len(seg)
        else:  # bot
            lead = _enc(tokenizer, sep + BOT_PREFIX)
            body = _enc(tokenizer, turn.text) + [eos_id]
            ids += lead + body
            labels += [IGNORE_INDEX] * len(lead) + body

    truncated = False
    if len(ids) > block_size:
        ids = ids[-block_size:]
        labels = labels[-block_size:]
        truncated = True

    # After truncation a window may hold no trainable token (e.g. it landed
    # entirely inside one long user turn) — drop it.
    if all(lbl == IGNORE_INDEX for lbl in labels):
        return None

    pad_len = block_size - len(ids)
    if pad_len > 0:
        ids = ids + [eos_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len

    return SFTExample(input_ids=ids, labels=labels, truncated=truncated)


def build_sft_dataset(
    config: Config,
    val_fraction: float = 0.02,
    seed: int = 1337,
) -> tuple[list[SFTExample], list[SFTExample], Tokenizer]:
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None, "Tokenizer must have <|endoftext|>"

    limit_env = os.environ.get("SFT_LIMIT")
    limit = int(limit_env) if limit_env else None

    print("Loading SFT sources...")
    convs = _dolly_conversations(limit) + _empathetic_conversations(limit)
    print(f"  {len(convs)} conversations; piratizing bot turns...")
    _piratize_bot_turns(convs)

    examples: list[SFTExample] = []
    skipped = 0
    truncated = 0
    for conv in convs:
        ex = encode_conversation(conv, tokenizer, config.block_size, eos_id)
        if ex is None:
            skipped += 1
            continue
        truncated += int(ex.truncated)
        examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    val = examples[:n_val]
    train = examples[n_val:]

    print(
        f"SFT dataset: {len(train)} train / {len(val)} val | "
        f"skipped {skipped} (no trainable tokens) | "
        f"left-truncated {truncated} ({truncated / max(1, len(examples)):.1%}) "
        f"@ block_size={config.block_size}"
    )
    return train, val, tokenizer


def get_sft_batch(
    split_examples: list[SFTExample],
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = torch.randint(0, len(split_examples), (config.batch_size,))
    rows = [split_examples[i] for i in idxs.tolist()]

    input_ids = torch.tensor([r.input_ids for r in rows], dtype=torch.long)
    labels = torch.tensor([r.labels for r in rows], dtype=torch.long)

    x = input_ids[:, :-1].contiguous()
    y = labels[:, 1:].contiguous()

    if config.device == "cuda":
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)
    return x, y


# --------------------------------------------------------------------------
# Inference: render a live chat history to ids ending in the bot cue.
# Mirrors the training layout exactly so the model sees a familiar context.
# --------------------------------------------------------------------------
def build_chat_prompt_ids(
    history: Conversation,
    tokenizer: Tokenizer,
    eos_id: int,
    block_size: int,
    reserve: int,
) -> list[int]:
    """Encode completed `history` turns + a trailing "Pirate: " cue.

    Drops the oldest turns until the prompt fits in `block_size - reserve`
    tokens — the rolling memory window. `reserve` is the room left for the
    reply being generated.
    """
    budget = max(1, block_size - reserve)
    hist = list(history)
    while True:
        ids: list[int] = []
        for i, turn in enumerate(hist):
            sep = "" if i == 0 else TURN_SEP
            if turn.role == "user":
                ids += _enc(tokenizer, sep + USER_PREFIX + turn.text)
            else:
                # Split the encode at the prefix boundary EXACTLY as training
                # does (see encode_conversation) — a joint encode of
                # "Pirate: <text>" merges BPE across the boundary and attaches a
                # leading space to the first reply word, producing token ids the
                # model never saw in training and degrading recall of prior turns.
                ids += _enc(tokenizer, sep + BOT_PREFIX) + _enc(tokenizer, turn.text) + [eos_id]
        # Trailing cue the model completes.
        cue_sep = "" if not hist else TURN_SEP
        ids += _enc(tokenizer, cue_sep + BOT_PREFIX)
        if len(ids) <= budget or len(hist) <= 1:
            return ids[-budget:]
        hist.pop(0)  # forget the oldest turn and retry
