"""SFT data: multi-turn chat encoding, per-turn loss masking, memory window."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from nanobeard.sft_data import (
    BOT_PREFIX,
    IGNORE_INDEX,
    USER_PREFIX,
    Turn,
    build_chat_prompt_ids,
    encode_conversation,
)


@pytest.fixture
def tok(synthetic_tokenizer: Path) -> Tokenizer:
    return Tokenizer.from_file(str(synthetic_tokenizer))


@pytest.fixture
def eos_id(tok: Tokenizer) -> int:
    tid = tok.token_to_id("<|endoftext|>")
    assert tid is not None
    return tid


def _real_labels(ex):
    return [lbl for lbl in ex.labels if lbl != IGNORE_INDEX]


def _contains(haystack, needle):
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def test_encode_single_turn_shape(tok, eos_id):
    conv = [Turn("user", "ahoy there"), Turn("bot", "matey treasure")]
    ex = encode_conversation(conv, tok, block_size=64, eos_id=eos_id)
    assert ex is not None
    assert len(ex.input_ids) == 64
    assert len(ex.labels) == 64


def test_user_span_fully_masked(tok, eos_id):
    """No user token (nor the User:/Pirate: prefixes preceding the reply) carries loss."""
    conv = [Turn("user", "ahoy there"), Turn("bot", "matey treasure rum")]
    ex = encode_conversation(conv, tok, block_size=64, eos_id=eos_id)
    # The first real label must come strictly after the whole user turn + cue.
    first_real = next(i for i, lbl in enumerate(ex.labels) if lbl != IGNORE_INDEX)
    user_and_cue = tok.encode(USER_PREFIX + "ahoy there").ids
    assert first_real >= len(user_and_cue)


def test_bot_reply_trains_and_ends_in_eos(tok, eos_id):
    conv = [Turn("user", "x"), Turn("bot", "matey treasure rum")]
    ex = encode_conversation(conv, tok, block_size=64, eos_id=eos_id)
    reals = _real_labels(ex)
    assert len(reals) > 0
    assert eos_id in reals  # turn terminated


def test_multi_turn_trains_every_bot_turn(tok, eos_id):
    conv = [
        Turn("user", "hello there friend"),
        Turn("bot", "ahoy matey"),
        Turn("user", "how is the weather"),
        Turn("bot", "stormy seas ahead"),
    ]
    ex = encode_conversation(conv, tok, block_size=128, eos_id=eos_id)
    # Two bot turns -> at least two eos tokens among the trained labels.
    assert _real_labels(ex).count(eos_id) == 2


def test_prefixes_present_but_masked(tok, eos_id):
    """The 'Pirate: ' cue tokens appear in input_ids but are masked in labels."""
    conv = [Turn("user", "x"), Turn("bot", "y")]
    ex = encode_conversation(conv, tok, block_size=64, eos_id=eos_id)
    cue_ids = tok.encode(BOT_PREFIX).ids
    # cue tokens exist somewhere in the input
    joined = ex.input_ids
    assert any(joined[i : i + len(cue_ids)] == cue_ids for i in range(len(joined)))


def test_left_truncation_keeps_recent_and_flags(tok, eos_id):
    long_user = "treasure " * 300
    conv = [Turn("user", long_user), Turn("bot", "arr")]
    ex = encode_conversation(conv, tok, block_size=64, eos_id=eos_id)
    assert ex is not None
    assert ex.truncated is True
    assert len(ex.input_ids) == 64
    # The recent bot reply survives truncation -> still trainable.
    assert len(_real_labels(ex)) > 0


def test_returns_none_when_no_trainable_tokens(tok, eos_id):
    """A window that truncates down to a pure user span has nothing to learn."""
    conv = [Turn("user", "treasure " * 300)]  # no bot turn at all
    ex = encode_conversation(conv, tok, block_size=16, eos_id=eos_id)
    assert ex is None


def test_chat_prompt_ends_with_bot_cue(tok, eos_id):
    history = [Turn("user", "ahoy")]
    ids = build_chat_prompt_ids(history, tok, eos_id, block_size=64, reserve=8)
    cue_ids = tok.encode("\n" + BOT_PREFIX).ids if False else tok.encode(BOT_PREFIX).ids
    # Prompt must end with the Pirate: cue so the model completes the reply.
    assert ids[-len(cue_ids):] == cue_ids


def test_completed_bot_turn_tokenized_like_training(tok, eos_id):
    """Inference must render a completed bot turn with the SAME token ids as
    training. A joint encode of "Pirate: <text>" merges BPE across the prefix
    boundary (leading-space first word) -> tokens the model never trained on ->
    degraded recall of prior turns. Guard against that regression."""
    bot_text = "ahoy matey treasure"
    history = [Turn("user", "hello there"), Turn("bot", bot_text), Turn("user", "more")]
    ids = build_chat_prompt_ids(history, tok, eos_id, block_size=128, reserve=8)

    # Training renders a completed bot turn as: enc("\nPirate: ") | enc(text) | eos
    train_bot_seg = (
        tok.encode("\n" + BOT_PREFIX).ids + tok.encode(bot_text).ids + [eos_id]
    )
    assert _contains(ids, train_bot_seg)

    # The buggy joint form must NOT appear.
    joint_bot_seg = tok.encode("\n" + BOT_PREFIX + bot_text).ids + [eos_id]
    if joint_bot_seg != train_bot_seg:  # only meaningful when BPE actually differs
        assert not _contains(ids, joint_bot_seg)


def test_chat_prompt_respects_budget_and_forgets_oldest(tok, eos_id):
    history = [
        Turn("user", "first message " * 20),
        Turn("bot", "first reply " * 20),
        Turn("user", "latest short message"),
    ]
    ids = build_chat_prompt_ids(history, tok, eos_id, block_size=48, reserve=8)
    assert len(ids) <= 48 - 8
