"""Chat playground — the parts that can be wrong without llama.cpp present.

The subprocess lifecycle needs a real llama-server, so it is not covered here.
What is covered is everything that silently produces a *plausible but wrong*
prompt or model list, which is the failure mode that wastes an afternoon:
picking the f16 intermediate instead of the quant, trimming a chat down to
nothing, or dropping the trailing space after "Pirate:".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobeard.chat.server import (
    DEFAULT_CTX,
    REPLY_HEADROOM,
    build_payload,
    discover_models,
    trim_turns,
)
from nanobeard.rejection.generate import STOPS
from nanobeard.sft_data import BOT_PREFIX, TURN_SEP, USER_PREFIX


def _gguf(root: Path, rel: str, size: int = 1024) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)
    return p


# ----- discovery -----
def test_missing_root_is_empty_not_an_error(tmp_path):
    assert discover_models(tmp_path / "nope") == []


def test_f16_intermediates_are_skipped(tmp_path):
    _gguf(tmp_path, "frigate-360m/frigate-360M-f16.gguf")
    _gguf(tmp_path, "frigate-360m/frigate-360M.Q8_0.gguf")
    labels = [m["label"] for m in discover_models(tmp_path)]
    assert labels == ["frigate-360m/frigate-360M.Q8_0.gguf"]


def test_newest_model_is_first(tmp_path):
    old = _gguf(tmp_path, "a/old.gguf")
    new = _gguf(tmp_path, "b/new.gguf")
    import os

    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert [m["label"] for m in discover_models(tmp_path)] == ["b/new.gguf", "a/old.gguf"]


def test_label_is_relative_and_size_reported(tmp_path):
    _gguf(tmp_path, "frigate-360m/m.Q4_K_M.gguf", size=2_500_000)
    (m,) = discover_models(tmp_path)
    assert m["label"] == "frigate-360m/m.Q4_K_M.gguf"
    assert m["path"] == str(tmp_path / "frigate-360m" / "m.Q4_K_M.gguf")
    assert m["size_mb"] == pytest.approx(2.5, abs=0.05)


# ----- history trimming -----
def _chars(text: str) -> int:
    """Stand-in token counter: 1 'token' per character, monotone in length."""
    return len(text)


def test_short_history_is_left_alone():
    turns = [{"role": "user", "text": "ahoy"}]
    assert trim_turns(turns, budget=10_000, count=_chars) == turns


def test_oldest_turns_go_first():
    turns = [
        {"role": "user", "text": "A" * 100},
        {"role": "bot", "text": "B" * 100},
        {"role": "user", "text": "C" * 10},
    ]
    kept = trim_turns(turns, budget=60, count=_chars)
    assert [t["text"][0] for t in kept] == ["C"]


def test_the_newest_turn_is_never_dropped():
    # Even a single turn over budget must survive — an empty prompt is worse
    # than one the server truncates.
    turns = [{"role": "user", "text": "X" * 5000}]
    assert trim_turns(turns, budget=10, count=_chars) == turns


def test_trimming_stops_as_soon_as_it_fits():
    turns = [{"role": "user", "text": "u"}, {"role": "bot", "text": "b"},
             {"role": "user", "text": "hello"}]
    assert trim_turns(turns, budget=10_000, count=_chars) == turns


# ----- payload -----
def test_prompt_ends_at_the_bot_prefix_with_its_trailing_space():
    # Load-bearing: byte-level BPE makes "Pirate:" + " hi" a different token
    # sequence from "Pirate: " + "hi", and only the latter matches training.
    p = build_payload([{"role": "user", "text": "ahoy"}], {})
    assert p["prompt"].endswith(BOT_PREFIX)
    assert BOT_PREFIX.endswith(" ")
    assert p["prompt"] == USER_PREFIX + "ahoy" + TURN_SEP + BOT_PREFIX


def test_multi_turn_prompt_alternates_prefixes():
    p = build_payload(
        [{"role": "user", "text": "hi"}, {"role": "bot", "text": "arr"},
         {"role": "user", "text": "more"}],
        {},
    )
    assert p["prompt"] == (
        USER_PREFIX + "hi" + TURN_SEP + BOT_PREFIX + "arr"
        + TURN_SEP + USER_PREFIX + "more" + TURN_SEP + BOT_PREFIX
    )


def test_payload_streams_and_stops_where_rejection_sampling_does():
    p = build_payload([{"role": "user", "text": "ahoy"}], {})
    assert p["stream"] is True
    assert p["stop"] == STOPS


def test_ui_options_are_coerced_from_strings():
    # The browser sends whatever the number inputs hold; llama-server rejects
    # a string where it wants a float.
    p = build_payload([{"role": "user", "text": "x"}],
                      {"temperature": "0.2", "top_k": "7", "max_tokens": "33"})
    assert p["temperature"] == 0.2
    assert p["top_k"] == 7
    assert p["n_predict"] == 33
    assert isinstance(json.dumps(p), str)


def test_defaults_fill_in_for_a_bare_request():
    p = build_payload([{"role": "user", "text": "x"}], {})
    assert p["n_predict"] == 200
    assert 0 < p["temperature"] <= 2


def test_reply_headroom_leaves_room_inside_the_context():
    assert REPLY_HEADROOM + 200 < DEFAULT_CTX
