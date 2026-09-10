"""The judge shuffles candidate order to kill position bias, then maps the
pick back. If that mapping is off by even one, every verdict points at the
wrong reply while still looking perfectly plausible — so it gets its own test.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

from nanobeard.rejection import judge as J


def _fake_response(pick_text: str):
    """Stub OpenRouter: always pick whichever presented slot holds pick_text."""

    class FakeResp:
        def __init__(self, body):
            self._b = json.dumps(body).encode()

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        sent = json.loads(req.data)
        user_msg = sent["messages"][1]["content"]
        block = user_msg.split("Candidate replies:\n", 1)[1]
        slot = next(
            i for i, line in enumerate(block.strip().splitlines(), 1)
            if pick_text in line
        )
        return FakeResp({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps({"pick": slot, "reason": "stub"})},
            }]
        })

    return _open


TURNS = [{"role": "user", "text": "hello"}]


def test_pick_maps_back_to_original_index():
    cands = [f"reply-{i}" for i in range(10)]
    target = "reply-7"
    with patch("urllib.request.urlopen", _fake_response(target)):
        # Many seeds => many different shuffles; the answer must not move.
        for seed in range(40):
            v = J.judge(TURNS, cands, api_key="x", seed=seed)
            assert v["pick"] is not None
            assert cands[v["pick"] - 1] == target, f"seed {seed} mapped to the wrong reply"


def test_shuffle_actually_varies_presented_order():
    cands = [f"reply-{i}" for i in range(10)]
    seen = set()

    def _capture(req, timeout=None):
        sent = json.loads(req.data)
        block = sent["messages"][1]["content"].split("Candidate replies:\n", 1)[1]
        seen.add(tuple(line.split(". ", 1)[1] for line in block.strip().splitlines()))
        raise AssertionError("stop")

    for seed in range(12):
        with contextlib.suppress(AssertionError), patch("urllib.request.urlopen", _capture):
            J.judge(TURNS, cands, api_key="x", seed=seed, retries=1)
    assert len(seen) > 1, "candidate order never changed — shuffle is a no-op"


def test_votes_take_the_majority():
    cands = [f"reply-{i}" for i in range(10)]
    with patch("urllib.request.urlopen", _fake_response("reply-3")):
        v = J.judge(TURNS, cands, api_key="x", seed=1, votes=3)
    assert cands[v["pick"] - 1] == "reply-3"
    assert v["votes"] == "3/3"


def test_out_of_range_pick_is_rejected():
    assert J._parse_verdict(json.dumps({"pick": 99, "reason": "x"}), 10)["pick"] is None
    assert J._parse_verdict("not json at all", 10)["pick"] is None
