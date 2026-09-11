"""Offer selection and bid derivation.

Two real failure modes are pinned here, both measured against the live vast CLI
on 2026-09-10:

  * `type=bid` is not a query field. Passing it gets a "Unrecognized field"
    *warning* and a list of on-demand offers — the run then costs 2-3x with
    nothing in the output that looks like an error. The flag is `-i`.
  * vast bills your bid, not the floor, so the bid must come from the chosen
    offer's own `min_bid`. Bidding the cap donates the difference.

Everything here runs offline; `search` is stubbed.
"""

from __future__ import annotations

import subprocess

import pytest

from nanobeard.vast_offers import (
    Offer,
    build_query,
    cheapest_per_gpu,
    render_board,
    search,
    to_offers,
)


def _raw(**over) -> dict:
    row = {
        "id": 123,
        "machine_id": 108820,
        "dph_total": 0.25,
        "min_bid": 0.20,
        "reliability2": 0.99,
        "inet_down": 800.0,
        "cuda_max_good": 13.0,
    }
    row.update(over)
    return row


# ----- bid derivation -----
def test_bid_sits_just_above_the_floor_not_at_the_cap():
    o = to_offers("RTX_4090", [_raw(min_bid=0.20)])[0]
    # The old script bid MAX_DPH (0.40) regardless — double, for nothing.
    assert o.bid(multiplier=1.15, cap=0.40) == 0.23


def test_bid_never_exceeds_the_cap():
    o = to_offers("RTX_5090", [_raw(min_bid=0.39)])[0]
    assert o.bid(multiplier=1.15, cap=0.40) == 0.40


def test_bid_multiplier_of_one_bids_the_floor():
    o = to_offers("RTX_3090", [_raw(min_bid=0.107)])[0]
    assert o.bid(multiplier=1.0, cap=0.40) == 0.107


# ----- parsing -----
def test_rows_without_a_price_are_dropped():
    assert to_offers("g", [_raw(dph_total=None), _raw()]) == to_offers("g", [_raw()])


def test_missing_min_bid_falls_back_to_dph():
    # On-demand rows carry no min_bid. Keep them comparable rather than
    # silently dropping the whole GPU from the board.
    o = to_offers("g", [_raw(min_bid=None, dph_total=0.31)])[0]
    assert o.min_bid == 0.31


def test_missing_optional_fields_do_not_crash():
    o = to_offers("g", [{"id": 1, "dph_total": 0.2}])[0]
    assert (o.reliability, o.inet_down, o.cuda) == (0.0, 0.0, 0.0)


# ----- query building -----
def test_query_never_contains_type_bid():
    q = build_query("RTX_4090", 0.40, 200, 0.95, "12.9", datacenter=False)
    assert "type=" not in q, "interruptible is the -i flag, not a query field"


def test_datacenter_is_opt_in():
    assert "datacenter" not in build_query("g", 0.4, 200, 0.95, "12.9", datacenter=False)
    assert "datacenter=true" in build_query("g", 0.4, 200, 0.95, "12.9", datacenter=True)


def test_query_carries_every_filter():
    q = build_query("RTX_5090", 0.33, 500, 0.98, "12.8", datacenter=False)
    for frag in ("gpu_name=RTX_5090", "dph_total<=0.33", "inet_down>=500",
                 "reliability>=0.98", "cuda_vers>=12.8", "num_gpus=1"):
        assert frag in q


# ----- the silent-fallback guard -----
def test_unrecognized_field_warning_is_fatal(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Warning: Unrecognized field: type\n[]", stderr=""
        )

    monkeypatch.setattr("nanobeard.vast_offers.shutil.which", lambda _: "/usr/bin/vastai")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="rejected a query field"):
        search("gpu_name=RTX_4090 type=bid")


def test_interruptible_passes_the_flag_not_a_field(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr("nanobeard.vast_offers.shutil.which", lambda _: "/usr/bin/vastai")
    monkeypatch.setattr(subprocess, "run", fake_run)
    search("gpu_name=RTX_4090", interruptible=True)
    assert "-i" in seen["cmd"]
    search("gpu_name=RTX_4090", interruptible=False)
    assert "-i" not in seen["cmd"]


def test_missing_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setattr("nanobeard.vast_offers.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="uv tool install vastai"):
        search("anything")


# ----- ranking -----
def test_board_is_ranked_by_floor_price_across_gpus(monkeypatch):
    prices = {"RTX_4090": 0.20, "RTX_5090": 0.33, "RTX_3090": 0.107}

    def fake_search(query, interruptible=True):
        gpu = query.split("gpu_name=")[1].split()[0]
        return [_raw(min_bid=prices[gpu], dph_total=prices[gpu] + 0.001)]

    monkeypatch.setattr("nanobeard.vast_offers.search", fake_search)
    ranked = cheapest_per_gpu(list(prices), max_dph=0.4, inet_down=200,
                              reliability=0.95, cuda_vers="12.9", datacenter=False)
    assert [o.gpu for o in ranked] == ["RTX_3090", "RTX_4090", "RTX_5090"]


def test_a_gpu_with_no_offers_is_skipped_not_fatal(monkeypatch):
    def fake_search(query, interruptible=True):
        return [] if "RTX_5090" in query else [_raw()]

    monkeypatch.setattr("nanobeard.vast_offers.search", fake_search)
    ranked = cheapest_per_gpu(["RTX_5090", "RTX_4090"], max_dph=0.4, inet_down=200,
                              reliability=0.95, cuda_vers="12.9", datacenter=False)
    assert [o.gpu for o in ranked] == ["RTX_4090"]


def test_broken_machines_can_be_excluded(monkeypatch):
    """A dead host advertises one offer per GPU slot, so a naive retry lands on
    the same machine. Measured: three launches in a row picked machine 108820,
    all refusing to start with "GPU error"."""
    monkeypatch.setattr(
        "nanobeard.vast_offers.search",
        lambda q, interruptible=True: [
            _raw(id=1, machine_id=108820, min_bid=0.05),
            _raw(id=2, machine_id=999, min_bid=0.30),
        ],
    )
    (best,) = cheapest_per_gpu(["RTX_4090"], exclude_machines={108820}, max_dph=0.4,
                               inet_down=200, reliability=0.95, cuda_vers="12.9",
                               datacenter=False)
    assert best.machine_id == 999, "cheapest offer was on the excluded machine"


def test_excluding_every_machine_yields_nothing(monkeypatch):
    monkeypatch.setattr("nanobeard.vast_offers.search",
                        lambda q, interruptible=True: [_raw(machine_id=7)])
    assert cheapest_per_gpu(["RTX_4090"], exclude_machines={7}, max_dph=0.4,
                            inet_down=200, reliability=0.95, cuda_vers="12.9",
                            datacenter=False) == []


def test_cheapest_offer_wins_within_one_gpu(monkeypatch):
    monkeypatch.setattr(
        "nanobeard.vast_offers.search",
        lambda q, interruptible=True: [_raw(id=1, min_bid=0.30), _raw(id=2, min_bid=0.11)],
    )
    (best,) = cheapest_per_gpu(["RTX_4090"], max_dph=0.4, inet_down=200,
                               reliability=0.95, cuda_vers="12.9", datacenter=False)
    assert best.offer_id == 2


def test_board_renders_one_row_per_offer():
    offers = [Offer("RTX_4090", 1, 108820, 0.201, 0.200, 0.999, 665, 13.0)]
    board = render_board(offers, multiplier=1.15, cap=0.40)
    assert "RTX_4090" in board
    assert "0.230" in board  # the derived bid, not the cap
    assert len(board.splitlines()) == 3  # header, rule, one row
