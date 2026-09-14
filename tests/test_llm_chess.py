"""Unit tests for evals/llm-chess.

The folder name has a hyphen and isn't a Python package, so we put it on
sys.path before importing.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LLM_CHESS = REPO / "evals" / "llm-chess"
sys.path.insert(0, str(LLM_CHESS))

from arena import (  # noqa: E402
    JudgeCache,
    ResponseCache,
    _cache_key,
    _generate_with_cache,
    _response_key,
    _seed_ratings_from_log,
    _swiss_pairings,
    consolidate_swap,
    run_swiss,
)
from elo import DEFAULT_K, expected, update  # noqa: E402
from judges import RandomJudge, Verdict  # noqa: E402
from leaderboard import _percentile, compute_ratings  # noqa: E402
from models import EchoModel, register  # noqa: E402

# ---------------------------------------------------------------------------
# elo
# ---------------------------------------------------------------------------


def test_expected_symmetric_at_equal_ratings():
    assert expected(1500, 1500) == 0.5


def test_expected_400_gap_is_91_percent():
    assert abs(expected(1900, 1500) - 0.909) < 0.01


def test_update_sum_conserved():
    a, b = update(1500, 1500, 1.0, k=32)
    assert abs((a + b) - 3000) < 1e-9


def test_update_draw_at_equal_ratings_is_noop():
    a, b = update(1500, 1500, 0.5, k=32)
    assert a == 1500 and b == 1500


def test_update_winner_gains_loser_loses():
    a_new, b_new = update(1500, 1500, 1.0, k=32)
    assert a_new > 1500 and b_new < 1500


def test_default_k_lowered_from_chess():
    assert DEFAULT_K <= 16


# ---------------------------------------------------------------------------
# JudgeCache + cache key
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    k1 = _cache_key("j", "v1", "p", "a", "b")
    k2 = _cache_key("j", "v1", "p", "a", "b")
    assert k1 == k2


def test_cache_key_sensitive_to_every_input():
    base = _cache_key("j", "v1", "p", "a", "b")
    assert _cache_key("j2", "v1", "p", "a", "b") != base
    assert _cache_key("j", "v2", "p", "a", "b") != base
    assert _cache_key("j", "v1", "p2", "a", "b") != base
    assert _cache_key("j", "v1", "p", "a2", "b") != base
    assert _cache_key("j", "v1", "p", "a", "b2") != base
    # A/B order matters — forward and reverse must hash differently
    assert _cache_key("j", "v1", "p", "b", "a") != base


def test_response_key_deterministic_and_sensitive():
    base = _response_key("m", "p")
    assert _response_key("m", "p") == base
    assert _response_key("m2", "p") != base
    assert _response_key("m", "p2") != base


def test_response_cache_persists_and_dedupes(tmp_path: Path):
    cpath = tmp_path / "r.jsonl"
    c = ResponseCache(cpath)
    c.put(_response_key("m", "p"), "m", "p", "hello")
    c.close()

    c2 = ResponseCache(cpath)
    assert c2.get(_response_key("m", "p")) == "hello"
    assert c2.get(_response_key("m", "other")) is None
    c2.close()


def test_generate_with_cache_calls_model_once(tmp_path: Path):
    calls = []

    @dataclass(frozen=True)
    class Counting:
        name: str = "counter"

        def generate(self, prompt: str) -> str:
            calls.append(prompt)
            return "x"

    m = Counting()
    rcache = ResponseCache(tmp_path / "r.jsonl")
    r1 = _generate_with_cache(m, "p1", rcache)
    r2 = _generate_with_cache(m, "p1", rcache)
    r3 = _generate_with_cache(m, "p2", rcache)
    rcache.close()
    assert r1 == r2 == r3 == "x"
    assert calls == ["p1", "p2"]


def test_run_swiss_response_cache_dedupes_across_runs(tmp_path: Path):
    gen_calls = []

    @dataclass(frozen=True)
    class Counting:
        name: str
        val: str

        def generate(self, prompt: str) -> str:
            gen_calls.append((self.name, prompt))
            return self.val

    models = [Counting("a", "A"), Counting("b", "B")]
    judges = [RandomJudge(seed=0)]
    prompts = ["p1", "p2", "p3"]
    rpath = tmp_path / "r.jsonl"

    run_swiss(
        models, judges, prompts, tmp_path / "m1.jsonl",
        rounds=2, response_cache_path=rpath, seed=0, quiet=True,
    )
    first_run = len(gen_calls)
    # 2 models × 3 prompts = 6 fresh generations
    assert first_run == 6

    run_swiss(
        models, judges, prompts, tmp_path / "m2.jsonl",
        rounds=2, response_cache_path=rpath, seed=0, quiet=True,
    )
    assert len(gen_calls) == first_run  # zero new generate calls on rerun


def test_cache_persists_across_reopen(tmp_path: Path):
    cache_path = tmp_path / "c.jsonl"
    c = JudgeCache(cache_path)
    c.put("k1", Verdict(winner="A", reason="because"))
    c.close()

    c2 = JudgeCache(cache_path)
    hit = c2.get("k1")
    assert hit == Verdict(winner="A", reason="because")
    assert c2.get("missing") is None
    c2.close()


# ---------------------------------------------------------------------------
# consolidate_swap
# ---------------------------------------------------------------------------


def test_swap_consistent_a_wins():
    fwd = Verdict("A", "fwd")
    rev = Verdict("B", "rev")  # reverse picked the slot now holding response_a
    winner, _ = consolidate_swap(fwd, rev)
    assert winner == "a"


def test_swap_consistent_b_wins():
    fwd = Verdict("B", "")
    rev = Verdict("A", "")
    winner, _ = consolidate_swap(fwd, rev)
    assert winner == "b"


def test_swap_disagreement_is_tie():
    fwd = Verdict("A", "")
    rev = Verdict("A", "")  # in reverse, "A" = response_b — judge flipped
    winner, _ = consolidate_swap(fwd, rev)
    assert winner == "tie"


def test_swap_explicit_tie_propagates():
    winner, _ = consolidate_swap(Verdict("tie", ""), Verdict("A", ""))
    assert winner == "tie"


def test_random_judge_under_swap_yields_about_half_ties():
    j = RandomJudge(seed=0)
    n = 2000
    ties = 0
    for _ in range(n):
        fwd = j.judge("p", "a", "b")
        rev = j.judge("p", "b", "a")
        w, _ = consolidate_swap(fwd, rev)
        if w == "tie":
            ties += 1
    # expected ~0.5; allow generous slack
    assert 0.40 < ties / n < 0.60


# ---------------------------------------------------------------------------
# Swiss pairings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeModel:
    name: str

    def generate(self, prompt: str) -> str:
        return f"{self.name}:{prompt}"


def test_swiss_pairings_no_repeats_until_exhausted():
    models = [FakeModel(n) for n in ["a", "b", "c", "d"]]
    ratings: dict[str, float] = {m.name: 1500.0 for m in models}
    played: set[frozenset[str]] = set()
    rng = random.Random(0)

    # 4 models → C(4,2) = 6 unique pairs → 3 unique rounds of 2 pairs each
    seen_pairs: list[frozenset[str]] = []
    for _ in range(3):
        pairs = _swiss_pairings(models, ratings, played, rng)
        for a, b in pairs:
            seen_pairs.append(frozenset([a.name, b.name]))
    assert len(set(seen_pairs)) == 6  # all unique


def test_swiss_pairings_orders_by_rating():
    models = [FakeModel(n) for n in ["weak", "mid", "strong", "best"]]
    ratings = {"weak": 1000.0, "mid": 1400.0, "strong": 1600.0, "best": 1800.0}
    played: set[frozenset[str]] = set()
    rng = random.Random(0)
    pairs = _swiss_pairings(models, ratings, played, rng)
    # Sorted ascending → first pair is (weak, mid), second is (strong, best)
    pair_names = [(a.name, b.name) for a, b in pairs]
    assert pair_names == [("weak", "mid"), ("strong", "best")]


# ---------------------------------------------------------------------------
# Warm-start ratings
# ---------------------------------------------------------------------------


def test_seed_ratings_from_log_replays_winners(tmp_path: Path):
    log = tmp_path / "m.jsonl"
    with log.open("w") as f:
        for _ in range(5):
            f.write(
                json.dumps(
                    {
                        "model_a": "alpha",
                        "model_b": "bravo",
                        "winner": "alpha",
                    }
                )
                + "\n"
            )
    ratings = _seed_ratings_from_log(log, base=1500, k=8)
    assert ratings["alpha"] > 1500
    assert ratings["bravo"] < 1500


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def test_percentile_basic():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(xs, 0.0) == 1.0
    assert _percentile(xs, 1.0) == 5.0
    assert _percentile(xs, 0.5) == 3.0


def test_compute_ratings_counts_ties(tmp_path: Path):
    log = tmp_path / "m.jsonl"
    rows = [
        {"model_a": "a", "model_b": "b", "winner": "a", "judge": "j", "judge_version": "v1"},
        {"model_a": "a", "model_b": "b", "winner": "b", "judge": "j", "judge_version": "v1"},
        {"model_a": "a", "model_b": "b", "winner": "tie", "judge": "j", "judge_version": "v1"},
    ]
    with log.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {r.model: r for r in compute_ratings(log, n_boot=0)}
    assert out["a"].wins == 1 and out["a"].losses == 1 and out["a"].draws == 1
    assert out["b"].wins == 1 and out["b"].losses == 1 and out["b"].draws == 1


def test_compute_ratings_filters_judge_versions(tmp_path: Path):
    log = tmp_path / "m.jsonl"
    rows = [
        {"model_a": "a", "model_b": "b", "winner": "a", "judge": "j", "judge_version": "v1"},
        {"model_a": "a", "model_b": "b", "winner": "b", "judge": "j", "judge_version": "v2"},
    ]
    with log.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {r.model: r for r in compute_ratings(log, n_boot=0, judge_versions={"j": "v1"})}
    assert out["a"].wins == 1 and out["a"].losses == 0


def test_compute_ratings_bootstrap_ci_brackets_point(tmp_path: Path):
    log = tmp_path / "m.jsonl"
    with log.open("w") as f:
        for w in ["a"] * 8 + ["b"] * 2:
            f.write(
                json.dumps(
                    {
                        "model_a": "a",
                        "model_b": "b",
                        "winner": w,
                        "judge": "j",
                        "judge_version": "v",
                    }
                )
                + "\n"
            )
    rows = {r.model: r for r in compute_ratings(log, n_boot=200)}
    a = rows["a"]
    # CI should bracket the point estimate
    assert a.ci_low <= a.rating <= a.ci_high
    # Bootstrap width should be non-trivial
    assert (a.ci_high - a.ci_low) > 0


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_run_swiss_end_to_end(tmp_path: Path, monkeypatch):
    # Two fake models so we don't need an LLM
    @dataclass(frozen=True)
    class Const:
        name: str
        val: str

        def generate(self, prompt: str) -> str:
            return self.val

    models = [Const("a", "alpha"), Const("b", "bravo")]
    judges = [RandomJudge(seed=42)]
    prompts = ["q1", "q2", "q3"]
    out = tmp_path / "matches.jsonl"
    cache = tmp_path / "cache.jsonl"

    matches = run_swiss(
        models, judges, prompts, out, rounds=2, cache_path=cache, seed=0, quiet=True
    )

    # Two models, one judge, three prompts → at most 1 unique pair per prompt
    # × 2 rounds capped at C(2,2)=1 → 3 matches total
    assert len(matches) == 3
    for m in matches:
        assert m.judge == "random"
        assert m.judge_version == "v1"
        assert m.winner in {"a", "b", "tie"}
        assert m.verdict_forward in {"A", "B", "tie"}
        assert m.verdict_reverse in {"A", "B", "tie"}


def test_model_registry_rejects_duplicate_name_different_config():
    # Re-registering with the same config is fine (idempotent)
    register(EchoModel())
    with pytest.raises(ValueError):

        @dataclass(frozen=True)
        class FakeEcho:
            name: str = "echo"

            def generate(self, prompt: str) -> str:
                return prompt + "!"

        register(FakeEcho())
