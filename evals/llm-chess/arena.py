"""Swiss-style matchmaking, A/B-swap judging, cached calls, JSONL persistence.

Why Swiss instead of round-robin: with M models the round-robin cost is
`C(M, 2)` matches per prompt — most are blowouts. Pairing close-rated models
(Elo proximity) concentrates judge calls where the ranking is uncertain.

Why the A/B swap: LLM judges have a slot bias — they prefer "A" or "B"
position regardless of content. For every pair we judge twice with the
response order swapped. If the verdicts agree, that's the winner; if they
disagree, it's a tie. This costs 2x judge calls but kills the dominant bias.

Why the cache: judges at temperature 0 are deterministic on the same inputs.
Same `(judge, version, prompt, response_a, response_b)` should not bill a
second time. Version is in the cache key so new judge releases miss every
prior entry by design.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from elo import DEFAULT_K, DEFAULT_RATING, update
from judges import Judge, Verdict
from models import Model
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

_console = Console(stderr=True)


@dataclass
class Match:
    match_id: str
    prompt: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    judge: str
    judge_version: str
    winner: str  # model_a name, model_b name, or "tie"
    verdict_forward: str  # "A" / "B" / "tie" — raw judge call with (resp_a, resp_b)
    verdict_reverse: str  # "A" / "B" / "tie" — raw judge call with (resp_b, resp_a)
    reason: str
    ts: float


def load_prompts(path: str | Path) -> list[str]:
    prompts: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        prompts.append(rec["prompt"] if isinstance(rec, dict) else rec)
    return prompts


# ---------------------------------------------------------------------------
# Judge call cache
# ---------------------------------------------------------------------------


def _cache_key(judge_name: str, judge_version: str, prompt: str, a: str, b: str) -> str:
    h = hashlib.sha256()
    for s in (judge_name, judge_version, prompt, a, b):
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _response_key(model_name: str, prompt: str) -> str:
    h = hashlib.sha256()
    for s in (model_name, prompt):
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class JudgeCache:
    """Append-only JSONL dedupe of judge verdicts, thread-safe-ish via lock."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.mem[rec["key"]] = rec["verdict"]
        self._f = self.path.open("a")
        import threading

        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Verdict | None:
        rec = self.mem.get(key)
        if rec is None:
            self.misses += 1
            return None
        self.hits += 1
        return Verdict(**rec)

    def put(self, key: str, verdict: Verdict) -> None:
        rec = asdict(verdict)
        with self._lock:
            self.mem[key] = rec
            self._f.write(json.dumps({"key": key, "verdict": rec}) + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()


def _judge_with_cache(
    judge: Judge,
    prompt: str,
    response_a: str,
    response_b: str,
    cache: JudgeCache | None,
) -> Verdict:
    if cache is None:
        return judge.judge(prompt, response_a, response_b)
    key = _cache_key(judge.name, judge.version, prompt, response_a, response_b)
    hit = cache.get(key)
    if hit is not None:
        return hit
    v = judge.judge(prompt, response_a, response_b)
    cache.put(key, v)
    return v


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Append-only JSONL dedupe of model responses.

    Key: sha256(model_name + prompt). Since model identity is enforced by
    the frozen-dataclass + registry contract (same name = same params), the
    cached response is a valid replay of "what this player said on this
    prompt". This freezes a single sample per (model, prompt); reruns reuse
    it instead of re-rolling a stochastic generation.

    To force a fresh sample for one model, either rename it (encode a new
    version suffix) or delete the cache file.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.mem[rec["key"]] = rec["response"]
        self._f = self.path.open("a")
        import threading

        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        rec = self.mem.get(key)
        if rec is None:
            self.misses += 1
            return None
        self.hits += 1
        return rec

    def put(self, key: str, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self.mem[key] = response
            self._f.write(
                json.dumps(
                    {"key": key, "model": model, "prompt": prompt, "response": response}
                )
                + "\n"
            )
            self._f.flush()

    def close(self) -> None:
        self._f.close()


def _generate_with_cache(model: Model, prompt: str, cache: ResponseCache | None) -> str:
    if cache is None:
        return model.generate(prompt)
    key = _response_key(model.name, prompt)
    hit = cache.get(key)
    if hit is not None:
        return hit
    resp = model.generate(prompt)
    cache.put(key, model.name, prompt, resp)
    return resp


# ---------------------------------------------------------------------------
# A/B swap → consolidated verdict
# ---------------------------------------------------------------------------


def consolidate_swap(forward: Verdict, reverse: Verdict) -> tuple[str, str]:
    """Combine a forward `(A=resp_a, B=resp_b)` verdict and a reverse
    `(A=resp_b, B=resp_a)` verdict into a single ('a' | 'b' | 'tie', reason).

    Consistent: forward and reverse pick the same model → that model wins.
    Inconsistent or any explicit tie → tie. Reason concatenates both.
    """
    fwd_picks = {"A": "a", "B": "b", "tie": "tie"}[forward.winner]
    rev_picks = {"A": "b", "B": "a", "tie": "tie"}[reverse.winner]
    reason = forward.reason
    if reverse.reason and reverse.reason != forward.reason:
        reason = f"{forward.reason} | swap: {reverse.reason}"
    if fwd_picks == "tie" or rev_picks == "tie":
        return "tie", reason
    if fwd_picks == rev_picks:
        return fwd_picks, reason
    return "tie", reason or "swap disagreed"


# ---------------------------------------------------------------------------
# Matchmaking
# ---------------------------------------------------------------------------


def _seed_ratings_from_log(out_path: Path, base: float, k: float) -> dict[str, float]:
    """Warm-start ratings from prior matches so Swiss converges faster."""
    ratings: dict[str, float] = defaultdict(lambda: base)
    if not out_path.exists():
        return ratings
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        a, b, w = m["model_a"], m["model_b"], m["winner"]
        if w == a:
            score_a = 1.0
        elif w == b:
            score_a = 0.0
        else:
            score_a = 0.5
        ratings[a], ratings[b] = update(ratings[a], ratings[b], score_a, k=k)
    return ratings


def _swiss_pairings(
    models: Sequence[Model],
    ratings: dict[str, float],
    played: set[frozenset[str]],
    rng: random.Random,
) -> list[tuple[Model, Model]]:
    """Sort by rating; pair each model with nearest-rated unplayed opponent."""
    ordered = sorted(models, key=lambda m: (ratings[m.name], rng.random()))
    remaining = list(ordered)
    pairs: list[tuple[Model, Model]] = []
    while len(remaining) >= 2:
        a = remaining.pop(0)
        partner_idx = None
        for i, candidate in enumerate(remaining):
            if frozenset([a.name, candidate.name]) not in played:
                partner_idx = i
                break
        if partner_idx is None:
            partner_idx = 0  # forced repeat — unique pairings exhausted
        b = remaining.pop(partner_idx)
        pairs.append((a, b))
        played.add(frozenset([a.name, b.name]))
    return pairs


def _judge_pair(
    j: Judge,
    prompt: str,
    ra: str,
    rb: str,
    cache: JudgeCache | None,
) -> tuple[Verdict, Verdict]:
    """Two calls: forward (A=ra, B=rb) and reverse (A=rb, B=ra)."""
    fwd = _judge_with_cache(j, prompt, ra, rb, cache)
    rev = _judge_with_cache(j, prompt, rb, ra, cache)
    return fwd, rev


def run_swiss(
    models: Sequence[Model],
    judges: Sequence[Judge],
    prompts: Sequence[str],
    out_path: str | Path,
    rounds: int = 3,
    k: float = DEFAULT_K,
    base: float = DEFAULT_RATING,
    cache_path: str | Path | None = None,
    response_cache_path: str | Path | None = None,
    seed: int = 0,
    max_workers: int = 8,
    quiet: bool = False,
) -> list[Match]:
    """For each prompt, run `rounds` Swiss rounds, A/B-swap every judge call.

    `cache_path` dedupes judge calls; `response_cache_path` dedupes model
    generations. Both default to None (disabled). `max_workers` bounds
    parallelism. `quiet=True` suppresses all rich output.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ratings = _seed_ratings_from_log(out_path, base=base, k=k)
    cache = JudgeCache(cache_path) if cache_path else None
    rcache = ResponseCache(response_cache_path) if response_cache_path else None
    rng = random.Random(seed)
    matches: list[Match] = []
    pool = ThreadPoolExecutor(max_workers=max_workers)
    unique_pairs_possible = len(list(itertools.combinations(models, 2)))

    if not quiet:
        _console.rule(
            f"[bold cyan]llm-chess[/]  ·  {len(models)} models  ·  "
            f"{len(judges)} judges  ·  {len(prompts)} prompts  ·  {rounds} rounds"
        )
        _console.log(f"[dim]models:[/] {', '.join(m.name for m in models)}")
        _console.log(
            "[dim]judges:[/] "
            + ", ".join(
                f"{j.name}@{getattr(j, 'version', '?')}" for j in judges
            )
        )
        _console.log(f"[dim]out:[/] {out_path}")
        if cache is not None:
            _console.log(f"[dim]judge cache:[/] {cache.path} (prior entries: {len(cache.mem)})")
        if rcache is not None:
            _console.log(
                f"[dim]response cache:[/] {rcache.path} (prior entries: {len(rcache.mem)})"
            )
        _console.log(f"[dim]workers:[/] {max_workers}  ·  [dim]K:[/] {k}  ·  [dim]seed:[/] {seed}")

    counts = {"a_wins": 0, "b_wins": 0, "ties": 0}

    def _run() -> None:
        with out_path.open("a") as f:
            progress = None
            task = None
            if not quiet:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn(
                        "[dim]· hits {task.fields[hits]} miss {task.fields[miss]} · "
                        "tie {task.fields[ties]}[/]"
                    ),
                    TimeElapsedColumn(),
                    console=_console,
                    transient=False,
                )
                progress.start()
                task = progress.add_task(
                    "prompts", total=len(prompts), hits=0, miss=0, ties=0
                )

            try:
                for i, prompt in enumerate(prompts, 1):
                    gen_futures = {
                        m.name: pool.submit(_generate_with_cache, m, prompt, rcache)
                        for m in models
                    }
                    responses = {n: fut.result() for n, fut in gen_futures.items()}

                    played: set[frozenset[str]] = set()
                    prompt_matches = 0
                    for _round in range(rounds):
                        if len(played) >= unique_pairs_possible:
                            break
                        pairs = _swiss_pairings(models, ratings, played, rng)

                        judge_tasks = [(a, b, j) for a, b in pairs for j in judges]
                        judge_futures = [
                            pool.submit(
                                _judge_pair,
                                j,
                                prompt,
                                responses[a.name],
                                responses[b.name],
                                cache,
                            )
                            for a, b, j in judge_tasks
                        ]

                        for (a, b, j), fut in zip(judge_tasks, judge_futures, strict=True):
                            ra, rb = responses[a.name], responses[b.name]
                            fwd, rev = fut.result()
                            winner_slot, reason = consolidate_swap(fwd, rev)
                            if winner_slot == "a":
                                winner_name = a.name
                                score_a = 1.0
                                counts["a_wins"] += 1
                            elif winner_slot == "b":
                                winner_name = b.name
                                score_a = 0.0
                                counts["b_wins"] += 1
                            else:
                                winner_name = "tie"
                                score_a = 0.5
                                counts["ties"] += 1
                            ratings[a.name], ratings[b.name] = update(
                                ratings[a.name], ratings[b.name], score_a, k=k
                            )
                            rec = Match(
                                match_id=str(uuid.uuid4()),
                                prompt=prompt,
                                model_a=a.name,
                                model_b=b.name,
                                response_a=ra,
                                response_b=rb,
                                judge=j.name,
                                judge_version=getattr(j, "version", ""),
                                winner=winner_name,
                                verdict_forward=fwd.winner,
                                verdict_reverse=rev.winner,
                                reason=reason,
                                ts=time.time(),
                            )
                            matches.append(rec)
                            prompt_matches += 1
                            f.write(json.dumps(asdict(rec)) + "\n")

                    if progress is not None and task is not None:
                        progress.update(
                            task,
                            advance=1,
                            description=f"prompt {i}/{len(prompts)}",
                            hits=cache.hits if cache else 0,
                            miss=cache.misses if cache else 0,
                            ties=counts["ties"],
                        )
            finally:
                if progress is not None:
                    progress.stop()

    try:
        _run()
    finally:
        pool.shutdown(wait=True)
        if cache is not None:
            cache.close()
        if rcache is not None:
            rcache.close()

    if not quiet:
        _console.log(
            f"[bold green]done[/]  matches={len(matches)}  "
            f"wins_a={counts['a_wins']} wins_b={counts['b_wins']} ties={counts['ties']}"
        )
        if rcache is not None:
            tot = rcache.hits + rcache.misses
            rate = (rcache.hits / tot * 100) if tot else 0.0
            _console.log(
                f"[dim]response cache[/]  hits={rcache.hits}  misses={rcache.misses}  "
                f"hit_rate={rate:.0f}%"
            )
        if cache is not None:
            tot = cache.hits + cache.misses
            rate = (cache.hits / tot * 100) if tot else 0.0
            _console.log(
                f"[dim]judge cache[/]    hits={cache.hits}  misses={cache.misses}  "
                f"hit_rate={rate:.0f}%"
            )

    return matches
