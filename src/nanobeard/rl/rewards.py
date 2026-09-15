"""Rewards for the search task, and the diagnostics that are deliberately NOT rewards.

The distinction is the whole design. The model is scored on whether its final
answer matches the dataset's — a string it cannot game. Retrieval quality is
measured and logged, but never paid for, because retrieval metrics are trivially
hackable: a query of "the" retrieves everything and scores perfect recall, while
a hyper-specific query returns one document and scores perfect precision. Neither
teaches search.

Answer normalisation follows the SQuAD/HotpotQA convention — lowercase, strip
punctuation, drop leading articles — so that "The Beatles", "beatles" and
"beatles." all count as the same answer. Without it the reward is mostly
measuring punctuation.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Partial credit, reported but not rewarded.

    Kept out of the reward on purpose: F1 pays for overlapping words, so a model
    can raise it by answering at length. Exact match cannot be padded.
    """
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


@dataclass
class RetrievalStats:
    """Diagnostics. Never summed into the reward — see the module docstring."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_retrieved: int = 0
    n_gold: int = 0


def retrieval_stats(retrieved_titles: list[str], gold_titles: list[str]) -> RetrievalStats:
    got, gold = set(retrieved_titles), set(gold_titles)
    if not gold:
        return RetrievalStats(n_retrieved=len(got))
    hit = len(got & gold)
    precision = hit / len(got) if got else 0.0
    recall = hit / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return RetrievalStats(precision, recall, f1, len(got), len(gold))


@dataclass
class Reward:
    total: float
    answer_em: float
    format_ok: float
    searched: float
    breakdown: dict = field(default_factory=dict)


# A small payment for producing a parseable answer at all. Without it an
# untrained policy gets a flat zero on every rollout, every advantage is zero,
# and GRPO has nothing to push against. It is deliberately far smaller than the
# answer reward so "emit <answer></answer> and stop" is never the better move.
FORMAT_BONUS = 0.05
SEARCH_BONUS = 0.05


def compute(
    pred_answer: str | None,
    gold_answer: str,
    *,
    did_search: bool,
    retrieved_titles: list[str] | None = None,
    gold_titles: list[str] | None = None,
) -> Reward:
    """Score one rollout. Only `answer_em` is worth real points."""
    format_ok = 1.0 if pred_answer is not None and pred_answer.strip() else 0.0
    em = exact_match(pred_answer or "", gold_answer)
    searched = 1.0 if did_search else 0.0
    total = em + FORMAT_BONUS * format_ok + SEARCH_BONUS * searched

    stats = retrieval_stats(retrieved_titles or [], gold_titles or [])
    return Reward(
        total=total,
        answer_em=em,
        format_ok=format_ok,
        searched=searched,
        breakdown={
            "token_f1": token_f1(pred_answer or "", gold_answer),
            "retrieval_precision": stats.precision,
            "retrieval_recall": stats.recall,
            "retrieval_f1": stats.f1,
            "n_retrieved": stats.n_retrieved,
        },
    )
