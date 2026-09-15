"""A searchable Wikipedia-paragraph corpus built from HotpotQA, plus BM25.

HotpotQA ships each question with 10 paragraphs — 2 that answer it and 8
distractors. Pooling those paragraphs across every question gives a corpus of
tens of thousands, so the model has to actually find things rather than pick 2
of 10. The supporting-fact titles are the retrieval ground truth, and the
`answer` field is the reward.

BM25 is written out here rather than pulled from a library: it is ~40 lines, it
is the one part of the search stack worth understanding when the reward curve
misbehaves, and a dependency would hide it.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

DEFAULT_INDEX = Path("data/search/hotpot_bm25.pkl")
# Standard BM25 constants. k1 controls term-frequency saturation, b how much a
# long document is penalised.
K1 = 1.5
B = 0.75
_WORD = re.compile(r"[a-z0-9]+")
# Words so common they cost time and add nothing. Kept short on purpose: an
# aggressive list would silently delete real query terms like "who" or "when".
STOPWORDS = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "of", "on", "or", "that", "the", "to", "was", "were", "with"])


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


@dataclass
class Doc:
    title: str
    text: str


class BM25:
    """Classic BM25 over an in-memory inverted index."""

    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, d in enumerate(docs):
            terms = Counter(tokenize(f"{d.title} {d.text}"))
            self.lengths.append(sum(terms.values()))
            for term, tf in terms.items():
                self.postings[term].append((i, tf))
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        n = len(docs)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = 1 - B + B * (self.lengths[i] / self.avg_len) if self.avg_len else 1.0
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:k]

    def __len__(self) -> int:
        return len(self.docs)


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    gold_titles: list[str]


def build(split: str = "validation", max_questions: int | None = None):
    """(BM25 index, questions). Paragraphs are deduplicated by title."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    if max_questions:
        ds = ds.select(range(min(max_questions, len(ds))))

    by_title: dict[str, str] = {}
    questions: list[Question] = []
    # Rows are dicts at runtime; the datasets stubs describe a wider union.
    for row in cast(Iterable[dict], ds):
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"], strict=True):
            if title not in by_title:
                by_title[title] = "".join(sentences).strip()
        gold = sorted(set(row["supporting_facts"]["title"]))
        questions.append(Question(
            qid=row["id"], question=row["question"], answer=row["answer"], gold_titles=gold
        ))

    docs = [Doc(title=t, text=x) for t, x in sorted(by_title.items())]
    return BM25(docs), questions


def save(index: BM25, questions: list[Question], path: Path = DEFAULT_INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"index": index, "questions": questions}, f)


def load(path: Path = DEFAULT_INDEX):
    with path.open("rb") as f:
        d = pickle.load(f)
    return d["index"], d["questions"]


def main() -> None:
    import argparse

    # Build through the imported module, not through this one. Run as
    # `python -m nanobeard.rl.corpus` these classes are `__main__.BM25`, and a
    # pickle of them cannot be loaded by anything else — the index would only
    # ever open in the script that wrote it.
    from nanobeard.rl import corpus as mod

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--out", default=str(DEFAULT_INDEX))
    args = ap.parse_args()

    index, questions = mod.build(args.split, args.max_questions)
    mod.save(index, questions, Path(args.out))
    print(f"corpus: {len(index):,} paragraphs | questions: {len(questions):,}")
    print(f"vocabulary: {len(index.postings):,} terms | mean length: {index.avg_len:.0f} tokens")

    # A retrieval floor: how often does one BM25 call on the raw question find
    # the gold paragraphs? Anything the policy learns has to beat this.
    hits = 0.0
    at1 = 0
    sample = questions[:300]
    for q in sample:
        got = [index.docs[i].title for i, _ in index.search(q.question, k=5)]
        gold = set(q.gold_titles)
        hits += len(gold & set(got)) / max(1, len(gold))
        at1 += 1 if got and got[0] in gold else 0
    print(f"\nbaseline, question used verbatim as the query (n={len(sample)}):")
    print(f"  gold recall@5: {hits / len(sample):.1%}")
    print(f"  precision@1:   {at1 / len(sample):.1%}")


if __name__ == "__main__":
    main()
