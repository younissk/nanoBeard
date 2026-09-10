"""Sanity-check a rejection-sampling run before you train on it.

    uv run python -m nanobeard.rejection.analyze --in runs/rejection/<run>.jsonl

Reports the three ways this pipeline has actually gone wrong:

  1. Diversity collapse  — N samples that are really K distinct ideas, so the
     extra generation bought nothing.
  2. Judge position bias — a judge that prefers slot 1 is picking by seed, not
     by quality. Measured with a chi-square against uniform.
  3. Judge length bias   — the classic LLM-judge failure: longest reply wins.

Everything here is descriptive. It never rewrites the run.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

# Pirate filler that carries no content; stripped when clustering restatements.
FILLER = re.compile(r"\b(oh|no|aye|arrr|arr|yarr|ye|th|be|me|matey|hearty|savvy|yo|ho)\b")


def content_key(text: str) -> str:
    """Normalise a reply down to its content words, order-insensitive."""
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    t = FILLER.sub(" ", t)
    return " ".join(sorted(set(t.split())))


def chi_square_uniform(counts: list[int]) -> float:
    total = sum(counts)
    if not total:
        return 0.0
    exp = total / len(counts)
    return sum((o - exp) ** 2 / exp for o in counts)


# 95th / 99th percentile of chi-square by degrees of freedom, for df we actually use.
CRIT = {9: (16.92, 21.67), 29: (42.56, 49.59), 19: (30.14, 36.19), 4: (9.49, 13.28)}


def report(prompts: list[dict], verdicts: dict | None) -> None:
    n_per = len(prompts[0]["samples"])
    print(f"{len(prompts)} prompts x {n_per} samples\n")

    # ---- 1. diversity -----------------------------------------------------
    exact, content, worst = [], [], []
    for p in prompts:
        texts = [s["text"] for s in p["samples"]]
        e = len({" ".join(t.strip().lower().split()) for t in texts})
        groups = collections.Counter(content_key(t) for t in texts)
        content.append(len(groups))
        exact.append(e)
        worst.append((len(groups), max(groups.values()), p["id"], p["category"]))

    print("DIVERSITY")
    print(f"  exact-unique     {statistics.mean(exact):>5.1f} / {n_per}")
    print(f"  content-unique   {statistics.mean(content):>5.1f} / {n_per}"
          "   (punctuation + pirate filler stripped)")
    worst.sort()
    print("  worst collapse:")
    for u, mx, pid, cat in worst[:5]:
        print(f"    {pid:<10} {cat:<14} {u:>3}/{n_per} distinct, biggest cluster {mx}")

    if not verdicts:
        print("\n(no verdicts file — run `judge --all` for judge diagnostics)")
        return

    picks = {pid: v["pick"] for pid, v in verdicts.items() if v.get("pick")}
    print(f"\nJUDGE  ({len(picks)}/{len(prompts)} with a pick)")

    # ---- 2. position bias -------------------------------------------------
    counts = [0] * n_per
    for pk in picks.values():
        counts[pk - 1] += 1
    chi = chi_square_uniform(counts)
    df = n_per - 1
    crit = CRIT.get(df)
    verdict = "n/a"
    if crit is not None:
        p95, p99 = crit
        verdict = ("BIASED (p<0.01)" if chi > p99 else
                   "biased (p<0.05)" if chi > p95 else "consistent with uniform")
    print(f"  position chi-square = {chi:.1f} (df={df})  -> {verdict}")
    if crit is not None:
        print(f"    critical: {crit[0]} at p=0.05, {crit[1]} at p=0.01")
    top = sorted(range(n_per), key=lambda i: -counts[i])[:3]
    exp = len(picks) / n_per
    print(f"    expected ~{exp:.1f} per slot; top slots: "
          + ", ".join(f"#{i+1}={counts[i]}" for i in top))

    # ---- 3. length bias ---------------------------------------------------
    ranks, chosen_len, all_len, trunc_hits, trunc_all = [], [], [], 0, 0
    for p in prompts:
        pk = picks.get(p["id"])
        toks = [s["n"] or 0 for s in p["samples"]]
        all_len += toks
        trunc_all += sum(1 for s in p["samples"] if s["stop"] == "limit")
        if not pk:
            continue
        chosen_len.append(toks[pk - 1])
        ranks.append(sorted(toks, reverse=True).index(toks[pk - 1]) + 1)
        if p["samples"][pk - 1]["stop"] == "limit":
            trunc_hits += 1
    mid = (n_per + 1) / 2
    print(f"  chosen length    {statistics.mean(chosen_len):>5.1f} tok "
          f"(pool mean {statistics.mean(all_len):.1f})")
    print(f"  length rank      {statistics.mean(ranks):>5.2f} of {n_per} "
          f"(1=longest; {mid:.1f} = no preference)")
    base = 100 * trunc_all / (len(prompts) * n_per)
    print(f"  picked truncated {100*trunc_hits/max(1,len(picks)):>5.1f}%  (base rate {base:.1f}%)")

    # ---- 4. how often the judge concedes the pool is bad ------------------
    hedge = re.compile(r"least bad|at least|despite|incoherent|nonsensical|none of|all of the")
    weak = [pid for pid, v in verdicts.items() if hedge.search(v.get("reason", "").lower())]
    print(f"\n  verdicts conceding a weak pool: {len(weak)}/{len(verdicts)} "
          f"({100*len(weak)/max(1,len(verdicts)):.0f}%)")
    bycat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    cat_of = {p["id"]: p["category"] for p in prompts}
    for pid in verdicts:
        c = cat_of.get(pid, "?")
        bycat[c][1] += 1
        if pid in weak:
            bycat[c][0] += 1
    for c, (w, t) in sorted(bycat.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        bar = "█" * round(20 * w / t)
        print(f"    {c:<14} {w:>3}/{t:<3} {100*w/t:>3.0f}% {bar}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--verdicts", default=None, help="Default: <in>.judged.json")
    args = ap.parse_args()

    from nanobeard.rejection.viewer import build_payload, load

    inp = Path(args.inp)
    prompts = build_payload(load(inp), inp.stem)["prompts"]

    vpath = Path(args.verdicts) if args.verdicts else inp.with_suffix(".judged.json")
    verdicts = json.loads(vpath.read_text())["verdicts"] if vpath.is_file() else None
    report(prompts, verdicts)


if __name__ == "__main__":
    main()
