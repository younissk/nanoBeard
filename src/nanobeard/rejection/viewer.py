"""Build a self-contained HTML viewer for a rejection-sampling run.

    uv run python -m nanobeard.rejection.viewer \
        --in runs/rejection/frigate-360m-q8.jsonl \
        --out runs/rejection/frigate-360m-q8.html

The data is inlined, so the result opens straight from disk (file://) with no
server and no network. Ratings you make in the page live in localStorage,
keyed by run, and export as JSONL for the next stage of the pipeline.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

TEMPLATE = Path(__file__).parent / "viewer_template.html"


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_payload(rows: list[dict], run_name: str) -> dict:
    """Group flat rows by prompt so prompt/turns aren't repeated 10x."""
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    by_id: dict[str, list[dict]] = collections.defaultdict(list)
    for r in ok:
        by_id[r["id"]].append(r)

    prompts = []
    for pid, rs in by_id.items():
        rs.sort(key=lambda r: r["sample"])
        head = rs[0]
        prompts.append({
            "id": pid,
            "category": head["category"],
            "turns": head["turns"],
            "prompt": head["prompt"],
            "samples": [{
                "i": r["sample"], "seed": r["seed"],
                "text": r["completion"],
                "n": r["n_tokens"], "stop": r["stop_type"],
            } for r in rs],
        })
    prompts.sort(key=lambda p: (p["category"], p["id"]))

    toks = [s["n"] for p in prompts for s in p["samples"] if s["n"] is not None]
    trunc = sum(1 for p in prompts for s in p["samples"] if s["stop"] == "limit")

    cats = []
    for cat in sorted({p["category"] for p in prompts}):
        cps = [p for p in prompts if p["category"] == cat]
        ct = [s["n"] for p in cps for s in p["samples"] if s["n"] is not None]
        ctr = sum(1 for p in cps for s in p["samples"] if s["stop"] == "limit")
        n = sum(len(p["samples"]) for p in cps)
        cats.append({
            "name": cat, "prompts": len(cps), "samples": n,
            "meanTok": round(statistics.mean(ct), 1) if ct else 0,
            "truncPct": round(100 * ctr / n, 1) if n else 0,
        })

    head = ok[0] if ok else {}
    return {
        "run": run_name,
        "model": head.get("model", "?"),
        "params": head.get("params", {}),
        "stats": {
            "prompts": len(prompts),
            "samples": len(ok),
            "perPrompt": len(prompts[0]["samples"]) if prompts else 0,
            "errors": len(errors),
            "meanTok": round(statistics.mean(toks), 1) if toks else 0,
            "medianTok": round(statistics.median(toks), 1) if toks else 0,
            "truncPct": round(100 * trunc / len(ok), 1) if ok else 0,
            "categories": len(cats),
        },
        "categories": cats,
        "prompts": prompts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None, help="Default: <input>.html")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp.with_suffix(".html")

    rows = load(inp)
    payload = build_payload(rows, inp.stem)

    # Pre-computed verdicts from `judge --all`, if they exist. Baked in so the
    # picks show instantly and offline; the live button still works without them.
    judged = inp.with_suffix(".judged.json")
    if judged.is_file():
        jd = json.loads(judged.read_text())
        payload["verdicts"] = jd.get("verdicts", {})
        payload["judgeModel"] = jd.get("model")
        print(f"  baked in {len(payload['verdicts'])} verdicts from {judged.name}")
    else:
        payload["verdicts"] = {}
    # `</` would close the inlining <script> early.
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)

    kb = out.stat().st_size / 1024
    s = payload["stats"]
    print(f"{s['prompts']} prompts x {s['perPrompt']} samples -> {out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
