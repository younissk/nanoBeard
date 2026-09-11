"""Run every gate against one model and write a comparable report.

    # against a GGUF (spawns its own llama-server)
    uv run python -m nanobeard.evals.run --model export/gguf/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf

    # against something already serving
    uv run python -m nanobeard.evals.run --server http://127.0.0.1:8899 --label stock

    # the prompt-only pirate baseline the fine-tune has to beat
    uv run python -m nanobeard.evals.run --model <gguf> --persona --label prompt-only

    # compare two reports
    uv run python -m nanobeard.evals.run --compare runs/evals/stock.json runs/evals/lora.json

The point of running all three together is that they are only meaningful as a
set. Voice going up is not a result; voice going up *while GSM8K and tool-calling
hold* is.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nanobeard.chat.server import LlamaServer
from nanobeard.evals import gsm8k, tools, voice
from nanobeard.evals.client import ChatClient

OUT_DIR = Path("runs/evals")

# The baseline persona. Kept here, not in a config, because every prompt-only
# number in the report is only interpretable next to the exact text that produced it.
PERSONA = (
    "You are Nano Beard, a salty pirate. Always speak in heavy pirate dialect: "
    "use ahoy, arr, matey, ye, be, th', doubloons. Never break character. "
    "Still answer the question correctly."
)


def evaluate(client: ChatClient, n_gsm8k: int, workers: int, skip: set[str]) -> dict:
    report: dict = {"gates": {}}
    if "gsm8k" not in skip:
        t0 = time.time()
        problems = gsm8k.load_problems(n_gsm8k)
        report["gates"]["gsm8k"] = gsm8k.run(client, problems, workers=workers)
        report["gates"]["gsm8k"]["seconds"] = round(time.time() - t0, 1)
    if "tools" not in skip:
        t0 = time.time()
        report["gates"]["tools"] = tools.run(client, workers=workers)
        report["gates"]["tools"]["seconds"] = round(time.time() - t0, 1)
    if "voice" not in skip:
        t0 = time.time()
        report["gates"]["voice"] = voice.run(client, workers=workers)
        report["gates"]["voice"]["seconds"] = round(time.time() - t0, 1)
    return report


def summarize(report: dict) -> str:
    g = report["gates"]
    lines = [f"\n{'=' * 60}", f"{report.get('label', '?')}", f"{'=' * 60}"]
    if "gsm8k" in g:
        r = g["gsm8k"]
        lines.append(f"  gsm8k accuracy     {r['accuracy']:>7.1%}  ({r['correct']}/{r['n']}, "
                     f"{r['no_answer']} unparseable, {r['seconds']}s)")
    if "tools" in g:
        r = g["tools"]
        lines.append(f"  tool well-formed   {r['well_formed']:>7.1%}")
        lines.append(f"  tool right choice  {r['right_tool']:>7.1%}  (n={r['n_positive']})")
        lines.append(f"  tool args correct  {r['args_ok']:>7.1%}")
        lines.append(f"  tool restraint     {r['restraint']:>7.1%}  (n={r['n_negative']}, "
                     f"no call when none applies)")
    if "voice" in g:
        r = g["voice"]
        lines.append(f"  pirate voice rate  {r['voice_rate']:>7.1%}")
        lines.append(f"  marker density     {r['density']:>7.2f}  per 100 words")
        lines.append(f"  mean reply length  {r['mean_words']:>7.0f}  words")
    return "\n".join(lines)


# (path into the report, label, higher-is-better)
METRICS = [
    ("gsm8k.accuracy", "gsm8k accuracy", True),
    ("tools.right_tool", "tool right choice", True),
    ("tools.args_ok", "tool args correct", True),
    ("tools.restraint", "tool restraint", True),
    ("voice.voice_rate", "pirate voice rate", True),
    ("voice.density", "marker density", True),
]


def _get(report: dict, dotted: str):
    gate, key = dotted.split(".")
    return report.get("gates", {}).get(gate, {}).get(key)


def compare(a: dict, b: dict) -> str:
    lines = [f"\n{'metric':<22}{a.get('label','A'):>14}{b.get('label','B'):>14}{'delta':>10}",
             "-" * 60]
    for dotted, label, _ in METRICS:
        x, y = _get(a, dotted), _get(b, dotted)
        if x is None or y is None:
            continue
        fmt = (lambda v: f"{v:.1%}") if dotted != "voice.density" else (lambda v: f"{v:.2f}")
        d = y - x
        mark = "" if abs(d) < 1e-9 else (" +" if d > 0 else " ")
        lines.append(f"{label:<22}{fmt(x):>14}{fmt(y):>14}{mark}{fmt(d):>8}")
    lines.append("\nA fine-tune ships only if voice went up and nothing else went down.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="GGUF to serve (spawns llama-server)")
    ap.add_argument("--server", help="Already-running llama-server URL")
    ap.add_argument("--label", default=None, help="Name for this run in the report")
    ap.add_argument("--persona", action="store_true", help="Apply the pirate system prompt")
    ap.add_argument("--system", default=None, help="Custom system prompt (overrides --persona)")
    ap.add_argument("--thinking", action="store_true", help="Let Qwen3 think before answering")
    ap.add_argument("--n-gsm8k", type=int, default=gsm8k.DEFAULT_N)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip", default="", help="Comma-separated gates to skip")
    ap.add_argument("--llama-port", type=int, default=8931)
    ap.add_argument("--ctx", type=int, default=4096, help="Context per request")
    ap.add_argument("--slots", type=int, default=None,
                    help="llama-server parallel slots (default: match --workers)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Diff two saved reports")
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        print(compare(a, b))
        return

    if not (args.model or args.server):
        raise SystemExit("pass --model <gguf> or --server <url>")

    system = args.system or (PERSONA if args.persona else None)
    label = args.label or (
        f"{Path(args.model).stem if args.model else args.server}"
        f"{'+persona' if system else ''}{'+thinking' if args.thinking else ''}"
    )

    llama = None
    try:
        if args.server:
            base = args.server
        else:
            llama = LlamaServer(
                port=args.llama_port, ctx=args.ctx, slots=args.slots or args.workers
            )
            print(f"loading {args.model} …")
            llama.start(args.model)
            base = llama.url

        client = ChatClient(base_url=base, system=system, enable_thinking=args.thinking)
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        report = evaluate(client, args.n_gsm8k, args.workers, skip)
        report.update({
            "label": label,
            "model": args.model or args.server,
            "system": system,
            "thinking": args.thinking,
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    finally:
        if llama is not None:
            llama.stop()

    print(summarize(report))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{label.replace('/', '_')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
