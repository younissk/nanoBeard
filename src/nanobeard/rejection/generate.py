"""Generate N candidate replies per prompt, for rejection sampling.

Run (llama-server must already be serving the GGUF):

    llama-server -m export/gguf/frigate-360m/frigate-360M.Q8_0.gguf \
        --host 127.0.0.1 --port 8899 -c 4096 -np 8 --no-webui

    uv run python -m nanobeard.rejection.generate \
        --out runs/rejection/frigate-360m.jsonl --n 10

Prompt format is the SFT transcript from `nanobeard.sft_data`:

    User: <text>\\nPirate: <reply><eos>

The rendered prompt ends with ``BOT_PREFIX`` *including its trailing space*.
This is load-bearing: the tokenizer is byte-level BPE, so "Pirate:" + " reply"
and "Pirate: " + "reply" are different token sequences, and only the latter
matches training. Ending the prompt at "Pirate:" makes the model degenerate.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nanobeard.sft_data import BOT_PREFIX, TURN_SEP, USER_PREFIX

DEFAULT_PROMPTS = Path(__file__).parent / "prompts.jsonl"
DEFAULT_SERVER = "http://127.0.0.1:8899"
# Stop before the model hallucinates the next user turn.
STOPS = [TURN_SEP + USER_PREFIX, USER_PREFIX.rstrip()]


def render_prompt(turns: list[dict]) -> str:
    """Transcript -> raw completion prompt, ending at 'Pirate: '."""
    parts = []
    for i, t in enumerate(turns):
        sep = "" if i == 0 else TURN_SEP
        prefix = USER_PREFIX if t["role"] == "user" else BOT_PREFIX
        parts.append(sep + prefix + t["text"])
    return "".join(parts) + TURN_SEP + BOT_PREFIX


def load_prompts(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def complete(server: str, prompt: str, *, seed: int, temperature: float,
             top_p: float, top_k: int, n_predict: int, timeout: float = 120.0) -> dict:
    payload = {
        "prompt": prompt, "n_predict": n_predict, "temperature": temperature,
        "top_p": top_p, "top_k": top_k, "seed": seed, "stop": STOPS,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        f"{server}/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--n", type=int, default=10, help="Samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--n-predict", type=int, default=120)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8, help="Match llama-server -np")
    ap.add_argument("--model-label", default="frigate-360M-Q8_0")
    ap.add_argument("--resume", action="store_true",
                    help="Keep rows already in --out and only fill the gaps")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    jobs = [(rec, k) for rec in prompts for k in range(args.n)]

    # Resume: llama-server can die mid-run (its prompt cache defaults to 8 GiB,
    # which OOM-kills it on a small machine). Without this, one death costs the
    # entire run rather than the missing rows.
    done: dict[tuple[str, int], dict] = {}
    out_path = Path(args.out)
    if args.resume and out_path.is_file():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue        # retry failures
                done[(r["id"], r["sample"])] = r
        jobs = [(rec, k) for rec, k in jobs if (rec["id"], k) not in done]
        print(f"resuming: {len(done)} rows kept, {len(jobs)} to generate")

    print(f"{len(prompts)} prompts x {args.n} samples = "
          f"{len(jobs)} generations{' (gaps only)' if args.resume else ''}")

    params = {
        "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "n_predict": args.n_predict,
    }

    def run(job):
        rec, k = job
        prompt = render_prompt(rec["turns"])
        seed = args.seed_base + k
        try:
            d = complete(args.server, prompt, seed=seed, **params)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {"id": rec["id"], "category": rec["category"], "sample": k,
                    "seed": seed, "error": f"{type(e).__name__}: {e}"}
        return {
            "id": rec["id"], "category": rec["category"], "sample": k, "seed": seed,
            "turns": rec["turns"], "prompt": prompt,
            "completion": d.get("content", ""),
            "n_tokens": d.get("tokens_predicted"),
            "stop_type": d.get("stop_type"),
            "model": args.model_label, "params": params,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    t0 = time.time()
    n_done, n_err = 0, 0
    with tmp.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in pool.map(run, jobs):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if "error" in row:
                n_err += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(jobs)}  ({n_done/(time.time()-t0):.1f}/s, {n_err} errors)")

    tmp.replace(out_path)
    dt = max(time.time() - t0, 1e-6)
    total = len(done) + n_done
    print(f"Wrote {total} rows -> {out_path}  ({dt:.0f}s, {n_done/dt:.1f}/s, {n_err} errors)")
    if n_err:
        print(f"  {n_err} failed — re-run with --resume to fill them in.")


if __name__ == "__main__":
    main()
