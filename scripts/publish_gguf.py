"""Publish a model's GGUF quants to a HuggingFace repo for on-device download.

The NanoBeard app downloads model files straight from
``huggingface.co/<repo>/resolve/main/<file>`` (see NanoBeard-App/installer.ts),
so "shipping" a GGUF == uploading it to a public HF model repo + a model card.

Dry-run by default (prints the plan); pass --push to actually upload.

Usage:
    # preview
    uv run python -m scripts.publish_gguf \
        --gguf-dir export/gguf/frigate-125m \
        --repo younissk/nanoBeard-frigate-125M-GGUF \
        --title "nanoBeard Frigate 125M (pirate chat)" \
        --params 125.9M --val-loss 2.882 \
        --base-model younissk/nanoBeard-frigate-125M-full

    # actually publish
    uv run python -m scripts.publish_gguf ... --push
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

CARD_TEMPLATE = """---
license: mit
language:
- en
library_name: gguf
pipeline_tag: text-generation
base_model: {base_model}
tags:
- llama.cpp
- gguf
- nanobeard
- pirate
- on-device
- mobile
---

# {title}

GGUF quantizations of a nanoBeard **Frigate** pirate chat model ({params} params),
for on-device inference with [llama.cpp](https://github.com/ggml-org/llama.cpp)
and the NanoBeard mobile app. SFT val loss ≈ {val_loss}.

Frigate's architecture (RoPE + SwiGLU + RMSNorm + per-head QK-norm, no biases,
tied embeddings) is Qwen3-equivalent, so these load with the upstream `qwen3`
GGUF arch — no custom runtime needed.

## Files

| file | quant | size | use |
|------|-------|------|-----|
{rows}

`Q4_K_M` is the default for phones (smallest + fastest). `Q8_0` is a near-lossless
fallback when you have the storage and want max quality.

## Chat format

Plain-text turns (no chat template). Build the prompt as:

```
User: <your message>
Pirate:
```

Turns are separated by a single newline; stop generation at the `<|endoftext|>`
token. Example with llama.cpp:

```bash
llama-completion -m {first_file} \\
  -p $'User: Tell me about the sea.\\nPirate:' -n 80 --temp 0.8 --top-k 40
```

## Tokenizer

Custom 16,384-token byte-level BPE (GPT-2-style pre-tokenizer), embedded in the
GGUF. No external tokenizer file required.
"""


def human(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    return f"{mb:.0f} MB" if mb >= 100 else f"{mb:.1f} MB"


def build_card(args, ggufs: list[Path]) -> str:
    rows = []
    for p in ggufs:
        quant = p.stem.split(".")[-1]  # frigate-125M.Q4_K_M -> Q4_K_M
        use = "default — phones" if "Q4" in quant else "quality fallback"
        rows.append(f"| `{p.name}` | {quant} | {human(p.stat().st_size)} | {use} |")
    return CARD_TEMPLATE.format(
        base_model=args.base_model,
        title=args.title,
        params=args.params,
        val_loss=args.val_loss,
        rows="\n".join(rows),
        first_file=ggufs[0].name,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--repo", required=True, help="e.g. younissk/nanoBeard-frigate-125M-GGUF")
    ap.add_argument("--title", required=True)
    ap.add_argument("--params", required=True, help="e.g. 125.9M")
    ap.add_argument("--val-loss", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push", action="store_true", help="Actually upload (default: dry-run)")
    args = ap.parse_args()

    gguf_dir = Path(args.gguf_dir)
    ggufs = sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        raise SystemExit(f"no .gguf files in {gguf_dir}")

    card = build_card(args, ggufs)
    card_path = gguf_dir / "README.md"
    card_path.write_text(card)

    total = sum(p.stat().st_size for p in ggufs)
    print(f"repo:   {args.repo}  ({'private' if args.private else 'public'})")
    print(f"files:  {len(ggufs)} GGUF + README.md  ({human(total)} total)")
    for p in ggufs:
        print(f"        {p.name:32s} {human(p.stat().st_size)}")
    print(f"card:   {card_path}")

    if not args.push:
        print("\n[dry-run] nothing uploaded. Re-run with --push to publish.")
        return

    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_file(
        path_or_fileobj=str(card_path), path_in_repo="README.md",
        repo_id=args.repo, repo_type="model",
    )
    for p in ggufs:
        print(f"  uploading {p.name} …")
        api.upload_file(
            path_or_fileobj=str(p), path_in_repo=p.name,
            repo_id=args.repo, repo_type="model",
        )
    print(f"\npublished -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
