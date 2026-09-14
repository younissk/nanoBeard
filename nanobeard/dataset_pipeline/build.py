"""Build a composed dataset from its recipe.

A *dataset* lives under `data/datasets/<name>/` and is defined by `recipe.json`:

    {
      "name": "PirateEnhanced",
      "description": "TinyPirateStories plus extra pirate corpora",
      "sources": [
        {"name": "tiny_stories_pirate", "weight": 1.0},
        {"name": "sea_shanties",        "weight": 1.0}
      ],
      "vocab_size": 8192
    }

Build = materialize each source -> combine -> train tokenizer -> tokenize to
train.bin/val.bin -> write metadata.json (provenance + token counts).

Run:
  uv run python -m nanobeard.dataset_pipeline.build --dataset tiny_pirate_stories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets

from nanobeard.dataset_pipeline.log import info, step
from nanobeard.dataset_pipeline.sources import REGISTRY, materialize
from nanobeard.dataset_pipeline.tokenize_corpus import encode_split
from nanobeard.dataset_pipeline.tokenize_ds import train_tokenizer

DATASETS_DIR = Path("data/datasets")
SPLITS = ("train", "validation")
SEED = 1337


def combine(parts: list[Dataset], weights: list[float]) -> Dataset:
    """Merge source splits. Equal weights -> plain concat (use every row once).
    Unequal weights -> probability-weighted interleave (no disk bloat; the
    smaller source is up/down-sampled to hit its target mixing share)."""
    if len(parts) == 1:
        return parts[0]
    if all(w == weights[0] for w in weights):
        return concatenate_datasets(parts)
    total = sum(weights)
    probs = [w / total for w in weights]
    return interleave_datasets(
        parts, probabilities=probs, stopping_strategy="all_exhausted", seed=SEED
    )


def read_recipe(dataset: str) -> dict:
    recipe_path = DATASETS_DIR / dataset / "recipe.json"
    if not recipe_path.exists():
        raise FileNotFoundError(f"No recipe at {recipe_path}")
    return json.loads(recipe_path.read_text())


def prepare_sources(dataset: str, force: bool = False) -> dict[str, DatasetDict]:
    """Materialize (download + piratize + cache) every source in the recipe.
    Does NOT train a tokenizer or write bins — that is `build()`.
    force=True rebuilds even cached sources (use after changing a builder)."""
    recipe = read_recipe(dataset)
    return {s["name"]: materialize(s["name"], force=force) for s in recipe["sources"]}


def build(dataset: str) -> dict:
    ds_dir = DATASETS_DIR / dataset
    recipe = read_recipe(dataset)

    src_specs = recipe["sources"]
    weights = [float(s.get("weight", 1.0)) for s in src_specs]
    step(f"Build dataset '{dataset}' — {len(src_specs)} source(s): {[s['name'] for s in src_specs]}")
    materialized = {s["name"]: materialize(s["name"]) for s in src_specs}

    # Combine per split, keeping only sources that actually carry that split.
    step("Combine sources")
    combined: dict[str, Dataset] = {}
    for split in SPLITS:
        present = [
            (materialized[s["name"]][split], w)
            for s, w in zip(src_specs, weights, strict=True)
            if split in materialized[s["name"]]
        ]
        if present:
            combined[split] = combine([p for p, _ in present], [w for _, w in present])
            info(f"{split}: {len(combined[split]):,} rows from {len(present)} source(s)")
    ds = DatasetDict(cast(dict, combined))
    if "train" not in ds:
        raise ValueError("No source provided a 'train' split")

    # Tokenizer (per-dataset; vocab_size is recipe-controlled for experiments).
    vocab_size = int(recipe.get("vocab_size", 8192))
    step(f"Train BPE tokenizer — vocab {vocab_size:,} over {len(ds['train']):,} train rows")
    tokenizer = train_tokenizer(ds["train"], vocab_size=vocab_size)
    tokenizer.save(str(ds_dir / "pirate_bpe.json"))
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    assert eot_id is not None, "Tokenizer must define <|endoftext|>"
    assert tokenizer.get_vocab_size() < 2**16, "vocab too large for uint16"
    info(f"tokenizer trained — {tokenizer.get_vocab_size():,} tokens -> {ds_dir / 'pirate_bpe.json'}")

    step("Encode splits → uint16 .bin")
    train_tokens = encode_split(ds["train"], tokenizer, eot_id, ds_dir / "train.bin")
    info(f"train.bin: {train_tokens:,} tokens")
    val_tokens = (
        encode_split(ds["validation"], tokenizer, eot_id, ds_dir / "val.bin")
        if "validation" in ds
        else 0
    )
    info(f"val.bin: {val_tokens:,} tokens")

    total_rows = sum(len(materialized[s["name"]].get("train", [])) for s in src_specs)
    meta = {
        "name": recipe.get("name", dataset),
        "description": recipe.get("description", ""),
        "sources": [
            {
                "name": s["name"],
                "weight": float(s.get("weight", 1.0)),
                "origin": REGISTRY[s["name"]]["origin"],
                "train_rows": len(materialized[s["name"]].get("train", [])),
                "row_share": round(len(materialized[s["name"]].get("train", [])) / total_rows, 4)
                if total_rows
                else 0.0,
            }
            for s in src_specs
        ],
        "vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    step(f"Done '{dataset}' — {train_tokens:,} train + {val_tokens:,} val tokens, vocab {meta['vocab_size']:,}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Name under data/datasets/, e.g. tiny_pirate_stories")
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Only download + piratize + cache the recipe's sources; skip tokenizer/bins.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --sources-only: rebuild cached sources instead of reusing them.",
    )
    args = parser.parse_args()

    if args.sources_only:
        prepared = prepare_sources(args.dataset, force=args.force)
        for name, ds in prepared.items():
            print(f"{name}: { {k: len(v) for k, v in ds.items()} }")
        print(f"\nPrepared {len(prepared)} source(s). Run without --sources-only to build the dataset.")
        return

    meta = build(args.dataset)
    print(json.dumps(meta, indent=2))
    print(f"\nBuilt -> {DATASETS_DIR / args.dataset}")


if __name__ == "__main__":
    main()
