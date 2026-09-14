"""Source registry — the reusable building blocks datasets are composed from.

A *source* is one piratized corpus (a `DatasetDict` with `text` columns), built
once and cached under `data/sources/<name>/`. Datasets (see `build.py`) combine
one or more sources into a tokenizer + train.bin/val.bin.

Add a new source by writing a builder fn and registering it in REGISTRY.

Run (build/cache a single source on its own):
  uv run python -m nanobeard.dataset_pipeline.sources --source tiny_stories_pirate
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

SOURCES_DIR = Path("data/sources")

# Subsample size for the cosmopedia 'stories' subset (~5M rows total).
# materialize() caches by name only — bump this then rebuild with `--force`.
COSMOPEDIA_STORIES_ROWS = 1_500_000

# Public-domain pirate-themed books pulled from Project Gutenberg (name -> ebook id).
GUTENBERG_BOOKS = {
    "pyle_book_of_pirates": 973,
    "treasure_island": 120,
    "general_history_of_pyrates": 40580,
    "peter_pan": 16,
    "coral_island": 646,
}


class SourceSpec(TypedDict):
    builder: Callable[[], DatasetDict]
    origin: str


def build_tiny_stories_pirate() -> DatasetDict:
    """roneneldan/TinyStories, run through the `arrr` piratizer."""
    from nanobeard.dataset_pipeline.piratize import piratize

    raw = load_dataset("roneneldan/TinyStories")
    return DatasetDict({split: piratize(ds, str(split)) for split, ds in raw.items()})


def build_cosmopedia_wikihow(val_rows: int = 5000) -> DatasetDict:
    """HuggingFaceTB/cosmopedia 'wikihow' subset, piratized.

    Only ships a train split (179k synthetic how-to articles), so we carve a
    small validation set for parity with the other sources.
    """
    from nanobeard.dataset_pipeline.piratize import piratize

    raw = load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train")
    raw = raw.select_columns(["text"])  # drop prompt/seed_data/etc; keep schema lean
    split = raw.train_test_split(test_size=val_rows, seed=1337)
    return DatasetDict(
        {
            "train": piratize(split["train"], "train"),
            "validation": piratize(split["test"], "validation"),
        }
    )


_STORIES_TOTAL_SHARDS = 43  # HuggingFaceTB/cosmopedia 'stories' parquet shards
_STORIES_ROWS_PER_SHARD = 116_000  # ~5M / 43


def _stories_shard_files(n_rows: int) -> list[str]:
    """Pick just enough parquet shards to cover n_rows, spread evenly across the
    corpus (the subset is topic-ordered, so adjacent shards share themes)."""
    k = min(_STORIES_TOTAL_SHARDS, math.ceil(n_rows / _STORIES_ROWS_PER_SHARD) + 2)
    if k >= _STORIES_TOTAL_SHARDS:
        ids = list(range(_STORIES_TOTAL_SHARDS))
    elif k <= 1:
        ids = [0]
    else:
        ids = sorted({round(i * (_STORIES_TOTAL_SHARDS - 1) / (k - 1)) for i in range(k)})
    return [f"data/stories/train-{i:05d}-of-{_STORIES_TOTAL_SHARDS:05d}.parquet" for i in ids]


def build_cosmopedia_stories(
    n_rows: int = COSMOPEDIA_STORIES_ROWS, val_rows: int = 2000, seed: int = 1337
) -> DatasetDict:
    """A subsample of HuggingFaceTB/cosmopedia 'stories' (~5M rows), piratized.

    Downloads only the shards needed to cover n_rows (resumable via the HF
    cache — far more robust than row-streaming, which dies on any network blip),
    then shuffles + selects locally. Arrow is memory-mapped, so RAM stays low.
    """
    from nanobeard.dataset_pipeline.piratize import piratize

    full = load_dataset(
        "HuggingFaceTB/cosmopedia", data_files=_stories_shard_files(n_rows), split="train"
    )
    full = full.select_columns(["text"])
    sample = full.shuffle(seed=seed).select(range(min(n_rows, len(full))))
    split = sample.train_test_split(test_size=val_rows, seed=seed)
    return DatasetDict(
        {"train": piratize(split["train"], "train"), "validation": piratize(split["test"], "validation")}
    )


def _build_cosmopedia_subset(config: str, val_rows: int, seed: int = 1337) -> DatasetDict:
    """Full HuggingFaceTB/cosmopedia <config> subset, piratized.

    Uses the named config (downloads every shard) rather than the selective
    shard-picking of build_cosmopedia_stories — we want the whole subset here.
    """
    from nanobeard.dataset_pipeline.piratize import piratize

    full = load_dataset("HuggingFaceTB/cosmopedia", config, split="train")
    full = full.select_columns(["text"])  # keep schema lean
    split = full.train_test_split(test_size=val_rows, seed=seed)
    return DatasetDict(
        {"train": piratize(split["train"], "train"), "validation": piratize(split["test"], "validation")}
    )


def build_cosmopedia_stories_full(val_rows: int = 5000) -> DatasetDict:
    """ALL of HuggingFaceTB/cosmopedia 'stories' (~5M rows), piratized.
    (build_cosmopedia_stories is the subsampled variant.)"""
    return _build_cosmopedia_subset("stories", val_rows)


def build_cosmopedia_stanford(val_rows: int = 5000) -> DatasetDict:
    """ALL of HuggingFaceTB/cosmopedia 'stanford' subset, piratized."""
    return _build_cosmopedia_subset("stanford", val_rows)


WIKIPEDIA_CONFIG = "20231101.en"
_WIKI_SHARDS = 41  # wikimedia/wikipedia 20231101.en parquet shards
_WIKI_PIRATE_RE = re.compile(r"\bpirate", re.I)  # pirate / pirates / pirated


def _wikipedia_shard_file(i: int) -> str:
    return f"{WIKIPEDIA_CONFIG}/train-{i:05d}-of-{_WIKI_SHARDS:05d}.parquet"


def build_wikipedia_pirate(val_rows: int = 500, seed: int = 1337) -> DatasetDict:
    """English Wikipedia articles mentioning 'pirate', NOT piratized.

    No server-side substring filter exists for wikimedia/wikipedia, so every
    shard's text must be scanned. To survive network drops / laptop sleep on the
    ~1h pass, this is RESUMABLE: it streams the 41 shards one at a time and only
    commits a shard's matches (+ marks it done) once that shard finishes whole.
    A crash resumes from the next unscanned shard rather than restarting. Still
    streaming — no full local copy of Wikipedia, and at most one shard's matches
    are held in memory. Kept as authentic prose (like gutenberg_books).
    """
    from nanobeard.dataset_pipeline.log import info

    scan_dir = source_dir("wikipedia_pirate") / "_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    matches_path = scan_dir / "matches.jsonl"
    done_path = scan_dir / "done_shards.json"

    done = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()
    kept_total = sum(1 for _ in matches_path.open(encoding="utf-8")) if matches_path.exists() else 0
    if done:
        info(f"wikipedia resume: {len(done)}/{_WIKI_SHARDS} shards already scanned, {kept_total:,} kept so far")

    for i in range(_WIKI_SHARDS):
        if i in done:
            continue
        info(f"wikipedia shard {i + 1}/{_WIKI_SHARDS}: streaming + scanning")
        shard = load_dataset(
            "wikimedia/wikipedia", data_files=_wikipedia_shard_file(i), split="train", streaming=True
        )
        buf = [row["text"] for row in shard if _WIKI_PIRATE_RE.search(row["text"])]
        # Commit only after the WHOLE shard streamed (a mid-shard crash redoes
        # this shard cleanly — no half-written duplicates).
        with matches_path.open("a", encoding="utf-8") as out:
            for text in buf:
                out.write(json.dumps({"text": text}) + "\n")
        done.add(i)
        done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
        kept_total += len(buf)
        info(f"wikipedia shard {i + 1}/{_WIKI_SHARDS} done: +{len(buf):,} matches ({kept_total:,} total)")

    info(f"wikipedia scan complete: {kept_total:,} articles kept across {_WIKI_SHARDS} shards")
    ds = load_dataset("json", data_files=str(matches_path), split="train")
    split = ds.train_test_split(test_size=val_rows, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


_PG_START = re.compile(r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK", re.I)


def _fetch_gutenberg(book_id: int) -> str:
    """Download a book's canonical UTF-8 plain text (PG 403s without a User-Agent)."""
    url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
    req = urllib.request.Request(url, headers={"User-Agent": "nanoBeard/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        return resp.read().decode("utf-8", errors="replace")


def _strip_pg(text: str) -> str:
    """Drop the Project Gutenberg license header/footer around the actual book."""
    start = m.end() if (m := _PG_START.search(text)) else 0
    end = m.start() if (m := _PG_END.search(text)) else len(text)
    return text[start:end]


def _paragraphs(text: str, min_chars: int = 200) -> list[str]:
    """Split into paragraphs on blank lines, unwrap hard-wrapped lines, drop short bits."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= min_chars:
            out.append(para)
    return out


def build_gutenberg_books(val_rows: int = 500, min_chars: int = 200) -> DatasetDict:
    """5 public-domain pirate books from Project Gutenberg, paragraph-chunked.

    NOT piratized — kept as authentic source prose (unlike the cosmopedia/
    TinyStories sources). Each paragraph becomes one `text` row.
    """
    texts: list[str] = []
    for book_id in GUTENBERG_BOOKS.values():
        texts.extend(_paragraphs(_strip_pg(_fetch_gutenberg(book_id)), min_chars))

    ds = Dataset.from_dict({"text": texts}).train_test_split(test_size=val_rows, seed=1337)
    return DatasetDict({"train": ds["train"], "validation": ds["test"]})


REGISTRY: dict[str, SourceSpec] = {
    "tiny_stories_pirate": {
        "builder": build_tiny_stories_pirate,
        "origin": "roneneldan/TinyStories piratized via arrr",
    },
    "cosmopedia_wikihow": {
        "builder": build_cosmopedia_wikihow,
        "origin": "HuggingFaceTB/cosmopedia (wikihow subset) piratized via arrr",
    },
    "cosmopedia_stories": {
        "builder": build_cosmopedia_stories,
        "origin": f"HuggingFaceTB/cosmopedia (stories subset, {COSMOPEDIA_STORIES_ROWS} subsample) piratized via arrr",
    },
    "cosmopedia_stories_full": {
        "builder": build_cosmopedia_stories_full,
        "origin": "HuggingFaceTB/cosmopedia (full stories subset) piratized via arrr",
    },
    "cosmopedia_stanford": {
        "builder": build_cosmopedia_stanford,
        "origin": "HuggingFaceTB/cosmopedia (full stanford subset) piratized via arrr",
    },
    "wikipedia_pirate": {
        "builder": build_wikipedia_pirate,
        "origin": "wikimedia/wikipedia 20231101.en, articles matching /\\bpirate/i, NOT piratized",
    },
    "gutenberg_books": {
        "builder": build_gutenberg_books,
        "origin": "Project Gutenberg, NOT piratized: "
        + ", ".join(f"{name} (#{bid})" for name, bid in GUTENBERG_BOOKS.items()),
    },
}


def source_dir(name: str) -> Path:
    return SOURCES_DIR / name


def is_cached(name: str) -> bool:
    return (source_dir(name) / "dataset_dict.json").exists()


def materialize(name: str, force: bool = False) -> DatasetDict:
    """Return a source's DatasetDict, building + caching it on first use."""
    from nanobeard.dataset_pipeline.log import info, step

    if name not in REGISTRY:
        raise KeyError(f"Unknown source {name!r}. Known: {sorted(REGISTRY)}")

    out = source_dir(name)
    if is_cached(name) and not force:
        ds = cast(DatasetDict, load_from_disk(str(out)))
        info(f"source [bold]{name}[/]: cache hit — { {k: len(v) for k, v in ds.items()} }")
        return ds

    spec = REGISTRY[name]
    step(f"Materialize source '{name}'{' (forced rebuild)' if force else ''}")
    info(f"origin: {spec['origin']}")
    ds = spec["builder"]()
    info(f"source [bold]{name}[/]: built — { {k: len(v) for k, v in ds.items()} }; caching -> {out}")
    if out.exists():  # clear a stale cache so old shards can't linger
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    (out / "source.json").write_text(
        json.dumps(
            {
                "name": name,
                "origin": spec["origin"],
                "rows": {split: len(part) for split, part in ds.items()},
            },
            indent=2,
        )
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = parser.parse_args()

    ds = materialize(args.source, force=args.force)
    print(f"{args.source}: { {k: len(v) for k, v in ds.items()} } -> {source_dir(args.source)}")


if __name__ == "__main__":
    main()
