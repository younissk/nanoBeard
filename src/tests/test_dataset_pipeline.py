"""Dataset pipeline — source registry, recipe consistency, combine(), builders.

All offline: no corpus is downloaded. The cosmopedia builder is exercised with
a monkeypatched `load_dataset` so its transform logic (column pruning, val
carve, piratize per split) is verified without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from nanobeard.dataset_pipeline import build, sources
from nanobeard.dataset_pipeline.tokenize_corpus import encode_split
from nanobeard.dataset_pipeline.tokenize_ds import train_tokenizer
from tests.conftest import REPO_ROOT

DATASETS_DIR = REPO_ROOT / "data" / "datasets"


# ----- registry -----


def test_registry_non_empty():
    assert sources.REGISTRY


def test_registry_entries_well_formed():
    for name, spec in sources.REGISTRY.items():
        assert callable(spec["builder"]), f"{name} builder not callable"
        assert isinstance(spec["origin"], str) and spec["origin"], f"{name} missing origin"


def test_expected_sources_registered():
    assert "tiny_stories_pirate" in sources.REGISTRY
    assert "cosmopedia_wikihow" in sources.REGISTRY
    assert "cosmopedia_stories" in sources.REGISTRY
    assert "gutenberg_books" in sources.REGISTRY


# ----- recipes -----


def _recipe_paths() -> list[Path]:
    return sorted(DATASETS_DIR.glob("*/recipe.json"))


def test_recipes_exist():
    assert _recipe_paths(), "no dataset recipes found"


@pytest.mark.parametrize("recipe_path", _recipe_paths(), ids=lambda p: p.parent.name)
def test_recipe_valid_and_sources_registered(recipe_path: Path):
    recipe = json.loads(recipe_path.read_text())
    assert recipe.get("sources"), f"{recipe_path} has no sources"
    assert isinstance(recipe.get("vocab_size", 8192), int)
    for s in recipe["sources"]:
        assert s["name"] in sources.REGISTRY, (
            f"{recipe_path} references unregistered source {s['name']!r}"
        )
        assert float(s.get("weight", 1.0)) > 0


# ----- combine() -----


def _text_ds(words: list[str]) -> Dataset:
    return Dataset.from_dict({"text": words})


def test_prepare_sources_materializes_all_recipe_sources(monkeypatch):
    """--sources-only path: every recipe source is materialized, nothing else."""
    called = []

    def fake_materialize(name, force=False):
        called.append(name)
        return build.DatasetDict({"train": _text_ds([f"{name} row"])})

    monkeypatch.setattr(build, "materialize", fake_materialize)

    recipe = build.read_recipe("pirate_enhanced")
    expected = [s["name"] for s in recipe["sources"]]

    prepared = build.prepare_sources("pirate_enhanced")

    assert called == expected
    assert set(prepared) == set(expected)


def test_combine_single_returns_input():
    a = _text_ds(["a", "b"])
    assert build.combine([a], [1.0]) is a


def test_combine_equal_weights_concatenates():
    a, b = _text_ds(["a", "b"]), _text_ds(["c"])
    out = build.combine([a, b], [1.0, 1.0])
    assert len(out) == 3
    assert set(out["text"]) == {"a", "b", "c"}


def test_combine_unequal_weights_interleaves():
    a = _text_ds([f"a{i}" for i in range(20)])
    b = _text_ds([f"b{i}" for i in range(5)])
    out = build.combine([a, b], [3.0, 1.0])
    assert len(out) > 0
    assert out.column_names == ["text"]


# ----- encode_split (long-row safety) -----


def test_encode_split_handles_long_rows(tmp_path):
    """Rows far longer than the old 500-token/row estimate must not overflow."""
    # A long, varied corpus so BPE merges don't collapse it to a few tokens.
    long_text = " ".join(f"word{i}" for i in range(4000))
    ds = _text_ds([long_text] * 5)
    tok = train_tokenizer(ds, vocab_size=300)
    eot = tok.token_to_id("<|endoftext|>")
    assert eot is not None

    out = tmp_path / "train.bin"
    total = encode_split(ds, tok, eot, out)

    arr = np.memmap(str(out), dtype=np.uint16, mode="r")
    assert len(arr) == total
    assert total > 5 * 500  # would have overflowed the old fixed buffer
    assert int(arr.max()) == eot or eot < tok.get_vocab_size()
    assert int((arr == eot).sum()) == 5  # one EOT terminator per row


# ----- piratize (streamed, memory-bounded) -----


def test_piratize_preserves_rows_and_order():
    from nanobeard.dataset_pipeline.piratize import piratize

    rows = [f"hello there friend number {i}" for i in range(25)]
    out = piratize(_text_ds(rows), "test", chunk_size=10)  # spans multiple chunks

    assert out.column_names == ["text"]
    assert len(out) == len(rows)
    # arrr is deterministic and changes the text (e.g. "friend" -> "matey")
    assert out["text"] != rows
    assert all(isinstance(t, str) and t for t in out["text"])


# ----- cosmopedia builder (offline) -----


def test_build_cosmopedia_wikihow_offline(monkeypatch):
    captured = {}

    def fake_load_dataset(path, name=None, split=None, **kw):
        captured["path"], captured["name"], captured["split"] = path, name, split
        # mimic cosmopedia's real schema — extra columns the builder must drop
        rows = 40
        return Dataset.from_dict(
            {
                "text": [f"how to do thing {i}" for i in range(rows)],
                "prompt": ["p"] * rows,
                "seed_data": ["s"] * rows,
                "format": ["f"] * rows,
                "audience": ["a"] * rows,
                "text_token_length": [5] * rows,
            }
        )

    monkeypatch.setattr(sources, "load_dataset", fake_load_dataset)

    ds = sources.build_cosmopedia_wikihow(val_rows=8)

    # hit the right dataset/config/split
    assert captured == {"path": "HuggingFaceTB/cosmopedia", "name": "wikihow", "split": "train"}
    # carved validation, kept train, text-only schema
    assert set(ds.keys()) == {"train", "validation"}
    assert len(ds["validation"]) == 8
    assert len(ds["train"]) == 32
    assert ds["train"].column_names == ["text"]
    assert ds["validation"].column_names == ["text"]


def test_stories_shard_files_cover_target_and_spread():
    """Shard selection: enough to cover n_rows, spread across the corpus, valid paths."""
    # small target -> few shards, drawn from across the range (not just the start)
    few = sources._stories_shard_files(200_000)
    assert 1 < len(few) < sources._STORIES_TOTAL_SHARDS
    assert all(f.startswith("data/stories/train-") and f.endswith("-of-00043.parquet") for f in few)
    assert few == sorted(set(few))  # unique + ordered

    # target beyond the corpus -> every shard, no duplicates
    allf = sources._stories_shard_files(10_000_000)
    assert len(allf) == sources._STORIES_TOTAL_SHARDS


def _patch_stories_load(monkeypatch, n_available: int = 100) -> dict:
    """Monkeypatch load_dataset to a small in-memory (non-streaming) Dataset."""
    captured: dict = {}

    def fake_load_dataset(path, data_files=None, split=None, **kw):
        captured.update(path=path, data_files=data_files, split=split)
        rows = n_available
        return Dataset.from_dict(
            {
                "text": [f"a pirate story {i}" for i in range(rows)],
                "prompt": ["p"] * rows,
                "audience": ["a"] * rows,
            }
        )

    monkeypatch.setattr(sources, "load_dataset", fake_load_dataset)
    return captured


def test_build_cosmopedia_stories_structure(monkeypatch):
    """Full builder: selects shards, samples n_rows, carves val, text-only schema."""
    captured = _patch_stories_load(monkeypatch)

    ds = sources.build_cosmopedia_stories(n_rows=30, val_rows=6)

    # downloaded by explicit shard file list, not the whole subset
    assert captured["path"] == "HuggingFaceTB/cosmopedia"
    assert isinstance(captured["data_files"], list) and captured["data_files"]
    assert captured["split"] == "train"
    # sampled n_rows, carved val, dropped extra columns
    assert set(ds.keys()) == {"train", "validation"}
    assert len(ds["train"]) == 24
    assert len(ds["validation"]) == 6
    assert ds["train"].column_names == ["text"]


# ----- gutenberg builder (offline) -----


def test_strip_pg_removes_header_and_footer():
    raw = (
        "license header blah blah\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TREASURE ISLAND ***\n"
        "real book body here\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TREASURE ISLAND ***\n"
        "footer license blah"
    )
    out = sources._strip_pg(raw)
    assert "real book body here" in out
    assert "license header" not in out
    assert "footer license" not in out


def test_strip_pg_passthrough_when_no_markers():
    assert sources._strip_pg("no markers at all") == "no markers at all"


def test_paragraphs_unwraps_and_filters():
    text = (
        "Short line.\n\n"  # below min_chars -> dropped
        "This is a long paragraph that has been hard-wrapped\n"
        "across several lines as Project Gutenberg does, and it\n"
        "should be joined into one row with single spaces."
    )
    paras = sources._paragraphs(text, min_chars=40)
    assert len(paras) == 1
    assert "\n" not in paras[0]
    assert "hard-wrapped across several" in paras[0]


def test_build_gutenberg_books_offline(monkeypatch):
    fetched = []

    def fake_fetch(book_id: int) -> str:
        fetched.append(book_id)
        body = "\n\n".join(
            f"Paragraph {i} of book {book_id} with enough characters to survive the filter." * 2
            for i in range(20)
        )
        return f"hdr\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n{body}\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nftr"

    monkeypatch.setattr(sources, "_fetch_gutenberg", fake_fetch)

    ds = sources.build_gutenberg_books(val_rows=10, min_chars=40)

    # fetched every registered book id
    assert fetched == list(sources.GUTENBERG_BOOKS.values())
    assert set(ds.keys()) == {"train", "validation"}
    assert len(ds["validation"]) == 10
    assert ds["train"].column_names == ["text"]
    # NOT piratized: source prose passes through unchanged, no PG boilerplate
    joined = " ".join(ds["train"]["text"])
    assert "Paragraph" in joined
    assert "PROJECT GUTENBERG" not in joined
