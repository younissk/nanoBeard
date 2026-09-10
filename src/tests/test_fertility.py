"""Tokenizer fertility measurement.

The numbers this tool prints are used to decide `block_size`, so a wrong metric
is worse than no metric: it would argue for a context length on false evidence.
These tests pin the arithmetic and the corpus invariants, not the values — the
values change whenever the tokenizer is retrained, and should.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from nanobeard.fertility import (
    BASELINE,
    BLOCK_SIZE,
    DEFAULT_CORPUS,
    Measurement,
    digit_probe,
    discover_tokenizers,
    load_corpus,
    measure,
)


@pytest.fixture(scope="module")
def tok(tmp_path_factory) -> Tokenizer:
    """A BPE with room for real merges.

    conftest's `synthetic_tokenizer` caps out at vocab 128, which the byte
    alphabet alone fills — it learns zero merges, so every text measures at
    exactly 1 token per character and no fertility difference can show up.
    Testing the metric needs a tokenizer that actually compresses something.
    """
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    corpus = [
        "ahoy matey treasure rum sea",
        "the pirate sails the seven seas",
        "yarr a chest of gold doubloons",
        "she said look out for sharks",
    ] * 50
    tok.train_from_iterator(
        corpus,
        trainer=BpeTrainer(
            vocab_size=600,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        ),
        length=len(corpus),
    )
    out = tmp_path_factory.mktemp("fert") / "tok.json"
    tok.save(str(out))
    return Tokenizer.from_file(str(out))


# ----- corpus -----
def test_corpus_ships_with_the_package():
    assert DEFAULT_CORPUS.exists(), "corpus must not depend on a built dataset"


def test_every_row_has_a_domain_text_and_note():
    for row in load_corpus():
        assert row["domain"] and row["text"]
        assert row["note"], f"{row['domain']} needs a note saying why it is in the set"


def test_domains_are_unique():
    domains = [r["domain"] for r in load_corpus()]
    assert len(domains) == len(set(domains))


def test_the_baseline_domain_is_present():
    # Every ratio in the report is computed against it; without it they are all 0.
    assert BASELINE in [r["domain"] for r in load_corpus()]


def test_exactly_one_row_is_the_episode():
    # The episode budget section assumes a single unit to report on.
    assert sum(1 for r in load_corpus() if r.get("episode")) == 1


def test_samples_are_long_enough_to_mean_something():
    # A 20-character sample measures nothing but its own edges.
    for row in load_corpus():
        assert len(row["text"]) >= 200, row["domain"]


# ----- metric arithmetic -----
def test_ratios_are_reciprocal():
    m = Measurement(domain="d", chars=100, words=20, tokens=25, single_char_tokens=5)
    assert m.tokens_per_char == 0.25
    assert m.chars_per_token == 4.0
    assert m.tokens_per_word == 1.25
    assert m.single_char_frac == 0.2


def test_empty_text_does_not_divide_by_zero():
    m = Measurement(domain="d", chars=0, words=0, tokens=0, single_char_tokens=0)
    assert (m.tokens_per_char, m.chars_per_token, m.tokens_per_word, m.single_char_frac) == (
        0.0, 0.0, 0.0, 0.0
    )


def test_measure_counts_chars_and_words_from_the_raw_text(tok):
    text = "ahoy matey treasure"
    m = measure(tok, "d", text)
    assert m.chars == len(text)
    assert m.words == 3
    assert m.tokens == len(tok.encode(text).ids)


def test_single_char_tokens_are_counted_per_occurrence(tok):
    # Counting distinct token strings instead of occurrences would under-report
    # exactly the case the metric exists for: long runs of fallback bytes.
    text = "zzzzzzzzzz"
    m = measure(tok, "d", text)
    assert m.single_char_tokens > 1
    assert m.single_char_tokens <= m.tokens


def test_in_domain_text_is_cheaper_than_out_of_domain(tok):
    # The synthetic tokenizer is trained on pirate words only, so this holds for
    # it as well as for the real one — it is the property the report measures.
    in_domain = measure(tok, "in", "ahoy matey treasure rum sea " * 8)
    out_domain = measure(tok, "out", "3.14159 * x^2 + 42 = {} " * 8)
    assert out_domain.tokens_per_char > in_domain.tokens_per_char


# ----- probes -----
def test_digit_probe_returns_pieces_for_every_sample(tok):
    probe = digit_probe(tok)
    assert probe
    for sample, pieces in probe:
        assert pieces, sample
        assert "".join(pieces).replace("Ġ", " ").strip() != "" or not sample.strip()


def test_digit_probe_accepts_custom_samples(tok):
    (sample, pieces), = digit_probe(tok, samples=("1234",))
    assert sample == "1234"
    assert len(pieces) >= 1


# ----- discovery -----
def test_discover_returns_nothing_for_a_missing_dir(tmp_path):
    assert discover_tokenizers(tmp_path / "absent") == []


def test_discover_finds_one_tokenizer_per_dataset(tmp_path):
    for name in ("alpha", "beta"):
        d = tmp_path / name
        d.mkdir()
        (d / "pirate_bpe.json").write_text("{}")
    (tmp_path / "gamma").mkdir()  # no tokenizer -> not listed
    assert [p.parent.name for p in discover_tokenizers(tmp_path)] == ["alpha", "beta"]


def test_block_size_matches_the_configs():
    # If frigate's block_size moves, this report is measuring the wrong window.
    cfg = Path("configs/frigate.py").read_text()
    assert f"block_size={BLOCK_SIZE}" in cfg
