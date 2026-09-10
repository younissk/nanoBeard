"""How many tokens does a domain cost under nanoBeard's tokenizer?

    uv run python -m nanobeard.fertility                      # every dataset tokenizer
    uv run python -m nanobeard.fertility --tokenizer <a.json> # just one
    uv run python -m nanobeard.fertility --text "3x + 7 = 22" # ad-hoc string
    uv run python -m nanobeard.fertility --reference Qwen/Qwen3-0.6B   # control

The question this answers, from TODO.md step 0: `pirate_bpe.json` is a byte-level
BPE trained on pirate stories and wikihow. Every merge it learned was paid for by
prose. Math and code get no merges, fall back toward one token per byte, and burn
context the model does not have — block_size is 512.

Fertility here is **tokens per character**, and the number that matters is the
ratio against plain English: 1.0 means a domain costs what prose costs, 2.0 means
the same passage takes twice the context. `--text` and the digit probe exist for
when a number looks wrong and you want to see the actual pieces.

Nothing downloads unless you ask for it: the corpus ships next to this file, and
`--reference` is the one flag that touches the network. Use it to separate "our
tokenizer is bad at math" from "math is just denser than prose" — a modern
tokenizer with digit handling is the control.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_CORPUS = Path(__file__).parent / "fertility_corpus.jsonl"
DATASETS_DIR = Path("data/datasets")
BASELINE = "english_prose"
# frigate/galleon pretrain at 512; the whole point of the measurement.
BLOCK_SIZE = 512
_WORD = re.compile(r"\S+")


@dataclass
class Measurement:
    domain: str
    chars: int
    words: int
    tokens: int
    single_char_tokens: int

    @property
    def tokens_per_char(self) -> float:
        return self.tokens / self.chars if self.chars else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens if self.tokens else 0.0

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words if self.words else 0.0

    @property
    def single_char_frac(self) -> float:
        """Share of tokens that decode to one character.

        A byte-level BPE never emits <unk>; it silently degrades to near-byte
        pieces instead. This is the visible symptom of that.
        """
        return self.single_char_tokens / self.tokens if self.tokens else 0.0


def load_corpus(path: Path = DEFAULT_CORPUS) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def measure(tok: Tokenizer, domain: str, text: str) -> Measurement:
    enc = tok.encode(text)
    # Count per occurrence, by id — a repeated single-char token costs every time.
    singles = sum(1 for i in enc.ids if len(tok.decode([i]).strip()) <= 1)
    return Measurement(
        domain=domain,
        chars=len(text),
        words=len(_WORD.findall(text)),
        tokens=len(enc.ids),
        single_char_tokens=singles,
    )


def discover_tokenizers(datasets_dir: Path = DATASETS_DIR) -> list[Path]:
    if not datasets_dir.exists():
        return []
    return sorted(datasets_dir.glob("*/pirate_bpe.json"))


def digit_probe(tok: Tokenizer, samples: tuple[str, ...] = (
    "7", "42", "137", "1024", "2024", "48/2=24", "3.5", "$12", "15%", "x^2",
)) -> list[tuple[str, list[str]]]:
    """Show how numbers and math punctuation actually split.

    A tokenizer that splits 1024 into four pieces is not broken — it just never
    saw enough four-digit numbers to merge them, and that costs 4x on every
    intermediate result in a chain of thought.
    """
    return [(s, tok.encode(s).tokens) for s in samples]


def resolve_tokenizer(ref: str) -> tuple[str, Tokenizer]:
    """Local path, or an HF repo id to pull `tokenizer.json` from."""
    p = Path(ref)
    if p.exists():
        return ref, Tokenizer.from_file(str(p))
    from huggingface_hub import hf_hub_download

    return ref, Tokenizer.from_file(hf_hub_download(ref, "tokenizer.json"))


def report(label: str, tok: Tokenizer, corpus: list[dict]) -> list[Measurement]:
    vocab = tok.get_vocab_size()
    print(f"\n{'=' * 78}\n{label}  (vocab {vocab:,})\n{'=' * 78}")

    ms = [measure(tok, row["domain"], row["text"]) for row in corpus]
    base = next((m for m in ms if m.domain == BASELINE), None)

    head = f"{'domain':<20} {'chars':>6} {'tokens':>7} {'tok/char':>9} " \
           f"{'ch/tok':>7} {'tok/word':>9} {'1-char':>7} {'vs prose':>9}"
    print(head)
    print("-" * len(head))
    for m in ms:
        rel = m.tokens_per_char / base.tokens_per_char if base and base.tokens_per_char else 0.0
        flag = "  <-- baseline" if m.domain == BASELINE else ""
        print(
            f"{m.domain:<20} {m.chars:>6} {m.tokens:>7} {m.tokens_per_char:>9.3f} "
            f"{m.chars_per_token:>7.2f} {m.tokens_per_word:>9.2f} "
            f"{m.single_char_frac:>6.0%} {rel:>8.2f}x{flag}"
        )

    print("\ndigit / symbol probe")
    for s, pieces in digit_probe(tok):
        print(f"  {s:<10} -> {len(pieces):>2} {pieces}")

    if base:
        print(f"\ncontext cost at block_size={BLOCK_SIZE}")
        for m in ms:
            if m.chars_per_token:
                print(f"  {m.domain:<20} {m.chars_per_token * BLOCK_SIZE:>7.0f} "
                      f"characters fit in the window")

    for row, m in zip(corpus, ms, strict=True):
        if row.get("episode"):
            _episode_budget(m)
    return ms


def _episode_budget(m: Measurement) -> None:
    """The number reasoning-roadmap step 1 actually turns on.

    Not "does CoT fit" in the abstract — does one problem-plus-solution fit, and
    is there room left for the few-shot prefix that makes a small model produce
    the format at all.
    """
    fits = BLOCK_SIZE // m.tokens if m.tokens else 0
    print(f"\none {m.domain} in SFT format: {m.tokens} tokens ({m.chars} chars)")
    print(f"  zero-shot, one episode      {m.tokens:>5} / {BLOCK_SIZE}  "
          f"{'fits' if m.tokens < BLOCK_SIZE else 'OVERFLOWS'}")
    for shots in (1, 3, 5):
        need = m.tokens * (shots + 1)
        print(f"  {shots}-shot prompt + generation  {need:>5} / {BLOCK_SIZE}  "
              f"{'fits' if need < BLOCK_SIZE else 'OVERFLOWS'}")
    print(f"  ({fits} such episodes fit end to end)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", action="append", default=None,
                    help="Tokenizer JSON (repeatable). Default: every data/datasets/*/pirate_bpe.json")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--reference", action="append", default=None,
                    help="Control tokenizer: local tokenizer.json or an HF repo id "
                         "(e.g. Qwen/Qwen3-0.6B). Downloads on first use.")
    ap.add_argument("--text", default=None, help="Measure this string instead of the corpus")
    args = ap.parse_args()

    paths = [Path(p) for p in args.tokenizer] if args.tokenizer else discover_tokenizers()
    if not paths:
        raise SystemExit(
            f"no tokenizers found under {DATASETS_DIR}/ — build a dataset first "
            f"(`make dataset DATASET=<name>`) or pass --tokenizer"
        )

    loaded = [(str(p), Tokenizer.from_file(str(p))) for p in paths]
    loaded += [resolve_tokenizer(r) for r in (args.reference or [])]

    if args.text:
        for label, tok in loaded:
            enc = tok.encode(args.text)
            m = measure(tok, "ad-hoc", args.text)
            print(f"\n{label}  (vocab {tok.get_vocab_size():,})")
            print(f"  {m.tokens} tokens / {m.chars} chars = {m.tokens_per_char:.3f} tok/char")
            print(f"  {enc.tokens}")
        return

    corpus = load_corpus(Path(args.corpus))
    all_ms = {label: report(label, tok, corpus) for label, tok in loaded}
    if len(all_ms) > 1:
        summarize(all_ms)


def summarize(all_ms: dict[str, list[Measurement]]) -> None:
    """Side-by-side tokens/char. Reading down a column says how much of a
    domain's cost is the domain and how much is the tokenizer."""
    labels = list(all_ms)
    short = [Path(lb).parent.name or lb for lb in labels]
    domains = [m.domain for m in next(iter(all_ms.values()))]
    print(f"\n{'=' * 78}\ntokens per character, side by side (lower is cheaper)\n{'=' * 78}")
    print(f"{'domain':<20}" + "".join(f"{s[:16]:>18}" for s in short))
    print("-" * (20 + 18 * len(short)))
    for i, d in enumerate(domains):
        row = f"{d:<20}"
        for label in labels:
            m = all_ms[label][i]
            base = all_ms[label][domains.index(BASELINE)]
            rel = m.tokens_per_char / base.tokens_per_char if base.tokens_per_char else 0
            row += f"{m.tokens_per_char:>12.3f} ({rel:.1f}x)"[-18:]
        print(row)


if __name__ == "__main__":
    main()
