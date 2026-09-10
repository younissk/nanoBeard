"""Shared pytest fixtures.

Design notes:
  - All fixtures use CPU + tiny models so the full fast suite runs in seconds.
  - Synthetic data (random uint16 .bin, generated tokenizer) — no network, no
    real corpus. The real shipped tokenizer is used only by tests that
    explicitly opt in via the `real_tokenizer` fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from nanobeard.config import Config


def _repo_root() -> Path:
    """Walk up to the dir holding pyproject.toml.

    Deliberately not `parent.parent[.parent]`: the suite has already been moved
    twice (root -> src/tests) and the `real_tokenizer` fixture *skips* rather
    than fails when the path is wrong, so a miscount would go unnoticed.
    """
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").is_file():
            return d
    raise RuntimeError("repo root not found — no pyproject.toml in any parent")


REPO_ROOT = _repo_root()
REAL_TOKENIZER = REPO_ROOT / "data" / "datasets" / "tiny_pirate_stories" / "pirate_bpe.json"


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Reset RNG state before every test."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Config:
    """CPU-only, microscopic model. Fits in <100ms forward."""
    data_dir = tmp_path / "data" / "tiny"
    run_dir = tmp_path / "runs" / "tiny"
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        run_name="tiny-test",
        model_name="sloop",
        data_dir=str(data_dir),
        run_dir=str(run_dir),
        vocab_size=128,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        batch_size=4,
        max_iters=5,
        warmup_iters=2,
        lr_decay_iters=5,
        eval_interval=2,
        eval_iters=2,
        log_interval=1,
        device="cpu",
        dtype="float32",
        compile=False,
        seed=0,
        resume=False,
        hf_ckpt_repo=None,
        hf_model_repo="test/repo",
    )


@pytest.fixture
def synthetic_bins(tiny_cfg: Config) -> Config:
    """Materialize train.bin and val.bin under tiny_cfg.data_dir."""
    rng = np.random.default_rng(0)
    train = rng.integers(0, tiny_cfg.vocab_size, size=2048, dtype=np.uint16)
    val = rng.integers(0, tiny_cfg.vocab_size, size=512, dtype=np.uint16)
    train.tofile(tiny_cfg.train_bin)
    val.tofile(tiny_cfg.val_bin)
    return tiny_cfg


@pytest.fixture(scope="session")
def synthetic_tokenizer(tmp_path_factory) -> Path:
    """Build a tiny BPE tokenizer on inline text once per session."""
    out = tmp_path_factory.mktemp("tok") / "tiny_bpe.json"
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=128,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = [
        "ahoy matey treasure rum sea",
        "the pirate sails the seven seas",
        "yarr a chest of gold doubloons",
        "she said look out for sharks",
    ] * 20
    tok.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    tok.save(str(out))
    return out


@pytest.fixture
def tokenized_cfg(synthetic_bins: Config, synthetic_tokenizer: Path) -> Config:
    """tiny_cfg + bins + tokenizer file copied to its data_dir."""
    dst = Path(synthetic_bins.tokenizer_path)
    dst.write_bytes(synthetic_tokenizer.read_bytes())
    return synthetic_bins


@pytest.fixture
def real_tokenizer() -> Path:
    """Real shipped tokenizer. Skip the test if it isn't present locally."""
    if not REAL_TOKENIZER.exists():
        pytest.skip(f"Real tokenizer not present at {REAL_TOKENIZER}")
    return REAL_TOKENIZER
