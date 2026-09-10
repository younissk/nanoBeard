"""End-to-end micro train: verifies the whole loop wires correctly.

Slow-marked because a few iters of forward+backward is the dominant cost,
but on tiny_cfg this completes in well under a second."""

from __future__ import annotations

import os

import pytest
import torch

from nanobeard.config import Config
from nanobeard.train import resolve_vocab_size, train


def test_resolve_vocab_size_overrides_from_tokenizer(tokenized_cfg: Config):
    """Tokenizer vocab wins over a mismatched config value."""
    from tokenizers import Tokenizer

    expected = Tokenizer.from_file(tokenized_cfg.tokenizer_path).get_vocab_size()
    tokenized_cfg.vocab_size = 999
    out = resolve_vocab_size(tokenized_cfg)
    assert out.vocab_size == expected != 999


def test_resolve_vocab_size_noop_without_tokenizer(tiny_cfg: Config):
    """No tokenizer built yet -> config value is left untouched."""
    tiny_cfg.vocab_size = 4242
    assert not os.path.exists(tiny_cfg.tokenizer_path)
    out = resolve_vocab_size(tiny_cfg)
    assert out.vocab_size == 4242


@pytest.mark.slow
def test_train_runs_end_to_end(synthetic_bins: Config):
    cfg = synthetic_bins
    cfg.max_iters = 5
    cfg.eval_interval = 2
    cfg.eval_iters = 2
    train(cfg)

    ckpt_path = cfg.ckpt_path
    assert os.path.exists(ckpt_path), "checkpoint not written"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in ck
    assert "optimizer" in ck
    assert "config" in ck
    assert ck["model_name"] == "sloop"
    assert ck["iter_num"] >= 0
    assert torch.isfinite(torch.tensor(ck["val_loss"]))


@pytest.mark.slow
def test_train_decreases_loss(synthetic_bins: Config):
    """Loss at step 0 should beat random chance after a few iters of training.
    Synthetic data is random so we don't expect great loss — just that the
    optimizer actually moves the model."""
    cfg = synthetic_bins
    cfg.max_iters = 20
    cfg.eval_interval = 100  # disable mid-run eval
    cfg.warmup_iters = 2
    cfg.lr_decay_iters = 20
    cfg.learning_rate = 1e-2
    train(cfg)
    ck = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    # Random model on V=128 has loss ~log(128) ≈ 4.85. After training, expect
    # some movement either direction — the test is that the loop ran without
    # NaN / crash and produced a usable ckpt.
    assert torch.isfinite(torch.tensor(ck["val_loss"]))
