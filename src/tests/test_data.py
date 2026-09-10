"""Data loader: get_batch shapes, ranges, missing files, determinism under seed."""

from __future__ import annotations

import pytest
import torch

from nanobeard.config import Config
from nanobeard.data import get_batch


def test_get_batch_shape(synthetic_bins: Config):
    x, y = get_batch("train", synthetic_bins)
    assert x.shape == (synthetic_bins.batch_size, synthetic_bins.block_size)
    assert y.shape == (synthetic_bins.batch_size, synthetic_bins.block_size)


def test_get_batch_targets_shifted_by_one(synthetic_bins: Config):
    """y[t] should equal x[t+1] for every position — that's the next-token target."""
    torch.manual_seed(42)
    x, y = get_batch("train", synthetic_bins)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_dtype_is_long(synthetic_bins: Config):
    """Embedding layers require int64."""
    x, _ = get_batch("train", synthetic_bins)
    assert x.dtype == torch.long


def test_get_batch_token_range(synthetic_bins: Config):
    x, y = get_batch("train", synthetic_bins)
    assert x.min().item() >= 0
    assert y.max().item() < synthetic_bins.vocab_size


def test_get_batch_val_split(synthetic_bins: Config):
    x, _ = get_batch("val", synthetic_bins)
    assert x.shape == (synthetic_bins.batch_size, synthetic_bins.block_size)


def test_get_batch_bad_split_raises(synthetic_bins: Config):
    with pytest.raises(ValueError):
        get_batch("test", synthetic_bins)


def test_get_batch_missing_file_raises(tiny_cfg: Config):
    with pytest.raises((FileNotFoundError, ValueError)):
        get_batch("train", tiny_cfg)


def test_get_batch_seeded_is_reproducible(synthetic_bins: Config):
    torch.manual_seed(123)
    x1, _ = get_batch("train", synthetic_bins)
    torch.manual_seed(123)
    x2, _ = get_batch("train", synthetic_bins)
    assert torch.equal(x1, x2)
