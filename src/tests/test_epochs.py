"""epochs -> training horizon resolution (resolve_max_iters)."""

from __future__ import annotations

import numpy as np

from nanobeard.config import Config
from nanobeard.train import resolve_max_iters


def _write_bin(cfg: Config, n_tokens: int) -> None:
    np.zeros(n_tokens, dtype=np.uint16).tofile(cfg.train_bin)


def test_default_epochs_is_one():
    assert Config().epochs == 1.0


def test_noop_without_train_bin(tiny_cfg: Config):
    tiny_cfg.max_iters = 1234
    out = resolve_max_iters(tiny_cfg)
    assert out.max_iters == 1234  # no train.bin -> untouched


def test_one_epoch_horizon(tiny_cfg: Config):
    # 8000 tokens, tokens/iter = 4*16*1 = 64 -> 125 iters per epoch.
    _write_bin(tiny_cfg, 8000)
    tiny_cfg.epochs = 1.0
    tiny_cfg.max_iters = 10_000  # high ceiling -> epochs governs
    out = resolve_max_iters(tiny_cfg)
    assert out.max_iters == 125
    assert out.lr_decay_iters == 125


def test_fractional_epochs(tiny_cfg: Config):
    _write_bin(tiny_cfg, 8000)  # 125 iters/epoch
    tiny_cfg.epochs = 1.6
    tiny_cfg.max_iters = 10_000
    out = resolve_max_iters(tiny_cfg)
    assert out.max_iters == 200  # round(1.6 * 125)


def test_max_iters_is_a_ceiling(tiny_cfg: Config):
    """Smoke runs: a tiny max_iters wins over a full epoch on a big corpus."""
    _write_bin(tiny_cfg, 8000)  # 125 iters/epoch at epochs=1.0
    tiny_cfg.epochs = 1.0
    tiny_cfg.max_iters = 50
    out = resolve_max_iters(tiny_cfg)
    assert out.max_iters == 50
    assert out.lr_decay_iters == 50


def test_horizon_at_least_one(tiny_cfg: Config):
    _write_bin(tiny_cfg, 16)
    tiny_cfg.epochs = 0.001
    tiny_cfg.max_iters = 10_000
    out = resolve_max_iters(tiny_cfg)
    assert out.max_iters >= 1
