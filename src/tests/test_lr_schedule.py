"""LR schedule math — warmup linear, cosine decay, post-decay floor."""

from __future__ import annotations

import math

import pytest

from nanobeard.config import Config
from nanobeard.train import get_lr


@pytest.fixture
def lr_cfg() -> Config:
    return Config(
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=100,
        lr_decay_iters=1000,
    )


def test_warmup_starts_below_peak(lr_cfg: Config):
    assert get_lr(0, lr_cfg) < lr_cfg.learning_rate
    assert get_lr(0, lr_cfg) > 0


def test_warmup_endpoint_near_peak(lr_cfg: Config):
    """At iter == warmup_iters - 1 the linear ramp should be near (but ≤) peak."""
    lr = get_lr(lr_cfg.warmup_iters - 1, lr_cfg)
    assert lr <= lr_cfg.learning_rate
    assert lr > 0.9 * lr_cfg.learning_rate


def test_post_decay_floor(lr_cfg: Config):
    assert get_lr(lr_cfg.lr_decay_iters + 1, lr_cfg) == lr_cfg.min_lr
    assert get_lr(lr_cfg.lr_decay_iters * 10, lr_cfg) == lr_cfg.min_lr


def test_cosine_midpoint_between_peak_and_floor(lr_cfg: Config):
    mid = (lr_cfg.warmup_iters + lr_cfg.lr_decay_iters) // 2
    lr = get_lr(mid, lr_cfg)
    expected = lr_cfg.min_lr + 0.5 * (lr_cfg.learning_rate - lr_cfg.min_lr)
    assert math.isclose(lr, expected, rel_tol=0.02)


def test_lr_monotone_decreasing_after_warmup(lr_cfg: Config):
    """After warmup, LR should be monotone non-increasing through decay."""
    prev = float("inf")
    for it in range(lr_cfg.warmup_iters, lr_cfg.lr_decay_iters + 1, 50):
        cur = get_lr(it, lr_cfg)
        assert cur <= prev + 1e-9
        prev = cur


def test_lr_monotone_increasing_during_warmup(lr_cfg: Config):
    prev = -1.0
    for it in range(0, lr_cfg.warmup_iters):
        cur = get_lr(it, lr_cfg)
        assert cur >= prev
        prev = cur
