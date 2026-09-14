"""Gradient accumulation tests.

Verifies that:
- N micro-batches are consumed per optimizer step when gradient_accumulation_steps=N.
- Loss is averaged across micro-batches (mean gradient), so training on N identical
  batches produces identical weights to a single step.
- Zero/negative accumulation is safely treated as 1.
- Horizon math (resolve_max_iters) agrees with the loop's token consumption.
"""

from __future__ import annotations

import os

import torch

from nanobeard import train as train_mod
from nanobeard.config import Config
from nanobeard.train import resolve_max_iters, train


def _no_eval(monkeypatch):
    """Stub estimate_loss out so it doesn't count towards training micro-batches."""
    monkeypatch.setattr(
        train_mod, "estimate_loss", lambda *a, **k: {"train": 1.0, "val": 1.0}
    )


def _count_batches(monkeypatch, cfg: Config) -> int:
    calls = {"n": 0}
    real = train_mod.get_batch

    def counting(split, config):
        if split == "train":
            calls["n"] += 1
        return real(split, config)

    _no_eval(monkeypatch)
    monkeypatch.setattr(train_mod, "get_batch", counting)
    train(cfg)
    return calls["n"]


def test_one_micro_batch_per_step_when_accum_is_one(synthetic_bins: Config, monkeypatch):
    cfg = synthetic_bins
    cfg.max_iters = 4
    cfg.eval_interval = 100
    cfg.gradient_accumulation_steps = 1
    assert _count_batches(monkeypatch, cfg) == 4


def test_n_micro_batches_per_step_when_accum_is_n(synthetic_bins: Config, monkeypatch):
    cfg = synthetic_bins
    cfg.max_iters = 4
    cfg.eval_interval = 100
    cfg.gradient_accumulation_steps = 3
    # 4 optimizer steps x 3 micro-batches = 12 total micro-batches
    assert _count_batches(monkeypatch, cfg) == 12


def test_zero_accum_is_treated_as_one(synthetic_bins: Config, monkeypatch):
    cfg = synthetic_bins
    cfg.max_iters = 3
    cfg.eval_interval = 100
    cfg.gradient_accumulation_steps = 0
    assert _count_batches(monkeypatch, cfg) == 3


def test_accumulating_identical_batches_matches_a_single_step(
    synthetic_bins: Config, monkeypatch, tmp_path
):
    """Averaging, not summing.

    Feed the same batch N times: the mean gradient over N identical
    micro-batches equals the gradient of one. So the trained weights must match
    a single-micro-batch run.
    """
    real_get_batch = train_mod.get_batch
    fixed = None

    def frozen(split, config):
        nonlocal fixed
        if fixed is None:
            fixed = real_get_batch(split, config)
        return fixed

    def run(accum: int, run_dir: str) -> dict:
        nonlocal fixed
        fixed = None
        cfg = Config(**{**synthetic_bins.__dict__, "run_dir": run_dir})
        cfg.max_iters = 3
        cfg.eval_interval = 100
        cfg.warmup_iters = 0
        cfg.resume = False
        cfg.gradient_accumulation_steps = accum
        torch.manual_seed(1234)
        _no_eval(monkeypatch)
        monkeypatch.setattr(train_mod, "get_batch", frozen)
        train(cfg)
        return torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["model"]

    one = run(1, str(tmp_path / "accum1"))
    four = run(4, str(tmp_path / "accum4"))

    for k in one:
        assert torch.allclose(one[k], four[k], atol=1e-5, rtol=1e-4), k


def test_horizon_math_counts_the_micro_batches(synthetic_bins: Config):
    """resolve_max_iters and the loop agree on tokens/iter."""
    cfg = synthetic_bins
    cfg.epochs = 1.0
    cfg.max_iters = 10**9
    cfg.gradient_accumulation_steps = 4

    train_tokens = os.path.getsize(cfg.train_bin) // 2
    out = resolve_max_iters(cfg)
    accum = max(1, cfg.gradient_accumulation_steps)
    tokens_consumed = (
        out.max_iters
        * accum
        * cfg.batch_size
        * cfg.block_size
    )
    assert abs(tokens_consumed - train_tokens) / train_tokens < 0.05
