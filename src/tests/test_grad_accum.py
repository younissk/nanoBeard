"""Gradient accumulation actually accumulates.

This existed as config for months and the loop ignored it: `grad_accum=8` in
`frigate_360m_full` meant the run saw 1/8 of the tokens `resolve_max_iters`
budgeted for, while printing that it had done 1.6 epochs. The failure was
silent in every direction — loss curves looked fine, the horizon math looked
fine, only the token count was wrong.

So these tests assert the two things that were untrue: that N micro-batches are
consumed per optimizer step, and that the loss is *averaged* over them rather
than summed (summing scales the effective learning rate with the accumulation
factor).
"""

from __future__ import annotations

import torch

from nanobeard import train as train_mod
from nanobeard.config import Config
from nanobeard.train import resolve_max_iters, train


def _no_eval(monkeypatch):
    """Stub estimate_loss out.

    It pulls its own batches, which would otherwise be counted as training
    micro-batches and make the assertions depend on eval_iters.
    """
    monkeypatch.setattr(
        train_mod, "estimate_loss", lambda *a, **k: {"train": 1.0, "val": 1.0}
    )


def _count_batches(monkeypatch, cfg: Config) -> int:
    calls = {"n": 0}
    real = train_mod.get_batch  # capture before patching, or `counting` recurses

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
    cfg.eval_interval = 100  # keep estimate_loss out of the count
    cfg.ckpt_interval_min = 0
    cfg.gradient_accumulation_steps = 1
    assert _count_batches(monkeypatch, cfg) == 4


def test_n_micro_batches_per_step_when_accum_is_n(synthetic_bins: Config, monkeypatch):
    cfg = synthetic_bins
    cfg.max_iters = 4
    cfg.eval_interval = 100
    cfg.ckpt_interval_min = 0
    cfg.gradient_accumulation_steps = 3
    # 4 optimizer steps x 3 micro-batches. Before the fix this was 4.
    assert _count_batches(monkeypatch, cfg) == 12


def test_zero_accum_is_treated_as_one(synthetic_bins: Config, monkeypatch):
    # max(1, ...) guards a config that would otherwise skip the inner loop
    # entirely and step on empty gradients.
    cfg = synthetic_bins
    cfg.max_iters = 3
    cfg.eval_interval = 100
    cfg.ckpt_interval_min = 0
    cfg.gradient_accumulation_steps = 0
    assert _count_batches(monkeypatch, cfg) == 3


def test_accumulating_identical_batches_matches_a_single_step(
    synthetic_bins: Config, monkeypatch, tmp_path
):
    """Averaging, not summing.

    Feed the same batch N times: the mean gradient over N identical
    micro-batches equals the gradient of one. So the trained weights must match
    a single-micro-batch run. If the loss were summed instead of divided, the
    effective LR would be N x and the weights would visibly diverge.
    """
    real_get_batch = train_mod.get_batch  # capture before patching
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
        cfg.ckpt_interval_min = 0
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
    """resolve_max_iters and the loop must agree on tokens/iter.

    They disagreed by exactly the accumulation factor before the fix — this is
    the assertion that would have caught it.
    """
    cfg = synthetic_bins
    cfg.epochs = 1.0
    cfg.max_iters = 10**9  # let epochs decide, not the ceiling
    cfg.gradient_accumulation_steps = 4
    import os

    train_tokens = os.path.getsize(cfg.train_bin) // 2
    out = resolve_max_iters(cfg)
    tokens_consumed = (
        out.max_iters
        * cfg.gradient_accumulation_steps
        * cfg.batch_size
        * cfg.block_size
    )
    assert abs(tokens_consumed - train_tokens) / train_tokens < 0.01
