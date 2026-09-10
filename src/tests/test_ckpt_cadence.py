"""Checkpoint cadence for interruptible hosts.

Spot instances are evicted with no warning, so what has to be bounded is
*minutes of lost work*, not iterations — iteration time is not constant, and on
a cheap host it varies a lot. Local saves are cheap; Hub pushes are hundreds of
MB and used to run synchronously on every eval, stalling a GPU billed by the
second.

These tests pin the two behaviours that make eviction cheap: saves happen on
wall-clock, and uploads neither block training nor race each other.
"""

from __future__ import annotations

import time

import torch

from nanobeard import train as train_mod
from nanobeard.config import Config
from nanobeard.train import save_checkpoint, train


def _no_eval(monkeypatch):
    monkeypatch.setattr(
        train_mod, "estimate_loss", lambda *a, **k: {"train": 1.0, "val": 1.0}
    )


def _saves(monkeypatch, cfg: Config) -> list[tuple[str, bool]]:
    """Run a short training and return (tag, push) for every save."""
    seen: list[tuple[str, bool]] = []
    real = train_mod.save_checkpoint

    def spy(*args, **kwargs):
        seen.append((kwargs.get("tag", "latest"), bool(kwargs.get("push", True))))
        return real(*args, **kwargs)

    _no_eval(monkeypatch)
    monkeypatch.setattr(train_mod, "save_checkpoint", spy)
    train(cfg)
    return seen


def _cfg(base: Config, **over) -> Config:
    cfg = Config(**{**base.__dict__, **over})
    cfg.max_iters = 4
    cfg.eval_interval = 100
    cfg.resume = False
    return cfg


def test_periodic_saves_fire_on_wall_clock(synthetic_bins: Config, monkeypatch):
    # An interval of ~0 makes every iteration due, which is the cheap way to
    # observe that the clock — not eval_interval — is driving the save.
    cfg = _cfg(synthetic_bins, ckpt_interval_min=1e-9, hub_push_interval_min=0)
    tags = [t for t, _ in _saves(monkeypatch, cfg)]
    assert tags.count("periodic") >= 3


def test_zero_interval_disables_periodic_saves(synthetic_bins: Config, monkeypatch):
    cfg = _cfg(synthetic_bins, ckpt_interval_min=0, hub_push_interval_min=0)
    tags = [t for t, _ in _saves(monkeypatch, cfg)]
    assert "periodic" not in tags


def test_a_long_interval_leaves_only_eval_and_final_saves(
    synthetic_bins: Config, monkeypatch
):
    # The iter-0 eval always saves (first val loss is by definition the best);
    # what a long interval must suppress is everything between it and the end.
    cfg = _cfg(synthetic_bins, ckpt_interval_min=600, hub_push_interval_min=600)
    tags = [t for t, _ in _saves(monkeypatch, cfg)]
    assert tags == ["best", "final"]


def test_training_always_ends_with_a_final_save(synthetic_bins: Config, monkeypatch):
    # The cadence otherwise discards everything since the last checkpoint,
    # which on a short run is most of it.
    cfg = _cfg(synthetic_bins, ckpt_interval_min=0, hub_push_interval_min=0)
    saves = _saves(monkeypatch, cfg)
    assert saves[-1] == ("final", True)


def test_periodic_saves_do_not_push_unless_the_push_clock_is_due(
    synthetic_bins: Config, monkeypatch
):
    # The whole point of splitting the cadences: frequent local saves, rare
    # uploads. A periodic save with the push clock not yet due must not upload.
    cfg = _cfg(synthetic_bins, ckpt_interval_min=1e-9, hub_push_interval_min=600)
    pushes = [push for tag, push in _saves(monkeypatch, cfg) if tag == "periodic"]
    assert pushes and not any(pushes)


def test_save_returns_no_thread_when_push_is_off(synthetic_bins: Config):
    cfg = Config(**{**synthetic_bins.__dict__, "hf_ckpt_repo": "someone/repo"})
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    assert save_checkpoint(model, opt, cfg, 0, 1.0, 1.0, push=False) is None


def test_save_returns_no_thread_without_a_ckpt_repo(synthetic_bins: Config):
    cfg = Config(**{**synthetic_bins.__dict__, "hf_ckpt_repo": None})
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    assert save_checkpoint(model, opt, cfg, 0, 1.0, 1.0, push=True) is None


def test_upload_runs_in_the_background_and_cleans_up(
    synthetic_bins: Config, monkeypatch, tmp_path
):
    """The upload must not block the training thread, and must not leave the
    staged copy behind — a stray .upload file per checkpoint fills a spot box's
    small disk."""
    started = []

    class FakeApi:
        def upload_file(self, path_or_fileobj, **kwargs):
            started.append(path_or_fileobj)
            time.sleep(0.05)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    cfg = Config(**{**synthetic_bins.__dict__, "hf_ckpt_repo": "someone/repo"})
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())

    t0 = time.perf_counter()
    thread = save_checkpoint(model, opt, cfg, 7, 1.0, 1.0, push=True)
    handed_back = time.perf_counter() - t0

    assert thread is not None
    assert handed_back < 0.05, "save_checkpoint blocked on the upload"
    thread.join(timeout=5)
    assert started == [f"{cfg.ckpt_path}.upload"]
    import os

    assert not os.path.exists(f"{cfg.ckpt_path}.upload"), "staged copy not cleaned up"
