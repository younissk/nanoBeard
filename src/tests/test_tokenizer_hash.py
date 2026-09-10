"""Tokenizer fingerprint roundtrip + gating behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from nanobeard.config import Config
from nanobeard.models import build_model
from nanobeard.tokenizer_hash import TokenizerMismatch, hash_file, verify_match
from nanobeard.train import save_checkpoint


def test_hash_file_deterministic(tmp_path: Path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello pirate")
    assert hash_file(f) == hash_file(f)
    # Match a known-bad-hash example.
    assert len(hash_file(f)) == 64


def test_hash_file_differs_on_content_change(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    assert hash_file(a) != hash_file(b)


def test_verify_match_accepts_correct_hash(tmp_path: Path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"matey")
    expected = hash_file(f)
    assert verify_match(f, expected) == expected


def test_verify_match_raises_on_mismatch(tmp_path: Path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"matey")
    with pytest.raises(TokenizerMismatch):
        verify_match(f, "0" * 64)


def test_verify_match_no_expected_is_silent(tmp_path: Path):
    """Legacy ckpts have no hash field — return actual, don't raise."""
    f = tmp_path / "f.bin"
    f.write_bytes(b"matey")
    actual = verify_match(f, None)
    assert actual == hash_file(f)


def test_save_checkpoint_records_tokenizer_hash(tokenized_cfg: Config):
    """save_checkpoint must write the on-disk tokenizer's hash into the ckpt."""
    model = build_model(tokenized_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    os.makedirs(tokenized_cfg.run_dir, exist_ok=True)
    save_checkpoint(model, optimizer, tokenized_cfg, iter_num=0, val_loss=1.0, best_val_loss=1.0)

    ck = torch.load(tokenized_cfg.ckpt_path, map_location="cpu", weights_only=False)
    assert ck["tokenizer_sha256"] is not None
    assert ck["tokenizer_sha256"] == hash_file(tokenized_cfg.tokenizer_path)


def test_save_checkpoint_no_tokenizer_present(tiny_cfg: Config):
    """When no tokenizer is on disk, hash field should be None (not an error)."""
    model = build_model(tiny_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    os.makedirs(tiny_cfg.run_dir, exist_ok=True)
    save_checkpoint(model, optimizer, tiny_cfg, iter_num=0, val_loss=1.0, best_val_loss=1.0)
    ck = torch.load(tiny_cfg.ckpt_path, map_location="cpu", weights_only=False)
    assert ck["tokenizer_sha256"] is None
