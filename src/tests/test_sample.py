"""Sampling: greedy determinism, top-k filtering, context-window cropping."""

from __future__ import annotations

import torch

from nanobeard.config import Config
from nanobeard.models import build_model
from nanobeard.sample import generate


def test_greedy_is_deterministic(tiny_cfg: Config):
    """top_k=1 makes sampling effectively greedy — same input -> same output."""
    model = build_model(tiny_cfg).eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out_a = generate(model, idx.clone(), max_new_tokens=5, temperature=1.0, top_k=1)
    out_b = generate(model, idx.clone(), max_new_tokens=5, temperature=1.0, top_k=1)
    assert torch.equal(out_a, out_b)


def test_generate_appends_correct_number_of_tokens(tiny_cfg: Config):
    model = build_model(tiny_cfg).eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = generate(model, idx, max_new_tokens=7, temperature=1.0, top_k=5)
    assert out.shape[1] == 3 + 7


def test_generate_crops_to_block_size(tiny_cfg: Config):
    """Should run cleanly even when starting context > block_size — it should
    crop internally and not raise."""
    model = build_model(tiny_cfg).eval()
    overlong = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.block_size + 5))
    out = generate(model, overlong, max_new_tokens=3, temperature=1.0, top_k=5)
    assert out.shape[1] == overlong.shape[1] + 3


def test_generate_output_in_vocab_range(tiny_cfg: Config):
    model = build_model(tiny_cfg).eval()
    idx = torch.tensor([[1, 2]], dtype=torch.long)
    out = generate(model, idx, max_new_tokens=10, temperature=1.0, top_k=None)
    assert out.max().item() < tiny_cfg.vocab_size
    assert out.min().item() >= 0


def test_generate_with_top_k_only_uses_top_tokens(tiny_cfg: Config):
    """With top_k=1, every new token must be the argmax of its position's logits."""
    torch.manual_seed(0)
    model = build_model(tiny_cfg).eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = generate(model, idx.clone(), max_new_tokens=1, temperature=1.0, top_k=1)
    with torch.no_grad():
        logits, _ = model(idx)
    expected = logits[0, -1].argmax().item()
    assert out[0, -1].item() == expected
