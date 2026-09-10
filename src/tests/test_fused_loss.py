"""Fused (chunked) cross-entropy must be the unfused one, exactly.

The fused path skips materialising the [B*T, vocab] logits tensor, which is the
allocation that decides how big a micro-batch fits on a rented GPU. That is only
worth having if the loss and the gradients are unchanged — a fused loss that is
subtly different is a silent training bug that shows up as "the run just didn't
work as well" days later.

The ragged-final-chunk case is the one that breaks naive implementations:
averaging per-chunk means instead of summing and dividing once mis-weights the
last, shorter chunk.
"""

from __future__ import annotations

import pytest
import torch

from nanobeard.config import Config
from nanobeard.models.frigate import GPT, fused_cross_entropy

# Named separately so the batch helpers can use them as ints — pulling them
# back out of the heterogeneous BASE dict types them as `object`.
VOCAB = 257
BLOCK = 32

BASE = dict(
    model_name="frigate",
    vocab_size=VOCAB,
    block_size=BLOCK,
    n_layer=2,
    n_head=2,
    n_embd=32,
    dropout=0.0,
    use_rope=True,
    use_swiglu=True,
    use_rmsnorm=True,
    use_qk_norm=True,
    device="cpu",
)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(1)
    return GPT(Config(**BASE, fused_loss=False)).eval()


def _batch(masked: int = 0):
    torch.manual_seed(0)
    idx = torch.randint(0, VOCAB, (3, BLOCK))
    tgt = torch.randint(0, VOCAB, (3, BLOCK))
    if masked:
        tgt[0, :masked] = -100  # what SFT label masking produces
    return idx, tgt


def _loss(model: GPT, idx, tgt, fused: bool):
    model.config = Config(**BASE, fused_loss=fused)
    return model(idx, tgt)[1]


def test_fused_loss_equals_unfused(model):
    idx, tgt = _batch()
    assert torch.allclose(_loss(model, idx, tgt, False), _loss(model, idx, tgt, True), atol=1e-6)


def test_fused_loss_equals_unfused_with_masked_targets(model):
    idx, tgt = _batch(masked=5)
    assert torch.allclose(_loss(model, idx, tgt, False), _loss(model, idx, tgt, True), atol=1e-6)


def test_fused_gradients_equal_unfused(model):
    idx, tgt = _batch(masked=5)
    model.zero_grad()
    _loss(model, idx, tgt, False).backward()
    unfused = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    _loss(model, idx, tgt, True).backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(unfused[n], p.grad, atol=1e-5, rtol=1e-4), n


@pytest.mark.parametrize("chunk", [1, 7, 32, 1024])
def test_chunk_size_does_not_change_the_result(model, chunk):
    """Including chunk sizes that leave a ragged remainder."""
    idx, tgt = _batch(masked=3)
    with torch.no_grad():
        hidden = model.ln_f(model.drop(model.wte(idx)))
        for block in model.blocks:
            hidden = block(hidden)
        hidden = model.ln_f(hidden)
        ref = fused_cross_entropy(hidden, model.lm_head, tgt, chunk_rows=10_000)
        got = fused_cross_entropy(hidden, model.lm_head, tgt, chunk_rows=chunk)
    assert torch.allclose(ref, got, atol=1e-6)


def test_fused_path_returns_no_logits(model):
    # Nothing reads logits when targets are supplied; skipping the projection
    # is the whole point. Sampling never passes targets.
    idx, tgt = _batch()
    model.config = Config(**BASE, fused_loss=True)
    logits, loss = model(idx, tgt)
    assert logits is None
    assert loss is not None


def test_sampling_still_gets_logits_under_fused_config(model):
    idx, _ = _batch()
    model.config = Config(**BASE, fused_loss=True)
    logits, loss = model(idx)  # no targets
    assert loss is None
    assert logits.shape == (3, BLOCK, VOCAB)


def test_all_masked_batch_is_zero_and_still_differentiable(model):
    idx, tgt = _batch()
    tgt[:] = -100
    model.config = Config(**BASE, fused_loss=True)
    loss = model(idx, tgt)[1]
    assert loss.item() == 0.0
    loss.backward()  # must not raise on a leaf tensor


def test_default_config_keeps_the_unfused_path(model):
    # fused_loss defaults off, so publish/sample/eval behaviour is untouched.
    assert Config(**BASE).fused_loss is False
