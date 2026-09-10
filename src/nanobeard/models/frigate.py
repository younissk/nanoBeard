"""nanoBeard Frigate — v2 architecture.

Galleon's config-driven GPT plus four modern upgrades, each gated by a Config
flag so the block stays readable and every change is auditable / ablatable:

  use_rope     RoPE rotary position embedding, replacing the learned wpe table.
  use_swiglu   SwiGLU feed-forward (8/3 expansion) instead of the 4x GELU MLP.
  use_rmsnorm  RMSNorm instead of LayerNorm.
  use_qk_norm  per-head RMSNorm on Q and K before attention — keeps logits in
               range so the deeper 12-layer stack trains stably.

With every flag False this reduces exactly to the Sloop/Galleon architecture
(learned wpe + GELU MLP + LayerNorm), which is what the parametrized model
contract exercises. configs/frigate.py flips all four on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanobeard.config import Config

ARCH_FIELDS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "use_rope",
    "use_swiglu",
    "use_rmsnorm",
    "use_qk_norm",
    "rope_theta",
)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability under bf16/fp16 autocast.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def make_norm(config: Config, dim: int) -> nn.Module:
    if config.use_rmsnorm:
        return RMSNorm(dim)
    return nn.LayerNorm(dim, bias=config.bias)


def build_rope_cache(
    block_size: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) of shape (block_size, head_dim) for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(block_size).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim) broadcast over B, n_head.
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with optional RoPE and QK-norm.

    Hand-rolled QKV (rather than nn.MultiheadAttention) so RoPE and per-head
    RMSNorm can be applied to Q/K before the dot product.
    """

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        self.use_qk_norm = config.use_qk_norm

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        if self.use_rope:
            cos, sin = build_rope_cache(
                config.block_size, self.head_dim, config.rope_theta
            )
            # Not persisted: rebuilt deterministically at init, so ckpts stay lean
            # and an arch flag flip can't desync a stale buffer.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            cos = self.rope_cos[:T]  # type: ignore[index]
            sin = self.rope_sin[:T]  # type: ignore[index]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """4x GELU feed-forward. Used when use_swiglu is False."""

    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: silu(W_gate x) * (W_up x) -> W_down.

    Hidden dim = 8/3 * n_embd (rounded up to a multiple of 64) so the gated FFN
    keeps roughly the same param count as the 4x GELU MLP it replaces.
    """

    def __init__(self, config: Config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)  # round up to multiple of 64
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(x))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = make_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config, config.n_embd)
        self.mlp = SwiGLU(config) if config.use_swiglu else MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # RoPE replaces the learned position table; wpe is None when use_rope.
        self.wpe = (
            None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        )
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = make_norm(config, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input embedding and LM head share weights.
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert self.config.block_size >= T, (
            f"Cannot forward sequence of length {T}; block_size is {self.config.block_size}"
        )

        x = self.wte(idx)

        if self.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.wpe(pos)

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def num_parameters(self) -> int:
        """Count parameters. Excludes the learned position embedding (if any)."""
        n = sum(p.numel() for p in self.parameters())
        if self.wpe is not None:
            n -= self.wpe.weight.numel()
        return n
