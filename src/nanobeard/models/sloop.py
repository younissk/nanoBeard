"""nanoBeard Sloop — v1 architecture. Frozen: do not modify after shipping."""

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
)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )

        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.triu(
            torch.full((config.block_size, config.block_size), float("-inf")),
            diagonal=1,
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        mask = self.causal_mask[:T, :T]  # type: ignore[index]

        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)

        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

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
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

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

        tok_emb = self.wte(idx)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)

        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # SFT masks non-assistant tokens with IGNORE_INDEX (-100)
            )

        return logits, loss

    def num_parameters(self) -> int:
        """Count parameters. Excludes the position embedding by convention."""
        n = sum(p.numel() for p in self.parameters())
        n -= self.wpe.weight.numel()
        return n
