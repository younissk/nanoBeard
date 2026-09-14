"""Optimizers — AdamW (default) and Muon.

Muon (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024) updates 2D
hidden weight matrices with an orthogonalized momentum step. It only applies to
the transformer's 2D matmul weights — embeddings, the LM head, norms, and biases
fall back to AdamW. We implement that hybrid in a single Optimizer subclass so
the training loop keeps one optimizer (one scaler.step, one state_dict for
resume); each param group carries a `use_muon` flag selecting the update rule.

Single-device only — no distributed all-gather. Fine for the single-4090 runs.

build_optimizer() is the one entry point: it reads config.optimizer ("adamw" |
"muon"), classifies params, and tags each group with `lr_ratio` so the caller's
cosine LR schedule scales every group from one scalar (Muon runs at a much
higher peak LR than AdamW, so its group keeps muon_lr/learning_rate as ratio).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from nanobeard.config import Config

# Param-name substrings that must NOT go to Muon (embeddings + tied LM head).
_ADAMW_ONLY = ("wte", "wpe", "lm_head")


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Orthogonalize G (2D) via a quintic Newton-Schulz iteration.

    Returns a matrix with roughly the same singular vectors as G but singular
    values pushed toward 1. Runs in bfloat16 — the iteration is robust to the
    reduced precision and it keeps the step cheap.
    """
    assert G.ndim == 2, "Newton-Schulz expects a 2D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)  # ensure top singular value <= 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Hybrid Muon + AdamW. Per-group `use_muon` selects the update rule.

    Muon groups expect: lr, momentum, nesterov, ns_steps, weight_decay.
    AdamW groups expect: lr, betas, eps, weight_decay.
    """

    def __init__(self, param_groups: list[dict]):
        for g in param_groups:
            if g.get("use_muon"):
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("nesterov", True)
                g.setdefault("ns_steps", 5)
                g.setdefault("weight_decay", 0.0)
            else:
                g["use_muon"] = False
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = state["momentum_buffer"] = torch.zeros_like(grad)
            buf.mul_(momentum).add_(grad)
            grad = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
            update = zeropower_via_newtonschulz5(grad, group["ns_steps"]).to(p.dtype)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            # Scale so the RMS of the update matches across non-square matrices.
            scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group: dict) -> None:
        lr, (b1, b2) = group["lr"], group["betas"]
        eps, wd = group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(b1).add_(grad, alpha=1.0 - b1)
            exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
            bias1, bias2 = 1.0 - b1**t, 1.0 - b2**t
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)


def _classify_params(
    model: nn.Module,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Split params into (muon 2D matrices, adamw-decay, adamw-no-decay).

    Tied weights (wte == lm_head) are de-duplicated by tensor id.
    """
    muon: list[Tensor] = []
    adam_decay: list[Tensor] = []
    adam_no_decay: list[Tensor] = []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        embed_or_head = any(k in name for k in _ADAMW_ONLY)
        if p.dim() >= 2 and not embed_or_head:
            muon.append(p)
        elif p.dim() >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    return muon, adam_decay, adam_no_decay


def _adamw_groups(model: nn.Module, config: Config) -> list[dict[str, Any]]:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": config.weight_decay, "lr_ratio": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_ratio": 1.0},
    ]


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Build optimizer for model according to config.optimizer ('adamw' | 'muon')."""
    if config.optimizer == "muon":
        muon, adam_decay, adam_no_decay = _classify_params(model)
        ratio = config.muon_lr / config.learning_rate
        groups: list[dict[str, Any]] = [
            {
                "params": muon,
                "use_muon": True,
                "lr": config.muon_lr,
                "momentum": config.muon_momentum,
                "ns_steps": config.muon_ns_steps,
                "weight_decay": 0.0,
                "lr_ratio": ratio,
            },
            {
                "params": adam_decay,
                "use_muon": False,
                "lr": config.learning_rate,
                "betas": (config.beta1, config.beta2),
                "weight_decay": config.weight_decay,
                "lr_ratio": 1.0,
            },
            {
                "params": adam_no_decay,
                "use_muon": False,
                "lr": config.learning_rate,
                "betas": (config.beta1, config.beta2),
                "weight_decay": 0.0,
                "lr_ratio": 1.0,
            },
        ]
        groups = [g for g in groups if g["params"]]  # drop empties
        optimizer: torch.optim.Optimizer = Muon(groups)
        n_muon = sum(p.numel() for p in muon)
        n_adam = sum(p.numel() for p in adam_decay + adam_no_decay)
        print(
            f"Optimizer: Muon (lr={config.muon_lr}, momentum={config.muon_momentum}, "
            f"ns_steps={config.muon_ns_steps}) + AdamW fallback (lr={config.learning_rate})"
        )
        print(f"  Muon params:   {n_muon:,} ({len(muon)} tensors)")
        print(f"  AdamW params:  {n_adam:,} ({len(adam_decay) + len(adam_no_decay)} tensors)")
        return optimizer

    if config.optimizer != "adamw":
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}; expected 'adamw' or 'muon'"
        )

    groups = _adamw_groups(model, config)
    optimizer = torch.optim.AdamW(
        groups, lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_no_decay = sum(p.numel() for p in groups[1]["params"])
    print(f"Optimizer: AdamW, lr={config.learning_rate}, betas=({config.beta1}, {config.beta2})")
    print(f"  Decayed params:    {n_decay:,} ({len(groups[0]['params'])} tensors)")
    print(f"  No-decay params:   {n_no_decay:,} ({len(groups[1]['params'])} tensors)")
    return optimizer
