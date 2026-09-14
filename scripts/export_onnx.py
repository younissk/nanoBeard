"""Export a nanoBeard checkpoint -> ONNX + js-tiktoken tokenizer for the app.

The mobile app (NanoBeard-App) runs models with onnxruntime-react-native and
tokenizes with js-tiktoken. This script turns a training checkpoint into the
three files the app downloads from the Hub:

    model.onnx              the weights (idx int64 [B,T] -> logits f32 [B,T,V])
    tokenizer.tiktoken.json js-tiktoken ranks (base64) + specials + pat_str
    model_config.json       arch dims + eot_id + kind, consumed by the JS runtime

Usage:
    uv run python -m scripts.export_onnx \
        --ckpt runs/galleon-sft/sft_ckpt.pt \
        --tokenizer data/datasets/pirate_enhanced/pirate_bpe.json \
        --kind chat \
        --out-dir export/galleon
    # then upload export/galleon/* to younissk/nanoBeard-galleon-34M

Notes:
  - Mirrors NanoBeard-App/scripts/convert-to-onnx.py: build the model with
    nn.MultiheadAttention (matches the checkpoint's state_dict layout), then
    swap each attention for a manual implementation that traces cleanly with a
    dynamic sequence length before exporting.
  - Arch dims (vocab/block/layers/heads/embd) are read from the checkpoint's
    stored Config — nothing is hardcoded, so the same script handles sloop and
    galleon.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Model — two attention layouts. nn.MHA matches the checkpoint; ManualAttention
# exports cleanly at dynamic T. Identical math, weights copied across.
# --------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.n_embd = c["n_embd"]
        self.n_head = c["n_head"]
        self.attn = nn.MultiheadAttention(
            embed_dim=c["n_embd"],
            num_heads=c["n_head"],
            dropout=0.0,
            bias=c["bias"],
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(0.0)
        mask = torch.triu(
            torch.full((c["block_size"], c["block_size"]), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        mask = self.causal_mask[:T, :T]
        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.resid_dropout(y)


class ManualAttention(nn.Module):
    """Same math as nn.MHA, manual ops — exports cleanly with dynamic T."""

    def __init__(self, c: dict):
        super().__init__()
        self.n_embd = c["n_embd"]
        self.n_head = c["n_head"]
        self.head_dim = c["n_embd"] // c["n_head"]
        assert self.head_dim * self.n_head == self.n_embd
        self.qkv = nn.Linear(c["n_embd"], 3 * c["n_embd"], bias=c["bias"])
        self.out_proj = nn.Linear(c["n_embd"], c["n_embd"], bias=c["bias"])

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.c_fc = nn.Linear(c["n_embd"], 4 * c["n_embd"], bias=c["bias"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * c["n_embd"], c["n_embd"], bias=c["bias"])
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c["n_embd"], bias=c["bias"])
        self.attn = CausalSelfAttention(c)
        self.ln_2 = nn.LayerNorm(c["n_embd"], bias=c["bias"])
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c["vocab_size"], c["n_embd"])
        self.wpe = nn.Embedding(c["block_size"], c["n_embd"])
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["n_layer"])])
        self.ln_f = nn.LayerNorm(c["n_embd"], bias=c["bias"])
        self.lm_head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(tok_emb + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# --------------------------------------------------------------------------
# Tokenizer: HF BPE tokenizer.json -> js-tiktoken ranks (base64).
# Ported from NanoBeard-App/scripts/convert-tokenizer.mjs.
# --------------------------------------------------------------------------
def _bytes_to_unicode() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def convert_tokenizer(hf_path: Path) -> tuple[dict, int]:
    """Return (js-tiktoken json, eot_id)."""
    import base64

    u2b = _bytes_to_unicode()
    hf = json.loads(hf_path.read_text(encoding="utf-8"))
    vocab = hf["model"]["vocab"]
    specials = {t["content"]: t["id"] for t in hf.get("added_tokens", []) if t.get("special")}

    lines = []
    for tok, tid in vocab.items():
        if tok in specials:
            continue
        raw = bytes(u2b[ch] for ch in tok)
        lines.append(f"! {tid} {base64.b64encode(raw).decode()}")

    eot_id = specials.get("<|endoftext|>", 0)
    payload = {
        "bpe_ranks": "\n".join(lines),
        "special_tokens": specials,
        "pat_str": "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    }
    return payload, eot_id


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ckpt.pt / sft_ckpt.pt")
    ap.add_argument("--tokenizer", required=True, help="Path to pirate_bpe.json")
    ap.add_argument("--kind", choices=["completion", "chat"], required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_obj = ckpt["config"]
    c = {
        "vocab_size": cfg_obj.vocab_size,
        "block_size": cfg_obj.block_size,
        "n_layer": cfg_obj.n_layer,
        "n_head": cfg_obj.n_head,
        "n_embd": cfg_obj.n_embd,
        "bias": cfg_obj.bias,
    }
    print(f"arch: {c}")

    model = GPT(c).eval()
    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # causal_mask buffers are recomputed, not loaded — ignore them in the report.
    missing = [k for k in missing if "causal_mask" not in k]
    unexpected = [k for k in unexpected if "causal_mask" not in k]
    if missing:
        print(f"  missing keys: {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")

    dummy = torch.zeros((1, 8), dtype=torch.long)
    with torch.no_grad():
        ref = model(dummy)
    print(f"sanity forward ok: {tuple(ref.shape)}")

    print("rewriting attention for export...")
    for blk in model.blocks:
        mha = blk.attn.attn
        manual = ManualAttention(c)
        with torch.no_grad():
            manual.qkv.weight.copy_(mha.in_proj_weight)
            manual.out_proj.weight.copy_(mha.out_proj.weight)
            if c["bias"]:
                manual.qkv.bias.copy_(mha.in_proj_bias)
                if mha.out_proj.bias is not None:
                    manual.out_proj.bias.copy_(mha.out_proj.bias)
        blk.attn = manual
    model.eval()

    with torch.no_grad():
        new = model(dummy)
    diff = (ref - new).abs().max().item()
    print(f"max abs diff after rewrite: {diff:.6e}")
    assert diff < 1e-4, "weight transfer mismatched"

    onnx_path = out / "model.onnx"
    print(f"exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["idx"],
        output_names=["logits"],
        dynamic_axes={"idx": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    tok_json, eot_id = convert_tokenizer(Path(args.tokenizer))
    (out / "tokenizer.tiktoken.json").write_text(json.dumps(tok_json))
    print(f"wrote tokenizer ({len(tok_json['bpe_ranks'].splitlines())} ranks, eot {eot_id})")

    model_cfg = {
        "vocab_size": c["vocab_size"],
        "block_size": c["block_size"],
        "n_layer": c["n_layer"],
        "n_head": c["n_head"],
        "n_embd": c["n_embd"],
        "bias": c["bias"],
        "eot_id": eot_id,
        "kind": args.kind,
    }
    (out / "model_config.json").write_text(json.dumps(model_cfg, indent=2))
    print(f"wrote model_config.json: {model_cfg}")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"done: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
