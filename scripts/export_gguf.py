"""Convert a nanoBeard **Frigate** checkpoint -> GGUF for on-device llama.cpp.

Frigate's architecture (RoPE + SwiGLU + RMSNorm + per-head QK-norm, no bias,
tied embeddings) is byte-for-byte a **Qwen3** model. So instead of teaching
llama.cpp a bespoke arch, we re-emit the weights as a HuggingFace
``Qwen3ForCausalLM`` folder and lean on the upstream, battle-tested
``convert_hf_to_gguf.py`` for the GGUF metadata + tensor naming.

Pipeline:
    sft_ckpt.pt  (fp32 weights + optimizer state)
      -> drop optimizer, remap weights to Qwen3 names, cast fp16
      -> <out>/hf/{model.safetensors, config.json, tokenizer*.json}
      -> convert_hf_to_gguf.py --outtype f16  ->  <out>/<name>-f16.gguf
      -> llama-quantize                        ->  <out>/<name>.Q4_K_M.gguf, .Q8_0.gguf

Weight remap (frigate -> Qwen3 HF), verified against nanobeard/models/frigate.py:
    wte.weight                      -> model.embed_tokens.weight  (tied; lm_head dropped)
    blocks.i.ln_1.weight            -> model.layers.i.input_layernorm.weight
    blocks.i.attn.c_attn.weight     -> split -> self_attn.{q,k,v}_proj.weight
    blocks.i.attn.q_norm.weight     -> self_attn.q_norm.weight   (Qwen3 has per-head QK-norm)
    blocks.i.attn.k_norm.weight     -> self_attn.k_norm.weight
    blocks.i.attn.c_proj.weight     -> self_attn.o_proj.weight
    blocks.i.ln_2.weight            -> model.layers.i.post_attention_layernorm.weight
    blocks.i.mlp.w_gate.weight      -> mlp.gate_proj.weight
    blocks.i.mlp.w_up.weight        -> mlp.up_proj.weight
    blocks.i.mlp.w_down.weight      -> mlp.down_proj.weight
    ln_f.weight                     -> model.norm.weight

RoPE: frigate's rotate_half (negate-second-half + concat) is the HF/NeoX
convention, identical to Qwen3's apply_rotary_pos_emb -> **no q/k permutation**.
QK-norm order (norm-then-RoPE) also matches Qwen3.

Usage:
    uv run python -m scripts.export_gguf \
        --ckpt runs/frigate-125m-full-sft/sft_ckpt.pt \
        --tokenizer data/datasets/pirate_enhanced_full/pirate_bpe.json \
        --name frigate-125M \
        --out-dir export/gguf/frigate-125m \
        --quants Q4_K_M Q8_0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# Default location of the cloned llama.cpp (holds convert_hf_to_gguf.py).
DEFAULT_LLAMA_CPP = Path.home() / "Documents" / "GitHub" / "llama.cpp"


# --------------------------------------------------------------------------
# 1. weight remap  frigate state_dict -> Qwen3 HF state_dict
# --------------------------------------------------------------------------
def remap_to_qwen3(state: dict[str, torch.Tensor], n_layer: int, n_embd: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    def put(name: str, t: torch.Tensor) -> None:
        out[name] = t.to(torch.float16).contiguous().clone()

    put("model.embed_tokens.weight", state["wte.weight"])
    put("model.norm.weight", state["ln_f.weight"])

    for i in range(n_layer):
        p = f"blocks.{i}."
        q = f"model.layers.{i}."

        # fused QKV [3*n_embd, n_embd] -> three [n_embd, n_embd] (rows: q | k | v)
        c_attn = state[p + "attn.c_attn.weight"]
        qw, kw, vw = c_attn.split(n_embd, dim=0)
        put(q + "self_attn.q_proj.weight", qw)
        put(q + "self_attn.k_proj.weight", kw)
        put(q + "self_attn.v_proj.weight", vw)

        put(q + "self_attn.q_norm.weight", state[p + "attn.q_norm.weight"])
        put(q + "self_attn.k_norm.weight", state[p + "attn.k_norm.weight"])
        put(q + "self_attn.o_proj.weight", state[p + "attn.c_proj.weight"])

        put(q + "input_layernorm.weight", state[p + "ln_1.weight"])
        put(q + "post_attention_layernorm.weight", state[p + "ln_2.weight"])

        put(q + "mlp.gate_proj.weight", state[p + "mlp.w_gate.weight"])
        put(q + "mlp.up_proj.weight", state[p + "mlp.w_up.weight"])
        put(q + "mlp.down_proj.weight", state[p + "mlp.w_down.weight"])

    return out


# --------------------------------------------------------------------------
# 2. build the HF Qwen3 staging folder
# --------------------------------------------------------------------------
def build_hf_dir(ckpt_path: Path, tokenizer_path: Path, hf_dir: Path) -> dict:
    hf_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    state = ck["model"]  # optimizer state in ck["optimizer"] is dropped here

    n_embd = cfg.n_embd
    n_head = cfg.n_head
    head_dim = n_embd // n_head
    intermediate = state["blocks.0.mlp.w_gate.weight"].shape[0]  # SwiGLU hidden

    new_state = remap_to_qwen3(state, cfg.n_layer, n_embd)
    save_file(new_state, str(hf_dir / "model.safetensors"))

    # eos/bos = the tokenizer's <|endoftext|> id (read from the HF tokenizer.json).
    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    specials = {t["content"]: t["id"] for t in tok.get("added_tokens", []) if t.get("special")}
    eot_id = specials.get("<|endoftext|>", 0)

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": n_embd,
        "intermediate_size": int(intermediate),
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,  # no GQA in frigate
        "head_dim": head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.block_size,
        "rms_norm_eps": 1e-5,
        "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "bos_token_id": eot_id,
        "eos_token_id": eot_id,
        "torch_dtype": "float16",
    }
    (hf_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Tokenizer: pirate_bpe.json IS a HF `tokenizers` tokenizer.json. Drop it in
    # as tokenizer.json + a minimal config so AutoTokenizer (used inside the
    # convert script) loads it as a PreTrainedTokenizerFast.
    shutil.copy(tokenizer_path, hf_dir / "tokenizer.json")
    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": None,
        "model_max_length": cfg.block_size,
        "clean_up_tokenization_spaces": False,
    }
    (hf_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2) + "\n")

    return {
        "n_layer": cfg.n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "intermediate": int(intermediate),
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "eot_id": eot_id,
        "val_loss": ck.get("val_loss"),
        "stage": ck.get("stage"),
    }


# --------------------------------------------------------------------------
# 3. convert HF -> GGUF f16, auto-patching the BPE pre-tokenizer hash
# --------------------------------------------------------------------------
def _patch_pretokenizer_hash(convert_py: Path, chkhsh: str) -> None:
    """Map an unknown BPE hash -> the standard 'gpt-2' pre-tokenizer.

    pirate_bpe is byte-level GPT-2-style; only its merges differ, so the hash
    llama.cpp computes won't be in its table. The regex pre-tokenization is
    identical to gpt-2, so this mapping is correct.
    """
    # get_vocab_base_pre lives in conversion/base.py in recent llama.cpp.
    base_py = convert_py.parent / "conversion" / "base.py"
    target = base_py if base_py.exists() else convert_py
    src = target.read_text(encoding="utf-8")
    branch = f'        if chkhsh == "{chkhsh}":\n            res = "gpt-2"\n'
    anchor = "        res = None\n"  # init line at the top of get_vocab_base_pre
    if branch.strip() in src:
        return
    idx = src.index(anchor) + len(anchor)
    target.write_text(src[:idx] + branch + src[idx:], encoding="utf-8")
    print(f"  patched {target.name}: hash {chkhsh[:16]}… -> gpt-2")


def convert_to_f16(hf_dir: Path, out_gguf: Path, llama_cpp: Path) -> None:
    convert_py = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {convert_py}")

    cmd = [
        sys.executable, str(convert_py), str(hf_dir),
        "--outfile", str(out_gguf), "--outtype", "f16",
    ]
    for attempt in (1, 2):
        print(f"  convert attempt {attempt}: {' '.join(cmd[1:])}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        blob = proc.stdout + proc.stderr
        m = re.search(r"chkhsh[:\s=]+['\"]?([0-9a-f]{16,})", blob)
        if m and attempt == 1:
            _patch_pretokenizer_hash(convert_py, m.group(1))
            continue
        sys.stderr.write(blob)
        sys.exit(f"convert failed (rc={proc.returncode})")


# --------------------------------------------------------------------------
# 4. quantize
# --------------------------------------------------------------------------
def quantize(f16_gguf: Path, out_gguf: Path, quant: str) -> None:
    binary = shutil.which("llama-quantize") or "llama-quantize"
    print(f"  quantize -> {quant}: {out_gguf.name}")
    proc = subprocess.run([binary, str(f16_gguf), str(out_gguf), quant], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit(f"llama-quantize {quant} failed (rc={proc.returncode})")


def _mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--name", required=True, help="Output basename, e.g. frigate-125M")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q8_0"])
    ap.add_argument("--llama-cpp", default=str(DEFAULT_LLAMA_CPP))
    ap.add_argument("--keep-f16", action="store_true", help="Keep the intermediate f16 GGUF")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_dir = out / "hf"

    print(f"[1/4] staging HF Qwen3 folder from {args.ckpt}")
    meta = build_hf_dir(Path(args.ckpt), Path(args.tokenizer), hf_dir)
    print(f"      arch: L{meta['n_layer']} d{meta['n_embd']} h{meta['n_head']} "
          f"ffn{meta['intermediate']} vocab{meta['vocab_size']} ctx{meta['block_size']} "
          f"eot{meta['eot_id']} | val_loss={meta['val_loss']} stage={meta['stage']}")

    f16 = out / f"{args.name}-f16.gguf"
    print("[2/4] convert -> f16 GGUF")
    convert_to_f16(hf_dir, f16, Path(args.llama_cpp))
    print(f"      f16: {_mb(f16):.1f} MB")

    print(f"[3/4] quantize: {args.quants}")
    produced = []
    for qd in args.quants:
        out_q = out / f"{args.name}.{qd}.gguf"
        quantize(f16, out_q, qd)
        produced.append(out_q)

    print("[4/4] done")
    for p in produced:
        print(f"      {p.name:36s} {_mb(p):8.1f} MB")
    if not args.keep_f16:
        f16.unlink(missing_ok=True)
        shutil.rmtree(hf_dir, ignore_errors=True)
        print("      (cleaned f16 + hf staging; pass --keep-f16 to retain)")


if __name__ == "__main__":
    main()
