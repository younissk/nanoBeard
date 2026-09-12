"""Merge a LoRA adapter into the base weights and hand off to GGUF.

    uv run --group finetune python -m nanobeard.finetune.merge \
        --adapter runs/lora/pirate-v1 --out runs/lora/pirate-v1-merged --gguf

Merging matters because llama.cpp does not load PEFT adapters: the phone runs a
single GGUF, so the adapter has to be folded back into the base weights first.

The GGUF step deliberately shells out to llama.cpp's own
`convert_hf_to_gguf.py` under its own interpreter. The merged model is a stock
Qwen3 checkpoint, so unlike the frigate path there is no remapping to do — and
the converter's `transformers==5.5.1` pin cannot coexist with this project's
`tokenizers>=0.23.1`, which is exactly why --converter-python exists.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

DEFAULT_LLAMA_CPP = Path.home() / "Documents" / "GitHub" / "llama.cpp"


def _load_export_gguf():
    """hf/ holds standalone scripts, not an importable package.

    Reusing export_gguf's interpreter resolution rather than copying it keeps one
    answer to "which python runs the converter" — the rule is fiddly enough
    (flag, then env var, then a venv in the checkout, then sys.executable) that
    a second copy would drift.
    """
    import importlib.util

    src = Path(__file__).resolve().parents[3] / "hf" / "export_gguf.py"
    spec = importlib.util.spec_from_file_location("nanobeard_export_gguf", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge(adapter: Path, out: Path, base: str | None = None, dtype: str = "bfloat16") -> Path:
    """Fold the adapter into the base weights.

    bfloat16, not float32: at fp32 a 0.6B model is ~2.4GB of weights plus
    another copy while saving, which is enough to take down an 8GB laptop — it
    did. The GGUF converter emits f16 from these weights regardless, so the
    extra precision buys nothing downstream.
    """
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = PeftConfig.from_pretrained(str(adapter))
    base_id = base or cfg.base_model_name_or_path
    print(f"base={base_id}\nadapter={adapter}")

    torch_dtype = getattr(torch, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch_dtype, low_cpu_mem_usage=True
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter))
    # nn.Module.__getattr__ is typed `Tensor | Module`, so the merge helper is
    # invisible to the checker without a cast.
    merged = cast(Any, peft_model).merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="500MB")
    # The tokenizer saved next to the adapter wins: it is the one the data was
    # rendered with, and a mismatch here is the silent-gibberish failure mode.
    src = adapter if (adapter / "tokenizer.json").exists() else Path(base_id)
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
    _inline_chat_template(out)
    print(f"merged -> {out}")
    return out


def _inline_chat_template(out: Path) -> None:
    """Copy chat_template.jinja into tokenizer_config.json.

    transformers>=5 saves the template as a separate .jinja file, but
    convert_hf_to_gguf.py only reads `chat_template` inside tokenizer_config.json.
    Without this the GGUF ships with no template at all and llama-server silently
    falls back to a generic ChatML one — which is close enough to Qwen3 that
    nothing looks broken, while `<think>` handling and the tool-call template are
    quietly wrong. Worse on a phone, where there is no server flag to patch it.
    """
    import json

    jinja = out / "chat_template.jinja"
    cfg_path = out / "tokenizer_config.json"
    if not (jinja.exists() and cfg_path.exists()):
        return
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("chat_template"):
        return
    cfg["chat_template"] = jinja.read_text()
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("  inlined chat_template into tokenizer_config.json")


def to_gguf(hf_dir: Path, out_gguf: Path, llama_cpp: Path, python: str | None) -> None:
    convert = llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {convert} — pass --llama-cpp")

    eg = _load_export_gguf()
    interpreter = eg.resolve_converter_python(python, llama_cpp)
    eg.check_converter_deps(interpreter)
    cmd = [interpreter, str(convert), str(hf_dir), "--outfile", str(out_gguf), "--outtype", "f16"]
    print(f"convert: {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit(f"convert failed (rc={proc.returncode})")
    print(f"gguf -> {out_gguf} ({out_gguf.stat().st_size / 1e6:.0f} MB)")


def quantize(f16: Path, out: Path, quant: str) -> None:
    binary = shutil.which("llama-quantize") or "llama-quantize"
    proc = subprocess.run([binary, str(f16), str(out), quant], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit(f"quantize failed (rc={proc.returncode})")
    print(f"{quant} -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default=None, help="Merged HF dir (default: <adapter>-merged)")
    ap.add_argument("--base", default=None, help="Override the adapter's recorded base model")
    ap.add_argument("--gguf", action="store_true", help="Also convert to GGUF")
    ap.add_argument("--quants", nargs="*", default=["Q4_K_M"])
    ap.add_argument("--llama-cpp", default=str(DEFAULT_LLAMA_CPP))
    ap.add_argument("--converter-python", default=None)
    ap.add_argument("--keep-f16", action="store_true")
    ap.add_argument("--dtype", default="bfloat16",
                    help="Merge precision. float32 doubles peak RAM for no gain: "
                         "the GGUF converter emits f16 either way.")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    out = Path(args.out) if args.out else adapter.with_name(adapter.name + "-merged")
    merge(adapter, out, args.base, dtype=args.dtype)

    if not args.gguf:
        return
    f16 = out / f"{out.name}-f16.gguf"
    to_gguf(out, f16, Path(args.llama_cpp), args.converter_python)
    for q in args.quants:
        quantize(f16, out / f"{out.name}.{q}.gguf", q)
    if not args.keep_f16:
        f16.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
