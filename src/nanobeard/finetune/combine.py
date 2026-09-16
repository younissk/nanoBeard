"""Combine two LoRA adapters into one model.

    uv run --group finetune python -m nanobeard.finetune.combine \
        --adapters runs/lora/pirate-v4 runs/rl/search-v3 \
        --weights 1.0 1.0 --out runs/lora/pirate-search

Two adapters trained separately on the same base do not simply add up: each was
fitted assuming the other was absent, and both write into the same projections.
PEFT's `add_weighted_adapter` offers three ways to reconcile them, and which one
wins is an empirical question, not a settled one:

  linear  weighted sum of the deltas. Cheapest. Interference shows up as both
          skills degrading at once.
  cat     concatenate the low-rank factors, so the result has rank r1+r2 and
          neither adapter's subspace is overwritten. Exact for the sum of the
          two deltas; the adapter doubles in size.
  svd     sum the deltas and re-factor back down to rank r. Keeps the size,
          loses whatever the truncation discards.

`cat` is the default here because it is the only one that does not throw
information away, and at r=16 the size cost is irrelevant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

BASE_MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="One per adapter (default: 1.0 each)")
    ap.add_argument("--combination-type", default="cat", choices=("cat", "linear", "svd"))
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    weights = args.weights or [1.0] * len(args.adapters)
    if len(weights) != len(args.adapters):
        raise SystemExit(f"{len(args.adapters)} adapters but {len(weights)} weights")

    loaded: Any = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    names = []
    model: Any = None
    for i, path in enumerate(args.adapters):
        name = f"a{i}"
        names.append(name)
        if model is None:
            model = PeftModel.from_pretrained(loaded, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        print(f"loaded {path} as {name} (weight {weights[i]})")

    merged_name = "combined"
    cast(Any, model).add_weighted_adapter(
        adapters=names, weights=weights, adapter_name=merged_name,
        combination_type=args.combination_type,
    )
    model.set_adapter(merged_name)
    print(f"combined with {args.combination_type}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Save only the combined adapter; the inputs stay where they are.
    model.save_pretrained(str(out), selected_adapters=[merged_name])
    # PEFT nests the save under the adapter name; hoist it so the directory
    # looks like every other adapter and `merge.py --adapter` just works.
    nested = out / merged_name
    if nested.is_dir():
        for f in nested.iterdir():
            f.rename(out / f.name)
        nested.rmdir()

    src = Path(args.adapters[0])
    tok_src = src if (src / "tokenizer.json").exists() else Path(args.base)
    AutoTokenizer.from_pretrained(str(tok_src)).save_pretrained(str(out))
    (out / "combine_args.json").write_text(json.dumps(vars(args), indent=2))

    cfg = json.loads((out / "adapter_config.json").read_text())
    print(f"saved to {out} (rank {cfg.get('r')})")


if __name__ == "__main__":
    main()
