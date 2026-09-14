"""Convert a checkpoint to a HF-ready folder and upload to the spec's HF model repo.

Run:
  uv run python -m nanobeard.publish --config configs/sloop.py
  uv run python -m nanobeard.publish --config configs/sloop.py --ckpt runs/sloop/sft_ckpt.pt
"""

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token
from safetensors.torch import save_model

from nanobeard.config import Config, load_config
from nanobeard.models import build_model, spec_for
from nanobeard.models.naming import display_name

load_dotenv()

DEFAULT_README = Path("README.md")
DEFAULT_BANNER = Path("banner.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config .py file")
    parser.add_argument(
        "--ckpt", default=None, help="Override checkpoint path (default: cfg.ckpt_path)"
    )
    parser.add_argument("--stage-dir", default="hf_release", help="Local staging dir")
    parser.add_argument("--readme", default=str(DEFAULT_README))
    parser.add_argument("--banner", default=str(DEFAULT_BANNER))
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = spec_for(cfg)
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.ckpt_path)
    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg: Config = ck["config"]

    model = build_model(ckpt_cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    save_model(model, str(stage_dir / "model.safetensors"))

    config_json = {k: getattr(ckpt_cfg, k) for k in spec.arch_fields}
    config_json["model_type"] = f"nanobeard-{spec.dispatch_key}"
    config_json["architectures"] = [spec.cls.__name__]
    config_json["codename"] = spec.codename
    config_json["model_name"] = spec.dispatch_key
    config_json["num_parameters"] = model.num_parameters()  # type: ignore[operator]
    config_json["display_name"] = display_name(ckpt_cfg, model)
    (stage_dir / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    tokenizer_src = Path(ckpt_cfg.tokenizer_path)
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, stage_dir / "pirate_bpe.json")
    else:
        print(f"  ! tokenizer not found at {tokenizer_src} — skipping")

    if Path(args.readme).exists():
        shutil.copy(args.readme, stage_dir / "README.md")
    if Path(args.banner).exists():
        shutil.copy(args.banner, stage_dir / "banner.png")

    training_meta = {
        "stage": ck.get("stage"),
        "model_name": ck.get("model_name") or ckpt_cfg.model_name,
        "codename": spec.codename,
        "iter_num": ck.get("iter_num"),
        "val_loss": ck.get("val_loss"),
        "best_val_loss": ck.get("best_val_loss"),
        "num_parameters": model.num_parameters(),  # type: ignore[operator]
        "display_name": display_name(ckpt_cfg, model),
        "full_train_config": asdict(ckpt_cfg),
    }
    (stage_dir / "training_metadata.json").write_text(
        json.dumps(training_meta, indent=2, default=str) + "\n"
    )

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not set and no cached token found. "
            "Set HF_TOKEN in your environment or .env, or run `huggingface-cli login`."
        )
    api = HfApi(token=token)
    api.create_repo(repo_id=spec.hf_repo, repo_type="model", exist_ok=True, token=token)
    commit_message = args.commit_message or f"Publish {display_name(ckpt_cfg, model)}"
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=spec.hf_repo,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Uploaded to https://huggingface.co/{spec.hf_repo}")


if __name__ == "__main__":
    main()
