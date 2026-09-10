"""Push a fully-built data/<version>/ to a private HF dataset repo.

Lets Vast.ai instances skip the 30-minute piratization step:
  - Locally (once): build the dataset, then `python -m nanobeard.dataset_pipeline.push_to_hf`.
  - On every new GPU instance: `huggingface-cli download <repo> --local-dir data/<version>/`.

Defaults push to `younissk/nanobeard-data-<version>` as a *private* dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="e.g. data/sloop")
    parser.add_argument("--version", default=None,
                        help="Override repo suffix (default: basename of --data-dir)")
    parser.add_argument("--owner", default="younissk")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--commit-message", default="upload nanobeard dataset")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    version = args.version or data_dir.name
    repo_id = f"{args.owner}/nanobeard-data-{version}"

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=args.private)

    # Skip the raw arrow shards if you only want the tokenized bins; here we
    # ship the full pipeline output so any downstream task can use it.
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        ignore_patterns=["*.lock", "__pycache__/*"],
    )
    print(f"Pushed {data_dir} -> https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
