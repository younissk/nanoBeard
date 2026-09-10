"""Bundle nanobeard/ into space/ and upload to the HF Space.

The Space app imports `from nanobeard.config import Config` and
`from nanobeard.models import build_model, MODEL_REGISTRY`. To avoid network
deps at Space build time we ship the package as a local subdirectory rather
than pip-installing it from git.
"""

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "younissk/nanoBeard-playground"
SPACE_DIR = Path("hf/space")
ASSETS_DIR = SPACE_DIR / "assets"
PACKAGE_SRC = Path("src/nanobeard")
PACKAGE_DST = SPACE_DIR / "nanobeard"

ASSET_COPIES = [
    (Path("hf/nanoBeard.png"), ASSETS_DIR / "nanoBeard.png"),
]


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    # Refresh bundled package.
    if PACKAGE_DST.exists():
        shutil.rmtree(PACKAGE_DST)
    shutil.copytree(
        PACKAGE_SRC,
        PACKAGE_DST,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"copied {PACKAGE_SRC} -> {PACKAGE_DST}")

    for src, dst in ASSET_COPIES:
        if src.exists():
            shutil.copy(src, dst)
            print(f"copied {src} -> {dst}")

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Update nanoBeard playground (multi-version)",
    )
    print(f"Uploaded to https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
