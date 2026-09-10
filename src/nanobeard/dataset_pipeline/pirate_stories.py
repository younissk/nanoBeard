"""Print a random piratized story from a cached source.

Run:
  uv run python -m nanobeard.dataset_pipeline.pirate_stories --source tiny_stories_pirate
"""

import argparse
import random

from datasets import load_from_disk

from nanobeard.dataset_pipeline.sources import source_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="tiny_stories_pirate", help="Source name under data/sources/"
    )
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = load_from_disk(str(source_dir(args.source)))
    split = ds[args.split]
    i = random.randint(0, len(split) - 1)
    print(split[i]["text"])


if __name__ == "__main__":
    main()
