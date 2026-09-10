"""Corpus -> token .bin helper.

`encode_split(ds_split, tokenizer, eot_id, out_path)` streams a dataset split
into a uint16 `.bin` and returns the token count. Used by `build.py`; there is
no standalone CLI.
"""

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

BATCH = 1000


def encode_split(ds_split, tokenizer: Tokenizer, eot_id: int, out_path: Path) -> int:
    """Tokenize every row, separated by `eot_id`, into a flat uint16 .bin.

    Streams batch-by-batch to the file — no preallocation — so it is safe for
    long-form rows (full articles) of any length and stays low-memory.
    """
    n_rows = len(ds_split)
    total = 0

    with open(out_path, "wb") as f:
        for start in tqdm(range(0, n_rows, BATCH), desc=f"Tokenizing → {out_path}"):
            texts = ds_split[start : start + BATCH]["text"]
            encs = tokenizer.encode_batch(texts)
            buf: list[int] = []
            for e in encs:
                if not e.ids:
                    continue
                buf.extend(e.ids)
                buf.append(eot_id)
            if buf:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                total += len(buf)

    return total
