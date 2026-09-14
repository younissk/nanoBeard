"""Tokenizer fingerprint used to gate ckpt loading.

Silent tokenizer drift between training and inference is the single most
expensive bug class for tiny LMs — the model emits gibberish and nothing in
the loss / logs tells you why. Hashing the tokenizer file at train time and
re-checking at load time turns that into a loud failure.
"""

from __future__ import annotations

import hashlib
import os

CHUNK = 1 << 16


def hash_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 hex digest of `path`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class TokenizerMismatch(RuntimeError):
    """Raised when a checkpoint's tokenizer hash does not match the on-disk one."""


def verify_match(tokenizer_path: str | os.PathLike[str], expected: str | None) -> str:
    """Return the actual hash; raise if `expected` is set and differs.

    A `None` expected value means the ckpt predates hash gating — return the
    actual hash but do not raise. Callers can decide whether to warn.
    """
    actual = hash_file(tokenizer_path)
    if expected is not None and expected != actual:
        raise TokenizerMismatch(
            f"Tokenizer mismatch:\n"
            f"  ckpt expected: {expected}\n"
            f"  on-disk      : {actual}\n"
            f"  path         : {tokenizer_path}\n"
            f"This ckpt was trained against a different tokenizer. Either point at "
            f"the right pirate_bpe.json or retrain."
        )
    return actual
