"""BPE tokenizer training helper.

`train_tokenizer(ds_train, vocab_size=...)` is the reusable entry point used by
`build.py`. There is no standalone CLI — build a dataset (which trains its own
tokenizer) via `python -m nanobeard.dataset_pipeline.build`.
"""

from collections.abc import Sequence

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_tokenizer(
    ds_train: Dataset,
    vocab_size: int = 8192,
    special_tokens: Sequence[str] = ("<|endoftext|>",),
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the `text` column of a dataset split."""
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    def batch_iter(ds: Dataset, batch_size: int = 1000):
        for i in range(0, len(ds), batch_size):
            yield ds[i : i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iter(ds_train), trainer=trainer, length=len(ds_train))
    return tokenizer
