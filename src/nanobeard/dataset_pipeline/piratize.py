from multiprocessing import Pool, cpu_count

from arrr import translate
from datasets import Dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from nanobeard.env import load_env

load_env()
console = Console()


def _translate_one(text: str) -> str:
    return translate(text)


def _piratize_gen(dataset: Dataset, split_name: str, chunk_size: int, n_proc: int):
    """Top-level generator (must be picklable for Dataset.from_generator's
    fingerprint). Yields one pirate-translated row at a time, pulling input in
    chunks so peak memory stays ~one chunk."""
    total = len(dataset)
    with (
        Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
        Pool(n_proc) as pool,
    ):
        task = progress.add_task(f"⚓ {split_name}", total=total)
        for start in range(0, total, chunk_size):
            texts = dataset[start : start + chunk_size]["text"]
            for pirate in pool.imap(_translate_one, texts, chunksize=64):
                yield {"text": pirate}
                progress.advance(task)


def piratize(dataset: Dataset, split_name: str = "", chunk_size: int = 10_000) -> Dataset:
    """Translate every `text` row to pirate-speak via `arrr`, in parallel.

    Memory-bounded: `Dataset.from_generator` writes the streamed rows straight
    to disk (Arrow), so peak RAM is ~one chunk rather than two full copies of
    the split. Output order matches input (`imap` is ordered)."""
    print(f"total {len(dataset):,}")
    return Dataset.from_generator(
        _piratize_gen,
        gen_kwargs={
            "dataset": dataset,
            "split_name": split_name,
            "chunk_size": chunk_size,
            "n_proc": max(1, cpu_count() - 1),
        },
    )
