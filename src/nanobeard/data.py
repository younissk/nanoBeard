import numpy as np
import torch

from nanobeard.config import Config


def get_batch(split: str, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from train.bin or val.bin."""
    if split == "train":
        path = config.train_bin
    elif split == "val":
        path = config.val_bin
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # Re-create memmap each call to avoid a memory leak in long-running loops.
    data = np.memmap(path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - config.block_size - 1, (config.batch_size,))

    x = torch.stack(
        [torch.from_numpy(data[i : i + config.block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + config.block_size].astype(np.int64)) for i in ix]
    )

    if config.device == "cuda":
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)

    return x, y
