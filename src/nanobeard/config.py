import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ----- Run identity -----
    run_name: str = "nanobeard-sloop"

    # ----- Model dispatch (key into nanobeard.models.MODEL_REGISTRY) -----
    model_name: str = "sloop"

    # ----- Paths (everything else derives from these two) -----
    data_dir: str = "data/datasets/tiny_pirate_stories"  # holds train.bin, val.bin, tokenizer
    run_dir: str = "runs/sloop"  # holds ckpt.pt, sft_ckpt.pt

    # ----- Model architecture -----
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # ----- Architecture flags (Frigate+; Sloop/Galleon leave all False) -----
    use_rope: bool = False  # RoPE rotary pos-emb instead of learned wpe table
    use_swiglu: bool = False  # SwiGLU FFN (8/3 expansion) instead of GELU MLP
    use_rmsnorm: bool = False  # RMSNorm instead of LayerNorm
    use_qk_norm: bool = False  # per-head RMSNorm on Q and K (stabilizes depth)
    rope_theta: float = 10000.0  # RoPE base frequency

    # ----- Optimizer -----
    optimizer: str = "adamw"  # "adamw" | "muon"
    learning_rate: float = 3e-4  # AdamW peak LR (and fallback LR under Muon)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Muon (only used when optimizer == "muon"); embeddings/head/norms stay AdamW.
    muon_lr: float = 0.02  # Muon peak LR — much higher than AdamW's
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5  # Newton-Schulz iterations per step

    # ----- LR schedule -----
    warmup_iters: int = 200
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5

    # ----- Training loop -----
    epochs: float = 1.0  # passes over train.bin; resolves to max_iters at train start
    max_iters: int = 20000  # hard ceiling on iters (and the smoke-run cap)
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10

    # ----- System -----
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False
    seed: int = 1337

    # ----- HF Hub -----
    hf_model_repo: str = "younissk/nanoBeard"  # final published model repo
    hf_ckpt_repo: str | None = None  # rolling ckpt sync repo (private)
    pretrained_ckpt_repo: str | None = None  # SFT source: repo holding pretraining ckpt.pt
    hf_private: bool = True
    resume: bool = True

    # ----- W&B -----
    wandb_project: str | None = None
    wandb_entity: str | None = None

    # ----- Derived paths (do not set directly) -----
    @property
    def train_bin(self) -> str:
        return os.path.join(self.data_dir, "train.bin")

    @property
    def val_bin(self) -> str:
        return os.path.join(self.data_dir, "val.bin")

    @property
    def tokenizer_path(self) -> str:
        return os.path.join(self.data_dir, "pirate_bpe.json")

    @property
    def out_dir(self) -> str:
        """Back-compat alias for run_dir."""
        return self.run_dir

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "ckpt.pt")

    @property
    def sft_ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "sft_ckpt.pt")


def load_config(path: str) -> Config:
    """Load a Config from a Python file that defines a `config` module-level variable
    or a `make_config()` function. E.g. `configs/sloop.py`."""
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_config"):
        return mod.make_config()
    if hasattr(mod, "config"):
        return mod.config
    raise AttributeError(
        f"{path} must define `config = Config(...)` or `def make_config() -> Config`"
    )
