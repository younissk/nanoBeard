"""Model registry. Add new versions by editing MODEL_REGISTRY below."""

from dataclasses import dataclass

import torch.nn as nn

from nanobeard.config import Config

from .frigate import ARCH_FIELDS as FRIGATE_ARCH_FIELDS
from .frigate import GPT as GPTFrigate
from .sloop import ARCH_FIELDS as SLOOP_ARCH_FIELDS
from .sloop import GPT as GPTSloop


@dataclass(frozen=True)
class ModelSpec:
    dispatch_key: str  # Config.model_name, ckpt key
    codename: str  # display name
    hf_repo: str  # final published HF model repo
    cls: type[nn.Module]  # model class
    arch_fields: tuple[str, ...]  # fields to persist into config.json


# ---- Single source of truth. Change codename / hf_repo here. ----
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "sloop": ModelSpec(
        dispatch_key="sloop",
        codename="Sloop",
        hf_repo="younissk/nanoBeard",
        cls=GPTSloop,
        arch_fields=SLOOP_ARCH_FIELDS,
    ),
    "frigate": ModelSpec(
        dispatch_key="frigate",
        codename="Frigate",
        hf_repo="younissk/nanoBeard-frigate-126M",
        cls=GPTFrigate,
        arch_fields=FRIGATE_ARCH_FIELDS,
    ),
}


def build_model(cfg: Config) -> nn.Module:
    return MODEL_REGISTRY[cfg.model_name].cls(cfg)


def spec_for(cfg: Config) -> ModelSpec:
    return MODEL_REGISTRY[cfg.model_name]
