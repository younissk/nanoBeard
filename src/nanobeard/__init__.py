"""nanoBeard — tiny pirate LLM, multi-version.

Backwards-compatibility shim: old checkpoints pickle Config under
`training.config.Config`. Map those module paths to the new package so
`torch.load(...)` keeps working on legacy ckpts.
"""

import sys

from nanobeard import config as _config

sys.modules.setdefault("training", sys.modules[__name__])
sys.modules.setdefault("training.config", _config)
