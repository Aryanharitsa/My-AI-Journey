from __future__ import annotations

import os
import random

import numpy as np
import torch

from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)

# Default seed = 1729 (Hardy-Ramanujan number).
DEFAULT_SEED = 1729


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed torch, numpy, random, and the Python hash seed.

    Sets ``torch.backends.cudnn.deterministic = True`` and disables the cuDNN
    benchmark when CUDA is present, which trades some throughput for
    reproducibility. Returns the seed actually applied.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    _log.info("seed.set value=%d", seed)
    return seed
