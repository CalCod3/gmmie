"""
research.seed — single source of truth for determinism.

Call `seed_everywhere(seed)` at the top of any entry point that should be
reproducible. Touches Python's `random`, NumPy, PyTorch (CPU & CUDA), and
disables non-deterministic CuDNN algos.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

logger = logging.getLogger(__name__)


def seed_everywhere(seed: int, *, deterministic_cudnn: bool = True,
                    log: bool = True) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    if log:
        logger.info("seed_everywhere(%d)", seed)
    return seed
