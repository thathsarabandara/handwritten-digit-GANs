# utils/__init__.py
"""Utility package initialization.

Provides common helper functions such as setting random seeds for reproducibility.
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across `random`, `numpy`, and `torch`.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior on CUDA (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
