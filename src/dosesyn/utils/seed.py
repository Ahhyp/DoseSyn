"""Determinism helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG this project can touch (models are out of scope for M0/M1)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
