"""M7-B dose-position masks.

Flat index convention is row-major over a 4x4 dose grid:
``index = row_idx * 4 + col_idx`` where row = dose A rank and col = dose B rank.
All masks are symmetric under transpose (row <-> col), so swapping the two drugs
maps every observed/held position onto an observed/held position.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

GRID = 4
N_POINTS = GRID * GRID


def _sorted(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(v) for v in values))


MASK_SPECS: Dict[str, Dict[str, object]] = {
    "center_holdout_75": {
        "held": _sorted([1 * 4 + 1, 1 * 4 + 2, 2 * 4 + 1, 2 * 4 + 2]),
        "description": "中心 2x2 四点留出，其余 12 点可观测（插值主实验）",
        "kind": "interpolation",
    },
    "checkerboard_holdout_50": {
        "held": _sorted([i * 4 + j for i in range(4) for j in range(4) if (i + j) % 2 == 1]),
        "description": "棋盘交错留出 8 点，观测 8 点（50% observed）",
        "kind": "sparsity",
    },
    "corners_holdout_25": {
        "held": _sorted([i * 4 + j for i in range(4) for j in range(4)
                         if (i, j) not in {(0, 0), (0, 3), (3, 0), (3, 3)}]),
        "description": "仅保留四角 4 点观测，留出 12 点（25% observed）",
        "kind": "sparsity",
    },
    "boundary_holdout_50": {
        "held": _sorted(set([3 * 4 + j for j in range(4)] + [i * 4 + 3 for i in range(4)])),
        "description": "最高剂量 A 整行 + 最高剂量 B 整列，留出 7 点（边界外推）",
        "kind": "boundary",
    },
}

for _name, _spec in MASK_SPECS.items():
    _held = set(_spec["held"])
    _obs = tuple(i for i in range(N_POINTS) if i not in _held)
    _spec["observed"] = _obs
    # transpose symmetry: (i,j)->(j,i)
    assert all((i % 4) * 4 + (i // 4) in _held for i in _held), _name
    assert len(_obs) + len(_held) == N_POINTS, _name
    _spec["n_observed"] = len(_obs)
    _spec["n_held"] = len(_held)


def mask_vector(name: str, *, observed: bool = False) -> np.ndarray:
    spec = MASK_SPECS[name]
    base = np.zeros(N_POINTS, dtype=np.float32)
    idx = spec["observed"] if observed else spec["held"]
    base[list(idx)] = 1.0
    return base


def mask_matrix(name: str) -> np.ndarray:
    return mask_vector(name).reshape(GRID, GRID)


def observed_mask(name: str, n: int) -> np.ndarray:
    return np.tile(mask_vector(name, observed=True), (n, 1))


def held_mask(name: str, n: int) -> np.ndarray:
    return np.tile(mask_vector(name), (n, 1))


def indices(name: str, *, observed: bool = False) -> List[int]:
    return list(MASK_SPECS[name]["observed" if observed else "held"])


def mask_version(name: str) -> str:
    payload = json.dumps({"name": name, "held": list(MASK_SPECS[name]["held"])}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe(name: str) -> Dict[str, object]:
    spec = MASK_SPECS[name]
    return {
        "name": name,
        "held_indices": list(spec["held"]),
        "observed_indices": list(spec["observed"]),
        "n_held": spec["n_held"],
        "n_observed": spec["n_observed"],
        "description": spec["description"],
        "kind": spec["kind"],
        "version": mask_version(name),
        "held_grid": [list(map(int, np.where(mask_matrix(name) == 1)[0])),
                      list(map(int, np.where(mask_matrix(name) == 1)[1]))],
    }


def all_masks() -> Dict[str, Dict[str, object]]:
    return {name: describe(name) for name in MASK_SPECS}
