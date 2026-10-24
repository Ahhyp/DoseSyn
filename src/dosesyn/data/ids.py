"""Stable, human-readable identifiers for every DoseSyn entity.

All IDs are deterministic functions of *content* (never of row order), so the
same experimental entity receives the same ID on every rerun.  The human
readable prefix makes logs and joins self-documenting.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from ..utils.io import make_id, slugify

DRUG_PREFIX = "DRUG"
CELL_PREFIX = "CELL"
PAIR_PREFIX = "PAIR"
COND_PREFIX = "COND"
MATRIX_PREFIX = "MTX"
MONO_CURVE_PREFIX = "MONO"
MONO_COND_PREFIX = "MCOND"


def resolve_name_ids(names: Iterable[str], prefix: str) -> Dict[str, str]:
    """Map raw entity names to ``PREFIX_SLUG`` IDs, disambiguating collisions.

    Two different raw names can slugify to the same token (e.g. ``5-FU`` and
    ``5 FU``); collisions get a short hash suffix instead of being merged.
    """
    unique = sorted({str(n).strip() for n in names})
    by_slug: Dict[str, List[str]] = {}
    for name in unique:
        by_slug.setdefault(slugify(name), []).append(name)
    mapping: Dict[str, str] = {}
    for name in unique:
        slug = slugify(name)
        if len(by_slug[slug]) == 1:
            mapping[name] = f"{prefix}_{slug}"
        else:
            mapping[name] = f"{prefix}_{slug}_{make_id('', [name], length=6).strip('_')}"
    return mapping


def drug_ids(names: Iterable[str]) -> Dict[str, str]:
    return resolve_name_ids(names, DRUG_PREFIX)


def cell_ids(names: Iterable[str]) -> Dict[str, str]:
    return resolve_name_ids(names, CELL_PREFIX)


def unordered_pair_id(drug_a_id: str, drug_b_id: str) -> Tuple[str, str, str]:
    """Return (pair_id, low_drug_id, high_drug_id) for an unordered pair."""
    low, high = sorted((drug_a_id, drug_b_id))
    return make_id(PAIR_PREFIX, [low, high]), low, high


def combo_condition_id(pair_id: str, cell_id: str) -> str:
    return make_id(COND_PREFIX, [pair_id, cell_id])


def mono_condition_id(drug_id: str, cell_id: str) -> str:
    return make_id(MONO_COND_PREFIX, [drug_id, cell_id])


def dose_grid_token(values: Sequence[float]) -> str:
    """Canonical, order-independent token for a dose grid (µM)."""
    return "|".join(f"{float(v):.12g}" for v in sorted(float(x) for x in values))


def matrix_id(
    batch_id: int,
    cell_id: str,
    drug_a_id: str,
    drug_b_id: str,
    grid_a: Sequence[float],
    grid_b: Sequence[float],
) -> str:
    return make_id(
        MATRIX_PREFIX,
        [int(batch_id), cell_id, drug_a_id, drug_b_id, dose_grid_token(grid_a), dose_grid_token(grid_b)],
    )


def mono_curve_id(batch_id: int, cell_id: str, drug_id: str, doses: Sequence[float]) -> str:
    return make_id(
        MONO_CURVE_PREFIX,
        [int(batch_id), cell_id, drug_id, dose_grid_token(doses)],
    )
