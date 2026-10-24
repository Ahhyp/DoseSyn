"""Monotherapy supervision policies.

Two policies are supported:

* ``known_monotherapy`` (S1/S2): all measured mono curves are allowed as
  auxiliary supervision.  They are *known prior labels*, never an independent
  monotherapy test set.  This is disclosed in every split metadata file.
* ``strict_cell_isolation`` (S3): only monotherapy curves belonging to training
  cells are trainable.  Test/validation cell labels are held out and cannot
  enter training or any scaler fit.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

POLICIES = ("known_monotherapy", "strict_cell_isolation")


def build_mono_assignments(
    mono_index: pd.DataFrame,
    condition_splits: pd.DataFrame,
    policy: str,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Create per-curve mono supervision assignments for one split."""
    if policy not in POLICIES:
        raise ValueError(f"unknown mono policy: {policy}")

    cell_roles: Dict[str, List[str]] = (
        condition_splits.groupby("cell_id")["split"]
        .agg(lambda s: sorted(set(s)))
        .to_dict()
    )

    rows: List[Dict[str, object]] = []
    for r in mono_index.itertuples(index=False):
        roles = cell_roles.get(r.cell_id, [])
        if policy == "strict_cell_isolation":
            if "train" in roles and not ({"val", "test"} & set(roles)):
                trainable = True
                access = "train"
                reason = "cell in training fold"
            else:
                trainable = False
                access = "held_out"
                reason = "cell is validation/test; strict unseen-cell policy"
        else:
            trainable = True
            access = "known_prior"
            reason = "known_monotherapy: measured drug-cell prior, not an independent test"

        rows.append(
            {
                "mono_curve_id": r.mono_curve_id,
                "mono_condition_id": r.mono_condition_id,
                "cell_id": r.cell_id,
                "cell_line": r.cell_line,
                "depmap_id": r.depmap_id,
                "disease": getattr(r, "disease", None),
                "drug_id": r.drug_id,
                "drug_name": r.drug_name,
                "n_points": int(r.n_points) if hasattr(r, "n_points") else None,
                "combo_roles": "|".join(roles) if roles else "none",
                "label_access": access,
                "trainable": bool(trainable),
                "policy": policy,
                "reason": reason,
            }
        )
    out = pd.DataFrame(rows)
    if out["mono_curve_id"].duplicated().any():
        raise ValueError("duplicate mono_curve_id in mono assignments")
    return out


def trainable_curve_ids(assignments: pd.DataFrame) -> List[str]:
    return sorted(assignments.loc[assignments["trainable"], "mono_curve_id"].tolist())


def held_out_cells(assignments: pd.DataFrame) -> Iterable[str]:
    held = assignments.loc[~assignments["trainable"], "cell_id"].unique()
    return sorted(held)
