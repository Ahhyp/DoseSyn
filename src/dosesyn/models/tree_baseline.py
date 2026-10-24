"""B1 Tree-Dose baseline: cell features + dose coordinates -> X/X0."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..training.m4_data import MatrixTable


class TreeDoseBaseline:
    model_name = "b1_tree"
    needs_graphs = False
    needs_drug_ids = False
    is_neural = False

    def __init__(
        self,
        *,
        max_iter: int = 400,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        random_state: int = 13,
    ) -> None:
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.random_state = int(random_state)
        self.model: Optional[HistGradientBoostingRegressor] = None

    def _features(self, table: MatrixTable) -> np.ndarray:
        n = len(table)
        cell = np.repeat(table.cell_features, 16, axis=0)
        da = np.repeat(table.dose_a_std, 16, axis=0).reshape(n * 16, 4)
        # repeat per point: each matrix row has 4 a-values and 4 b-values
        a = np.tile(table.dose_a_std.reshape(n, 1, 4), (1, 16, 1)).reshape(n * 16, 4)
        b = np.tile(table.dose_b_std.reshape(n, 1, 4), (1, 16, 1)).reshape(n * 16, 4)
        za = np.tile(table.dose_a_zero.reshape(n, 1, 4), (1, 16, 1)).reshape(n * 16, 4)
        zb = np.tile(table.dose_b_zero.reshape(n, 1, 4), (1, 16, 1)).reshape(n * 16, 4)
        return np.concatenate([cell, a, b, za, zb], axis=1).astype(np.float32)

    def fit(self, table: MatrixTable) -> "TreeDoseBaseline":
        x = self._features(table)
        y = table.y.reshape(-1)
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.model.fit(x, y)
        return self

    def predict(self, table: MatrixTable) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        pred = self.model.predict(self._features(table))
        return pred.reshape(-1, 4, 4).astype(np.float64)

    def n_parameters(self) -> int:
        # Number of tree nodes is not a parameter count; report leaf count instead.
        if self.model is None:
            return 0
        return int(
            sum(len(tree.nodes) for stage in self.model._predictors for tree in stage)
        )
