"""
probe.py — Hallucination probe classifier.

Two CatBoost models trained on geometric + LDA-projection features:
  - Accuracy model  -> used by ``predict``      (primary competition metric)
  - ROC-AUC model   -> used by ``predict_proba`` (AUC ranking)

LDA projection directions are fit exclusively on the training fold inside
``fit``, so no label information leaks into validation or test splits.
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from aggregation import PROJECTION_LAYERS, HIDDEN_DIM, GeometricAggregationConfig

_N_PROJ_LAYERS: int   = len(PROJECTION_LAYERS)
_N_EMBED: int         = _N_PROJ_LAYERS * HIDDEN_DIM
_GEO_NAMES: list[str] = GeometricAggregationConfig.feature_names()

# ── Accuracy model ────────────────────────────────────────────────────────────

_ACC_PROJ_LAYERS: list[int] = [16, 18, 20, 22, 24]

_ACC_SELECTED: list[str] = [
    "drift_22_23",
    "all_tokens_max_kurtosis_layer15",
    "last_token_energy_entropy_layer23",
    "drift_10_11",
    "last_token_softmax_entropy_layer23",
    "drift_18_19",
    "norm_L12",
    "norm_L13",
    "proj_L22",
    "last_token_softmax_entropy_layer16",
    "drift_21_22",
    "all_tokens_max_kurtosis_layer17",
    "last_token_softmax_entropy_layer18",
    "drift_4_5",
    "kirchhoff_index_layer22",
    "norm_L22",
    "norm_L23",
    "drift_5_6",
    "last_token_kurtosis_layer21",
    "proj_L20",
    "drift_12_13",
    "last_token_kurtosis_layer23",
    "drift_9_10",
    "drift_14_15",
    "last_token_sparsity_layer21",
    "norm_L21",
    "drift_6_7",
    "drift_11_12",
    "last_token_energy_entropy_layer21",
    "kirchhoff_index_layer14",
    "norm_L17",
    "last_token_sparsity_layer23",
    "sv_top2_layer16",
]

_ACC_PARAMS: dict = {
    "iterations":         700,
    "depth":              5,
    "learning_rate":      0.0049993886084806595,
    "l2_leaf_reg":        14.648708318190316,
    "min_data_in_leaf":   1,
    "random_strength":    2.8286832461254536,
    "bootstrap_type":     "Bernoulli",
    "subsample":          0.5258730541344544,
    "auto_class_weights": "SqrtBalanced",
    "random_seed":        42,
    "verbose":            0,
    "task_type":          "CPU",
}

# ── ROC-AUC model ─────────────────────────────────────────────────────────────

_AUC_PROJ_LAYERS: list[int] = list(range(12, 25))

_AUC_SELECTED: list[str] = [
    "proj_L23",
    "proj_L12",
    "last_token_softmax_entropy_layer23",
    "kirchhoff_index_layer18",
    "proj_L13",
    "all_tokens_max_softmax_entropy_layer24",
    "last_token_sparsity_layer21",
    "all_tokens_max_energy_entropy_layer17",
    "last_token_softmax_entropy_layer16",
    "last_token_softmax_entropy_layer18",
    "sv_bottom0_layer22",
    "effective_rank_layer14",
    "all_tokens_max_kurtosis_layer15",
    "all_tokens_max_sparsity_layer15",
    "proj_L18",
    "kirchhoff_index_layer22",
    "all_tokens_max_energy_entropy_layer24",
    "last_token_kurtosis_layer21",
    "proj_L17",
    "sv_top1_layer20",
    "last_token_kurtosis_layer18",
    "last_token_energy_entropy_layer16",
    "last_token_kurtosis_layer23",
    "last_token_sparsity_layer23",
    "effective_rank_layer18",
    "all_tokens_max_softmax_entropy_layer15",
    "proj_L19",
    "last_token_kurtosis_layer24",
    "all_tokens_max_kurtosis_layer17",
    "proj_L16",
    "all_tokens_max_energy_entropy_layer19",
    "all_tokens_max_energy_entropy_layer15",
    "all_tokens_max_sparsity_layer23",
    "last_token_kurtosis_layer16",
    "sv_bottom0_layer16",
    "sv_bottom0_layer20",
    "effective_rank_layer16",
    "sv_top2_layer20",
    "sv_top2_layer16",
    "sv_bottom2_layer20",
    "effective_rank_layer23",
]

_AUC_PARAMS: dict = {
    "iterations":         120,
    "depth":              4,
    "learning_rate":      0.021906311083664507,
    "l2_leaf_reg":        5.778195217708458,
    "min_data_in_leaf":   8,
    "random_strength":    0.09636606300153401,
    "bootstrap_type":     "Bernoulli",
    "subsample":          0.7100323701372399,
    "auto_class_weights": "SqrtBalanced",
    "random_seed":        42,
    "verbose":            0,
    "task_type":          "CPU",
}


# ═════════════════════════════════════════════════════════════════════════════

class HallucinationProbe:
    """Binary classifier that detects hallucinations from hidden-state features.

    Feature vector layout (produced by ``aggregation_and_feature_extraction``
    with ``use_geometric=True``):
      X[:, :_N_EMBED]  — last-token embeddings for layers in PROJECTION_LAYERS,
                         used to compute per-layer LDA projection scalars.
      X[:, _N_EMBED:]  — 102 geometric / statistical features from
                         GeometricAggregationConfig.

    ``predict``       uses the accuracy-optimised CatBoost model.
    ``predict_proba`` uses the ROC-AUC-optimised CatBoost model.
    """

    def __init__(self) -> None:
        self._directions: np.ndarray | None = None
        self._acc_clf: CatBoostClassifier | None = None
        self._auc_clf: CatBoostClassifier | None = None
        self._threshold: float = 0.5

    def _compute_directions(
        self, X_emb: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        embs = X_emb.reshape(-1, _N_PROJ_LAYERS, HIDDEN_DIM)
        directions = np.zeros((_N_PROJ_LAYERS, HIDDEN_DIM), dtype=np.float32)
        for i in range(_N_PROJ_LAYERS):
            layer_embs = embs[:, i, :]
            mean_h = layer_embs[y == 1].mean(axis=0)
            mean_n = layer_embs[y == 0].mean(axis=0)
            d = mean_h - mean_n
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d /= norm
            directions[i] = d
        return directions

    def _build_features(
        self,
        X: np.ndarray,
        proj_layers: list[int],
        selected: list[str],
    ) -> np.ndarray:
        n = len(X)
        embs = X[:, :_N_EMBED].reshape(n, _N_PROJ_LAYERS, HIDDEN_DIM)
        X_geo = X[:, _N_EMBED:]

        proj_vals: dict[str, np.ndarray] = {}
        for i, layer_idx in enumerate(PROJECTION_LAYERS):
            if layer_idx in proj_layers:
                proj_vals[f"proj_L{layer_idx}"] = (
                    embs[:, i, :] * self._directions[i]
                ).sum(axis=-1)

        geo_vals: dict[str, np.ndarray] = {
            name: X_geo[:, col] for col, name in enumerate(_GEO_NAMES)
        }

        all_vals = {**proj_vals, **geo_vals}
        return np.column_stack([all_vals[name] for name in selected])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        y = y.astype(int)
        self._directions = self._compute_directions(X[:, :_N_EMBED], y)

        X_acc = self._build_features(X, _ACC_PROJ_LAYERS, _ACC_SELECTED)
        self._acc_clf = CatBoostClassifier(**_ACC_PARAMS)
        self._acc_clf.fit(X_acc, y)

        X_auc = self._build_features(X, _AUC_PROJ_LAYERS, _AUC_SELECTED)
        self._auc_clf = CatBoostClassifier(**_AUC_PARAMS)
        self._auc_clf.fit(X_auc, y)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_acc = self._build_features(X, _ACC_PROJ_LAYERS, _ACC_SELECTED)
        probs = self._acc_clf.predict_proba(X_acc)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_auc = self._build_features(X, _AUC_PROJ_LAYERS, _AUC_SELECTED)
        probs = self._auc_clf.predict_proba(X_auc)[:, 1]
        return np.stack([1.0 - probs, probs], axis=1)
