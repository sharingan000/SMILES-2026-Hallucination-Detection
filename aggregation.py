"""
aggregation.py — Feature extraction from hidden states.

aggregate() is the single entry point called from solution.py.
It always returns the full feature vector regardless of the use_geometric
flag: last-token embeddings for PROJECTION_LAYERS followed by geometric
and statistical features from extract_geometric_features().
"""

from __future__ import annotations

from typing import Callable

import torch

PROJECTION_LAYERS: list[int] = list(range(12, 25))
HIDDEN_DIM: int = 896


class GeometricAggregationConfig:
    USE_KIRCHHOFF_INDEX: bool = True
    KIRCHHOFF_INDEX_LAYERS: list[int] = [14, 18, 22]
    KIRCHHOFF_INDEX_EPS: float = 1e-6

    USE_EFFECTIVE_RANK: bool = True
    EFFECTIVE_RANK_LAYERS: list[int] = [14, 16, 18, 23]

    USE_SINGULAR_VALUES: bool = True
    SINGULAR_VALUES_LAYERS: list[int] = [16, 20, 22]
    SINGULAR_VALUES_TOP_K: int = 3
    SINGULAR_VALUES_BOTTOM_K: int = 3

    USE_PER_LAYER_LAST_TOKEN_ENTROPY: bool = True
    PER_LAYER_LAST_TOKEN_ENTROPY_LAYERS: list[int] = [16, 18, 21, 23, 24]

    USE_PER_LAYER_ALL_TOKENS_MAX_ENTROPY: bool = True
    PER_LAYER_ALL_TOKENS_MAX_ENTROPY_LAYERS: list[int] = [15, 17, 19, 23, 24]

    ENTROPY_FEATURES: dict[str, bool] = {
        "softmax_entropy": True,
        "energy_entropy": True,
        "kurtosis": True,
        "sparsity": True,
    }

    USE_LAYER_NORMS: bool = True
    LAYER_NORM_LAYERS: list[int] = list(range(12, 25))

    USE_COSINE_DRIFT: bool = True
    COSINE_DRIFT_PAIRS: list[int] = list(range(24))

    @classmethod
    def feature_names(cls) -> list[str]:
        """Ordered list of geometric feature names matching ``extract_geometric_features``."""
        names: list[str] = []

        if cls.USE_KIRCHHOFF_INDEX:
            for layer_idx in cls.KIRCHHOFF_INDEX_LAYERS:
                names.append(f"kirchhoff_index_layer{layer_idx}")

        if cls.USE_EFFECTIVE_RANK:
            for layer_idx in cls.EFFECTIVE_RANK_LAYERS:
                names.append(f"effective_rank_layer{layer_idx}")

        if cls.USE_SINGULAR_VALUES:
            for layer_idx in cls.SINGULAR_VALUES_LAYERS:
                for k in range(cls.SINGULAR_VALUES_TOP_K):
                    names.append(f"sv_top{k}_layer{layer_idx}")
                for k in range(cls.SINGULAR_VALUES_BOTTOM_K):
                    names.append(f"sv_bottom{k}_layer{layer_idx}")

        if cls.USE_PER_LAYER_LAST_TOKEN_ENTROPY:
            for layer_idx in cls.PER_LAYER_LAST_TOKEN_ENTROPY_LAYERS:
                for feat_name, enabled in cls.ENTROPY_FEATURES.items():
                    if enabled:
                        names.append(f"last_token_{feat_name}_layer{layer_idx}")

        if cls.USE_PER_LAYER_ALL_TOKENS_MAX_ENTROPY:
            for layer_idx in cls.PER_LAYER_ALL_TOKENS_MAX_ENTROPY_LAYERS:
                for feat_name, enabled in cls.ENTROPY_FEATURES.items():
                    if enabled:
                        names.append(f"all_tokens_max_{feat_name}_layer{layer_idx}")

        if cls.USE_LAYER_NORMS:
            for layer_idx in cls.LAYER_NORM_LAYERS:
                names.append(f"norm_L{layer_idx}")

        if cls.USE_COSINE_DRIFT:
            for i in cls.COSINE_DRIFT_PAIRS:
                names.append(f"drift_{i}_{i+1}")

        return names

    @classmethod
    def __len__(cls) -> int:
        return len(cls.feature_names())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the full feature vector for one sample.

    Shape: (len(PROJECTION_LAYERS) * HIDDEN_DIM + len(GeometricAggregationConfig),)

    First block — last-token embeddings for each layer in PROJECTION_LAYERS.
    Second block — geometric and statistical features.

    Tensors are moved to CPU on MPS backends where torch.linalg ops are unsupported.
    """
    if hidden_states.device.type == "mps":
        hidden_states = hidden_states.cpu()
        attention_mask = attention_mask.cpu()

    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    embs = torch.hstack([*hidden_states[PROJECTION_LAYERS, last_pos, :]])
    geo  = extract_geometric_features(hidden_states, attention_mask)
    return torch.cat([embs, geo], dim=0)


def _kirchhoff_index(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    cfg = GeometricAggregationConfig
    real_tokens = attention_mask.bool()
    parts: list[torch.Tensor] = []

    for layer_idx in cfg.KIRCHHOFF_INDEX_LAYERS:
        layer = hidden_states[layer_idx][real_tokens].float()
        normed = layer / (layer.norm(p=2, dim=-1, keepdim=True) + 1e-10)
        W = (normed @ normed.T).clamp(min=0.0)
        W.fill_diagonal_(0.0)
        D = torch.diag(W.sum(dim=-1))
        L = D - W
        eigvals = torch.linalg.eigvalsh(L)
        nonzero = eigvals[eigvals > cfg.KIRCHHOFF_INDEX_EPS]
        n = layer.shape[0]
        kf = n * (1.0 / nonzero).sum() if nonzero.numel() > 0 else torch.tensor(0.0)
        parts.append(kf.unsqueeze(0) if kf.dim() == 0 else kf[:1])

    return torch.cat(parts, dim=0)


def _effective_rank(hidden_states: torch.Tensor) -> torch.Tensor:
    cfg = GeometricAggregationConfig
    parts: list[torch.Tensor] = []
    for layer_idx in cfg.EFFECTIVE_RANK_LAYERS:
        sv = torch.linalg.svdvals(hidden_states[layer_idx].float())
        p = sv**2 / (sv**2).sum()
        erank = torch.exp(-(p * torch.log(p + 1e-10)).sum())
        parts.append(erank.unsqueeze(0))
    return torch.cat(parts, dim=0)


def _singular_values(hidden_states: torch.Tensor) -> torch.Tensor:
    cfg = GeometricAggregationConfig
    parts: list[torch.Tensor] = []
    for layer_idx in cfg.SINGULAR_VALUES_LAYERS:
        sv = torch.linalg.svdvals(hidden_states[layer_idx].float())
        parts.append(sv[: cfg.SINGULAR_VALUES_TOP_K])
        parts.append(sv[-cfg.SINGULAR_VALUES_BOTTOM_K :])
    return torch.cat(parts, dim=0)


def _softmax_entropy(emb: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(emb, dim=-1)
    h = -(p * torch.log(p + 1e-10)).sum(dim=-1)
    return h if emb.dim() > 1 else h.unsqueeze(0)


def _energy_entropy(emb: torch.Tensor) -> torch.Tensor:
    energy = emb**2
    p = energy / (energy.sum(dim=-1, keepdim=True) + 1e-10)
    h = -(p * torch.log(p + 1e-10)).sum(dim=-1)
    return h if emb.dim() > 1 else h.unsqueeze(0)


def _kurtosis(emb: torch.Tensor) -> torch.Tensor:
    mean = emb.mean(dim=-1, keepdim=True)
    std = emb.std(dim=-1, keepdim=True) + 1e-10
    k = (((emb - mean) / std) ** 4).mean(dim=-1) - 3.0
    return k if emb.dim() > 1 else k.unsqueeze(0)


def _sparsity(emb: torch.Tensor) -> torch.Tensor:
    l1 = emb.abs().sum(dim=-1)
    l2 = emb.norm(p=2, dim=-1) + 1e-10
    s = l1 / l2
    return s if emb.dim() > 1 else s.unsqueeze(0)


_PER_LAYER_FEATURE_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softmax_entropy": _softmax_entropy,
    "energy_entropy": _energy_entropy,
    "kurtosis": _kurtosis,
    "sparsity": _sparsity,
}


def _per_layer_last_token_entropy(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    cfg = GeometricAggregationConfig
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    parts: list[torch.Tensor] = []

    for layer_idx in cfg.PER_LAYER_LAST_TOKEN_ENTROPY_LAYERS:
        emb = hidden_states[layer_idx, last_pos, :].float()
        for name, fn in _PER_LAYER_FEATURE_FNS.items():
            if cfg.ENTROPY_FEATURES.get(name):
                parts.append(fn(emb))

    return torch.cat(parts, dim=0)


def _per_layer_all_tokens_max_entropy(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    cfg = GeometricAggregationConfig
    real_tokens = attention_mask.bool()
    parts: list[torch.Tensor] = []

    for layer_idx in cfg.PER_LAYER_ALL_TOKENS_MAX_ENTROPY_LAYERS:
        layer = hidden_states[layer_idx][real_tokens].float()
        layer = layer[len(layer) // 3 * 2 :]
        token_feats = torch.stack(
            [
                fn(layer)
                for name, fn in _PER_LAYER_FEATURE_FNS.items()
                if cfg.ENTROPY_FEATURES.get(name)
            ],
            dim=1,
        )
        parts.append(token_feats.max(dim=0).values)

    return torch.cat(parts, dim=0)


def _layer_norms(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """L2 norm of the last-token embedding for each layer in LAYER_NORM_LAYERS."""
    cfg = GeometricAggregationConfig
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    layers = torch.tensor(cfg.LAYER_NORM_LAYERS, dtype=torch.long)
    return hidden_states[layers, last_pos, :].float().norm(p=2, dim=-1)


def _cosine_drift(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between adjacent-layer last-token embeddings."""
    cfg = GeometricAggregationConfig
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    parts: list[torch.Tensor] = []

    for i in cfg.COSINE_DRIFT_PAIRS:
        a = hidden_states[i,     last_pos, :].float()
        b = hidden_states[i + 1, last_pos, :].float()
        cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-10)
        parts.append(cos.unsqueeze(0))

    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract all hand-crafted features from hidden states.

    Feature groups (controlled by GeometricAggregationConfig flags):
      - Kirchhoff index of the token similarity graph
      - Effective rank of the activation matrix
      - Top-K and bottom-K singular values
      - Per-layer last-token entropy statistics
      - Per-layer all-tokens max entropy statistics
      - L2 norms of the last-token embedding per layer
      - Cosine similarity between adjacent-layer last-token embeddings

    The returned tensor length is fixed for all samples and matches
    ``len(GeometricAggregationConfig)``.
    """
    if hidden_states.device.type == "mps":
        hidden_states = hidden_states.cpu()
        attention_mask = attention_mask.cpu()
    parts: list[torch.Tensor] = []

    if GeometricAggregationConfig.USE_KIRCHHOFF_INDEX:
        parts.append(_kirchhoff_index(hidden_states, attention_mask))

    if GeometricAggregationConfig.USE_EFFECTIVE_RANK:
        parts.append(_effective_rank(hidden_states))

    if GeometricAggregationConfig.USE_SINGULAR_VALUES:
        parts.append(_singular_values(hidden_states))

    if GeometricAggregationConfig.USE_PER_LAYER_LAST_TOKEN_ENTROPY:
        parts.append(_per_layer_last_token_entropy(hidden_states, attention_mask))

    if GeometricAggregationConfig.USE_PER_LAYER_ALL_TOKENS_MAX_ENTROPY:
        parts.append(_per_layer_all_tokens_max_entropy(hidden_states, attention_mask))

    if GeometricAggregationConfig.USE_LAYER_NORMS:
        parts.append(_layer_norms(hidden_states, attention_mask))

    if GeometricAggregationConfig.USE_COSINE_DRIFT:
        parts.append(_cosine_drift(hidden_states, attention_mask))

    if not parts:
        return torch.zeros(0)

    return torch.cat(parts, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Entry point called from solution.py.

    Geometric features are always included regardless of use_geometric.
    The flag is kept only for API compatibility.
    """
    _ = use_geometric
    return aggregate(hidden_states, attention_mask)
