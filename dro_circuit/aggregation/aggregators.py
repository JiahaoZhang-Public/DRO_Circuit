"""DRO aggregators: Max, CVaR, Softmax over corruption dimension."""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


def normalize_per_corruption(scores: torch.Tensor, method: str = "max") -> torch.Tensor:
    """Normalize scores per corruption so all corruptions are on the same scale.

    This prevents aggregators (especially Max) from being dominated by the
    corruption family with the largest raw score magnitudes.

    Args:
        scores: shape (K, ...) where dim-0 is the corruption dimension.
        method: normalization method:
            - "max": divide each corruption's scores by its max absolute value → [-1, 1]
            - "l2": divide by L2 norm → unit-length score vectors
            - "zscore": standardize to zero mean, unit variance

    Returns:
        Normalized scores of the same shape.
    """
    K = scores.shape[0]
    # Flatten everything after dim-0 for per-corruption stats
    flat = scores.reshape(K, -1)

    if method == "max":
        # Each corruption scaled to [-1, 1]
        scale = flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
        flat_normed = flat / scale
    elif method == "l2":
        scale = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-10)
        flat_normed = flat / scale
    elif method == "zscore":
        mu = flat.mean(dim=1, keepdim=True)
        sigma = flat.std(dim=1, keepdim=True).clamp(min=1e-10)
        flat_normed = (flat - mu) / sigma
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return flat_normed.reshape(scores.shape)


class DROAggregator(ABC):
    """Aggregates per-corruption scores into a single robust score per edge."""

    @abstractmethod
    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: shape (K, n_forward, n_backward) -- per-corruption scores.

        Returns:
            Tensor (n_forward, n_backward) -- aggregated robust scores.
        """
        ...


class MaxAggregator(DROAggregator):
    """
    DRO aggregation: s_e = max_k |ŝ_e^{(k)}|

    An edge is important if it matters under ANY corruption.
    When normalize is set, scores are normalized per corruption before aggregation
    to prevent domination by the corruption with the largest raw magnitudes.
    """

    def __init__(self, absolute: bool = True, normalize: str | None = None):
        self.absolute = absolute
        self.normalize = normalize

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            scores = normalize_per_corruption(scores, method=self.normalize)
        if self.absolute:
            scores = scores.abs()
        return scores.max(dim=0).values


class CVaRAggregator(DROAggregator):
    """
    CVaR aggregation: average of top-alpha fraction of |ŝ_e^{(k)}|.

    alpha near 0 -> approaches max (just worst corruption).
    alpha=1 -> mean (all corruptions).
    """

    def __init__(self, alpha: float = 0.5, absolute: bool = True, normalize: str | None = None):
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.absolute = absolute
        self.normalize = normalize

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        K = scores.shape[0]
        if self.normalize:
            scores = normalize_per_corruption(scores, method=self.normalize)
        if self.absolute:
            scores = scores.abs()

        sorted_scores, _ = scores.sort(dim=0, descending=True)
        n_top = max(1, int(torch.ceil(torch.tensor(self.alpha * K)).item()))
        top_scores = sorted_scores[:n_top]
        return top_scores.mean(dim=0)


class MeanAggregator(DROAggregator):
    """
    ERM aggregation: s_e = mean_k |s_e^{(k)}|

    Treats all corruption families equally (average-case).
    Equivalent to CVaR with alpha=1.
    """

    def __init__(self, absolute: bool = True, normalize: str | None = None):
        self.absolute = absolute
        self.normalize = normalize

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            scores = normalize_per_corruption(scores, method=self.normalize)
        if self.absolute:
            scores = scores.abs()
        return scores.mean(dim=0)


class LocalDROAggregator(DROAggregator):
    """
    Per-example worst-case (Local DRO) aggregation.

    Requires per-example scores of shape (K, N, n_forward, n_backward).
    For each example, takes the max over corruptions, then averages over examples.

    S_DRO_local(e) = (1/N) sum_i max_k |s(e; x_i, x̃_ik)|
    """

    def __init__(self, absolute: bool = True, normalize: str | None = None):
        self.absolute = absolute
        self.normalize = normalize

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: shape (K, N, n_forward, n_backward) -- per-example, per-corruption scores.

        Returns:
            Tensor (n_forward, n_backward) -- aggregated robust scores.
        """
        if scores.ndim != 4:
            raise ValueError(
                f"LocalDROAggregator requires per-example scores of shape "
                f"(K, N, n_forward, n_backward), got {scores.shape}"
            )
        if self.normalize:
            scores = normalize_per_corruption(scores, method=self.normalize)
        if self.absolute:
            scores = scores.abs()
        per_example_worst = scores.max(dim=0).values  # (N, n_forward, n_backward)
        return per_example_worst.mean(dim=0)  # (n_forward, n_backward)


class SoftmaxAggregator(DROAggregator):
    """
    Softmax-weighted aggregation: sum_k softmax(|s_k|/tau) * |s_k|.

    tau -> 0: approaches max. tau -> inf: approaches mean.
    """

    def __init__(self, temperature: float = 1.0, absolute: bool = True, normalize: str | None = None):
        self.temperature = temperature
        self.absolute = absolute
        self.normalize = normalize

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            scores = normalize_per_corruption(scores, method=self.normalize)
        if self.absolute:
            abs_scores = scores.abs()
        else:
            abs_scores = scores

        weights = F.softmax(abs_scores / self.temperature, dim=0)
        return (weights * abs_scores).sum(dim=0)


def make_aggregator(name: str, **kwargs) -> DROAggregator:
    """Factory for creating aggregators by name."""
    # Filter out None values from kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    registry = {
        "max": MaxAggregator,
        "mean": MeanAggregator,
        "local_dro": LocalDROAggregator,
        "cvar": CVaRAggregator,
        "softmax": SoftmaxAggregator,
    }
    if name not in registry:
        raise ValueError(f"Unknown aggregator: {name}. Choose from {list(registry.keys())}")
    return registry[name](**kwargs)
