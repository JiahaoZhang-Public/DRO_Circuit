"""DRO aggregators: Max, CVaR, Softmax over corruption dimension."""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


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
    DRO aggregation: s_e = max_k |s_e^{(k)}|

    An edge is important if it matters under ANY corruption.
    """

    def __init__(self, absolute: bool = True):
        self.absolute = absolute

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.absolute:
            scores = scores.abs()
        return scores.max(dim=0).values


class CVaRAggregator(DROAggregator):
    """
    CVaR aggregation: average of top-alpha fraction of |s_e^{(k)}|.

    alpha near 0 -> approaches max (just worst corruption).
    alpha=1 -> mean (all corruptions).
    """

    def __init__(self, alpha: float = 0.5, absolute: bool = True):
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.absolute = absolute

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        K = scores.shape[0]
        if self.absolute:
            scores = scores.abs()

        sorted_scores, _ = scores.sort(dim=0, descending=True)
        n_top = max(1, int(torch.ceil(torch.tensor(self.alpha * K)).item()))
        top_scores = sorted_scores[:n_top]
        return top_scores.mean(dim=0)


class SoftmaxAggregator(DROAggregator):
    """
    Softmax-weighted aggregation: sum_k softmax(|s_k|/tau) * |s_k|.

    tau -> 0: approaches max. tau -> inf: approaches mean.
    """

    def __init__(self, temperature: float = 1.0, absolute: bool = True):
        self.temperature = temperature
        self.absolute = absolute

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
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
        "cvar": CVaRAggregator,
        "softmax": SoftmaxAggregator,
    }
    if name not in registry:
        raise ValueError(f"Unknown aggregator: {name}. Choose from {list(registry.keys())}")
    return registry[name](**kwargs)
