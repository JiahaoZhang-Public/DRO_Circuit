"""Tensor storage for per-corruption edge scores."""

from typing import List

import torch


class ScoreStore:
    """
    Stores per-corruption edge scores.

    Shape: (K, n_forward, n_backward) where K = number of corruption variants.
    """

    def __init__(
        self,
        corruption_names: List[str],
        n_forward: int,
        n_backward: int,
    ):
        self.corruption_names = corruption_names
        self._name_to_idx = {name: i for i, name in enumerate(corruption_names)}
        K = len(corruption_names)
        self.scores = torch.zeros(K, n_forward, n_backward)

    @property
    def n_corruptions(self) -> int:
        """Number of corruption families (K)."""
        return len(self.corruption_names)

    def set_scores(self, corruption_name: str, scores: torch.Tensor):
        """Set scores for one corruption. scores shape: (n_forward, n_backward)."""
        idx = self._name_to_idx[corruption_name]
        self.scores[idx] = scores

    def get_scores(self, corruption_name: str) -> torch.Tensor:
        """Get scores for one corruption. Returns (n_forward, n_backward)."""
        idx = self._name_to_idx[corruption_name]
        return self.scores[idx]

    def all_scores(self) -> torch.Tensor:
        """Return all scores. Shape: (K, n_forward, n_backward)."""
        return self.scores

    def save(self, path: str):
        torch.save(
            {
                "scores": self.scores,
                "corruption_names": self.corruption_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ScoreStore":
        data = torch.load(path, weights_only=False)
        store = cls.__new__(cls)
        store.scores = data["scores"]
        store.corruption_names = data["corruption_names"]
        store._name_to_idx = {n: i for i, n in enumerate(store.corruption_names)}
        return store


class PerExampleScoreStore:
    """Stores per-example, per-corruption edge scores.

    Shape: (K, N, n_forward, n_backward) where K = corruption families, N = examples.
    """

    def __init__(
        self,
        corruption_names: List[str],
        n_examples: int,
        n_forward: int,
        n_backward: int,
    ):
        self.corruption_names = corruption_names
        self.n_examples = n_examples
        self._name_to_idx = {name: i for i, name in enumerate(corruption_names)}
        K = len(corruption_names)
        self.scores = torch.zeros(K, n_examples, n_forward, n_backward)

    @property
    def n_corruptions(self) -> int:
        """Number of corruption families (K)."""
        return len(self.corruption_names)

    def set_scores(self, corruption_name: str, scores: torch.Tensor):
        """Set scores for one corruption. scores shape: (N, n_forward, n_backward)."""
        idx = self._name_to_idx[corruption_name]
        self.scores[idx] = scores

    def get_scores(self, corruption_name: str) -> torch.Tensor:
        """Get per-example scores for one corruption. Returns (N, n_forward, n_backward)."""
        idx = self._name_to_idx[corruption_name]
        return self.scores[idx]

    def all_scores(self) -> torch.Tensor:
        """Return all scores. Shape: (K, N, n_forward, n_backward)."""
        return self.scores

    def to_aggregated(self) -> ScoreStore:
        """Reduce to ScoreStore by averaging over examples.

        Returns ScoreStore with shape (K, n_forward, n_backward).
        """
        store = ScoreStore(
            corruption_names=self.corruption_names,
            n_forward=self.scores.shape[2],
            n_backward=self.scores.shape[3],
        )
        store.scores = self.scores.mean(dim=1)  # average over N
        return store

    def save(self, path: str):
        torch.save(
            {
                "scores": self.scores,
                "corruption_names": self.corruption_names,
                "n_examples": self.n_examples,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PerExampleScoreStore":
        data = torch.load(path, weights_only=False)
        store = cls.__new__(cls)
        store.scores = data["scores"]
        store.corruption_names = data["corruption_names"]
        store.n_examples = data["n_examples"]
        store._name_to_idx = {n: i for i, n in enumerate(store.corruption_names)}
        return store
