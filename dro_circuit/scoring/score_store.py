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
