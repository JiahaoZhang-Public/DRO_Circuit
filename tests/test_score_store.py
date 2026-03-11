"""Tests for ScoreStore."""

import os
import tempfile

import torch

from dro_circuit.scoring.score_store import ScoreStore


class TestScoreStore:
    def test_init_shape(self):
        store = ScoreStore(
            corruption_names=["a", "b", "c"],
            n_forward=5,
            n_backward=8,
        )
        assert store.scores.shape == (3, 5, 8)
        assert store.n_corruptions == 3

    def test_set_get(self):
        store = ScoreStore(["a", "b"], n_forward=3, n_backward=4)
        scores_a = torch.randn(3, 4)
        store.set_scores("a", scores_a)
        retrieved = store.get_scores("a")
        assert torch.allclose(retrieved, scores_a)

    def test_all_scores(self):
        store = ScoreStore(["a", "b"], n_forward=3, n_backward=4)
        all_s = store.all_scores()
        assert all_s.shape == (2, 3, 4)

    def test_save_load(self):
        store = ScoreStore(["x", "y", "z"], n_forward=5, n_backward=8)
        store.set_scores("x", torch.randn(5, 8))
        store.set_scores("y", torch.randn(5, 8))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            store.save(path)
            loaded = ScoreStore.load(path)
            assert loaded.corruption_names == store.corruption_names
            assert torch.allclose(loaded.scores, store.scores)
        finally:
            os.unlink(path)
