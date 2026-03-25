"""Tests for ScoreStore."""

import os
import tempfile

import torch

from dro_circuit.scoring.score_store import PerExampleScoreStore, ScoreStore


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


class TestPerExampleScoreStore:
    def test_init_shape(self):
        store = PerExampleScoreStore(["a", "b"], n_examples=10, n_forward=5, n_backward=3)
        assert store.scores.shape == (2, 10, 5, 3)
        assert store.n_corruptions == 2
        assert store.n_examples == 10

    def test_set_get(self):
        store = PerExampleScoreStore(["a", "b"], n_examples=4, n_forward=2, n_backward=2)
        data = torch.randn(4, 2, 2)
        store.set_scores("a", data)
        result = store.get_scores("a")
        assert torch.equal(data, result)

    def test_all_scores(self):
        store = PerExampleScoreStore(["a", "b"], n_examples=4, n_forward=2, n_backward=2)
        assert store.all_scores().shape == (2, 4, 2, 2)

    def test_to_aggregated(self):
        store = PerExampleScoreStore(["a", "b"], n_examples=4, n_forward=2, n_backward=2)
        store.set_scores("a", torch.ones(4, 2, 2))
        store.set_scores("b", torch.ones(4, 2, 2) * 2)
        agg = store.to_aggregated()
        assert agg.scores.shape == (2, 2, 2)
        assert torch.allclose(agg.get_scores("a"), torch.ones(2, 2))
        assert torch.allclose(agg.get_scores("b"), torch.ones(2, 2) * 2)

    def test_save_load(self):
        store = PerExampleScoreStore(["x", "y"], n_examples=3, n_forward=2, n_backward=2)
        store.set_scores("x", torch.randn(3, 2, 2))
        store.set_scores("y", torch.randn(3, 2, 2))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            store.save(path)
            loaded = PerExampleScoreStore.load(path)
            assert torch.equal(store.scores, loaded.scores)
            assert store.corruption_names == loaded.corruption_names
            assert store.n_examples == loaded.n_examples
        finally:
            os.unlink(path)
