"""Tests for DRO aggregators."""

import torch

from dro_circuit.aggregation.aggregators import (
    CVaRAggregator,
    MaxAggregator,
    SoftmaxAggregator,
    make_aggregator,
)


def _make_scores(K=3, n_fwd=5, n_bwd=8):
    """Create deterministic test scores."""
    torch.manual_seed(42)
    return torch.randn(K, n_fwd, n_bwd)


class TestMaxAggregator:
    def test_shape(self):
        scores = _make_scores(K=3, n_fwd=5, n_bwd=8)
        agg = MaxAggregator(absolute=True)
        result = agg.aggregate(scores)
        assert result.shape == (5, 8)

    def test_max_absolute(self):
        scores = torch.tensor([[[1.0, -3.0]], [[2.0, 2.0]], [[-4.0, 1.0]]])  # (3, 1, 2)
        agg = MaxAggregator(absolute=True)
        result = agg.aggregate(scores)
        assert torch.allclose(result, torch.tensor([[4.0, 3.0]]))

    def test_max_no_absolute(self):
        scores = torch.tensor([[[1.0, -3.0]], [[2.0, 2.0]], [[-4.0, 1.0]]])
        agg = MaxAggregator(absolute=False)
        result = agg.aggregate(scores)
        assert torch.allclose(result, torch.tensor([[2.0, 2.0]]))


class TestCVaRAggregator:
    def test_small_alpha_approaches_max(self):
        """alpha near 0 -> n_top=1 -> just the max corruption."""
        scores = torch.tensor([[[1.0]], [[3.0]], [[2.0]]])  # (3, 1, 1)
        max_agg = MaxAggregator(absolute=True)
        # alpha=0 -> n_top=max(1, ceil(0))=1 -> top-1 = max
        cvar_agg = CVaRAggregator(alpha=0.0, absolute=True)
        assert torch.allclose(
            max_agg.aggregate(scores), cvar_agg.aggregate(scores)
        )

    def test_alpha_one_equals_mean(self):
        """alpha=1 -> n_top=K -> mean of all corruptions."""
        scores = torch.tensor([[[1.0]], [[3.0]], [[2.0]]])
        cvar_agg = CVaRAggregator(alpha=1.0, absolute=True)
        result = cvar_agg.aggregate(scores)
        expected = scores.abs().mean(dim=0)
        assert torch.allclose(result, expected)

    def test_shape(self):
        scores = _make_scores()
        agg = CVaRAggregator(alpha=0.5)
        result = agg.aggregate(scores)
        assert result.shape == (5, 8)


class TestSoftmaxAggregator:
    def test_shape(self):
        scores = _make_scores()
        agg = SoftmaxAggregator(temperature=1.0)
        result = agg.aggregate(scores)
        assert result.shape == (5, 8)

    def test_low_temperature_approaches_max(self):
        scores = torch.tensor([[[1.0]], [[10.0]], [[2.0]]])
        agg_soft = SoftmaxAggregator(temperature=0.01, absolute=True)
        agg_max = MaxAggregator(absolute=True)
        soft_result = agg_soft.aggregate(scores)
        max_result = agg_max.aggregate(scores)
        assert torch.allclose(soft_result, max_result, atol=0.1)

    def test_high_temperature_approaches_mean(self):
        scores = torch.tensor([[[1.0]], [[3.0]], [[2.0]]])
        agg = SoftmaxAggregator(temperature=100.0, absolute=True)
        result = agg.aggregate(scores)
        expected = scores.abs().mean(dim=0)
        assert torch.allclose(result, expected, atol=0.1)


class TestFactory:
    def test_make_max(self):
        agg = make_aggregator("max")
        assert isinstance(agg, MaxAggregator)

    def test_make_cvar(self):
        agg = make_aggregator("cvar", alpha=0.3)
        assert isinstance(agg, CVaRAggregator)
        assert agg.alpha == 0.3

    def test_make_softmax(self):
        agg = make_aggregator("softmax", temperature=2.0)
        assert isinstance(agg, SoftmaxAggregator)
        assert agg.temperature == 2.0

    def test_unknown_raises(self):
        try:
            make_aggregator("unknown")
            assert False, "Should have raised"
        except ValueError:
            pass
