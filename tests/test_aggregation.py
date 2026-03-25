"""Tests for DRO aggregators."""

import torch

from dro_circuit.aggregation.aggregators import (
    CVaRAggregator,
    LocalDROAggregator,
    MaxAggregator,
    MeanAggregator,
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


class TestMeanAggregator:
    def test_shape(self):
        """Mean of (K=3, 4, 5) -> (4, 5)."""
        agg = MeanAggregator()
        scores = torch.randn(3, 4, 5)
        result = agg.aggregate(scores)
        assert result.shape == (4, 5)

    def test_mean_absolute(self):
        """Mean of absolute values along dim 0."""
        scores = torch.tensor([[[1.0, -2.0]], [[-3.0, 4.0]]])  # (2, 1, 2)
        agg = MeanAggregator(absolute=True)
        result = agg.aggregate(scores)
        expected = torch.tensor([[2.0, 3.0]])  # mean of [1,3] and [2,4]
        assert torch.allclose(result, expected)

    def test_equivalent_to_cvar_alpha_one(self):
        """MeanAggregator should give same result as CVaR(alpha=1)."""
        scores = torch.randn(5, 10, 8)
        mean_result = MeanAggregator().aggregate(scores)
        cvar_result = CVaRAggregator(alpha=1.0).aggregate(scores)
        assert torch.allclose(mean_result, cvar_result)


class TestLocalDROAggregator:
    def test_shape(self):
        """LocalDRO of (K=3, N=10, 4, 5) -> (4, 5)."""
        agg = LocalDROAggregator()
        scores = torch.randn(3, 10, 4, 5)
        result = agg.aggregate(scores)
        assert result.shape == (4, 5)

    def test_rejects_3d_input(self):
        """LocalDRO requires 4D input."""
        agg = LocalDROAggregator()
        scores = torch.randn(3, 4, 5)
        try:
            agg.aggregate(scores)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_per_example_worst_case(self):
        """For each example, take max over K, then mean over N."""
        # (K=2, N=3, 1, 1)
        scores = torch.tensor([
            [[[1.0]], [[2.0]], [[3.0]]],   # corruption 0: examples have scores 1, 2, 3
            [[[4.0]], [[1.0]], [[2.0]]],   # corruption 1: examples have scores 4, 1, 2
        ])
        agg = LocalDROAggregator(absolute=True)
        result = agg.aggregate(scores)
        # Per-example max: [max(1,4)=4, max(2,1)=2, max(3,2)=3]
        # Mean: (4+2+3)/3 = 3.0
        assert abs(result.item() - 3.0) < 1e-6

    def test_with_negative_scores(self):
        """Absolute values should be taken before max."""
        scores = torch.tensor([
            [[[-5.0]], [[1.0]]],
            [[[2.0]], [[-3.0]]],
        ])
        agg = LocalDROAggregator(absolute=True)
        result = agg.aggregate(scores)
        # abs: corruption 0 = [5, 1], corruption 1 = [2, 3]
        # per-example max: [max(5,2)=5, max(1,3)=3]
        # mean: (5+3)/2 = 4.0
        assert abs(result.item() - 4.0) < 1e-6


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

    def test_make_mean(self):
        agg = make_aggregator("mean")
        assert isinstance(agg, MeanAggregator)

    def test_make_local_dro(self):
        agg = make_aggregator("local_dro")
        assert isinstance(agg, LocalDROAggregator)

    def test_unknown_raises(self):
        try:
            make_aggregator("unknown")
            assert False, "Should have raised"
        except ValueError:
            pass
