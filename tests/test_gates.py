"""Tests for EdgeGates."""

import torch

from dro_circuit.selection.gates import EdgeGates


class TestEdgeGates:
    def test_output_shape(self):
        gates = EdgeGates(n_forward=5, n_backward=8)
        out = gates(hard=False)
        assert out.shape == (5, 8)

    def test_hard_mode_binary(self):
        gates = EdgeGates(n_forward=5, n_backward=8, init_value=2.0)
        out = gates(hard=True)
        assert torch.all((out == 0.0) | (out == 1.0))

    def test_training_mode_continuous(self):
        gates = EdgeGates(n_forward=5, n_backward=8)
        gates.train()
        out = gates(hard=False)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_l1_regularization(self):
        gates = EdgeGates(n_forward=3, n_backward=4)
        reg = gates.l1_regularization()
        assert reg.shape == ()
        assert reg.item() >= 0

    def test_n_active_edges(self):
        gates = EdgeGates(n_forward=3, n_backward=4, init_value=5.0)
        # Large init -> all gates on
        assert gates.n_active_edges() == 12

        gates2 = EdgeGates(n_forward=3, n_backward=4, init_value=-5.0)
        # Large negative init -> all gates off
        assert gates2.n_active_edges() == 0

    def test_to_binary_mask(self):
        gates = EdgeGates(n_forward=3, n_backward=4)
        # Set known values
        with torch.no_grad():
            gates.log_alpha.fill_(-10.0)
            gates.log_alpha[0, 0] = 10.0
            gates.log_alpha[1, 1] = 10.0
            gates.log_alpha[2, 2] = 10.0

        mask = gates.to_binary_mask(n_edges=2)
        assert mask.sum().item() == 2
        assert mask.dtype == torch.bool
