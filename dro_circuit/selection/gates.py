"""Learnable edge gates for Plan B: differentiable circuit selection."""

import torch
import torch.nn as nn


class EdgeGates(nn.Module):
    """
    Continuous edge gates m_e in [0,1] for differentiable circuit selection.

    Uses Gumbel-sigmoid parameterization for gradient estimation during training.

    log_alpha: learnable parameters (n_forward, n_backward)
    gate = sigmoid((log_alpha + noise) / temperature)  [training]
    gate = (sigmoid(log_alpha) > 0.5).float()          [eval/hard]
    """

    def __init__(
        self,
        n_forward: int,
        n_backward: int,
        temperature: float = 0.1,
        init_value: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.log_alpha = nn.Parameter(torch.full((n_forward, n_backward), init_value))

    def forward(self, hard: bool = False) -> torch.Tensor:
        """
        Returns gate values. Shape: (n_forward, n_backward).

        Args:
            hard: If True, return binary gates (for evaluation).
        """
        if hard:
            return (torch.sigmoid(self.log_alpha) > 0.5).float()

        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-8, 1 - 1e-8)
            noise = torch.log(u) - torch.log(1 - u)
            gates = torch.sigmoid((self.log_alpha + noise) / self.temperature)
        else:
            gates = torch.sigmoid(self.log_alpha / self.temperature)

        return gates

    def l0_regularization(self) -> torch.Tensor:
        """Expected L0 norm: sum of gate-on probabilities."""
        return torch.sigmoid(self.log_alpha).sum()

    def l1_regularization(self) -> torch.Tensor:
        """L1 penalty on expected gate values."""
        return torch.sigmoid(self.log_alpha).sum()

    def n_active_edges(self) -> int:
        """Count edges with gate > 0.5."""
        with torch.no_grad():
            return int((torch.sigmoid(self.log_alpha) > 0.5).sum().item())

    def to_binary_mask(self, n_edges: int) -> torch.Tensor:
        """Extract top-n binary mask by gate magnitude."""
        with torch.no_grad():
            gate_values = torch.sigmoid(self.log_alpha)
            flat = gate_values.flatten()
            _, topk_indices = flat.topk(min(n_edges, flat.numel()))
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk_indices] = True
            return mask.reshape(self.log_alpha.shape)
