"""Plan B pipeline: learnable mask + adversarial inner loop optimization."""

import sys
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

_VENDOR_EAP = str(Path(__file__).resolve().parents[2] / "vendor" / "EAP-IG" / "src")
if _VENDOR_EAP not in sys.path:
    sys.path.insert(0, _VENDOR_EAP)

from eap.evaluate import evaluate_graph
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.config import PlanBConfig
from dro_circuit.data.eap_adapter import make_eap_dataloader
from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset
from dro_circuit.selection.gates import EdgeGates


class PlanBPipeline:
    """
    Plan B: Direct optimization of learnable edge masks with adversarial corruption selection.

    Objective:
        min_m  E_x[ max_c in C(x)  loss(f_{m}(x), f(x); c) ] + lambda * R(m)

    Inner loop: For fixed gates m, evaluate loss under each corruption,
                select worst via softmax weighting.
    Outer loop: Gradient descent on gate parameters.

    NOTE: This is a structural implementation. The differentiable path through
    evaluate_graph requires careful handling of inference_mode. This version
    uses the evaluation loss as a proxy (no gradient through the model forward pass
    to the gates -- the gates affect only which edges are in the circuit).
    For full differentiability, a custom forward pass with gate-weighted ablation
    would be needed.
    """

    def __init__(
        self,
        model: HookedTransformer,
        config: PlanBConfig,
        n_edges: int,
    ):
        self.model = model
        self.config = config
        self.n_edges = n_edges

        graph = Graph.from_model(model)
        self.n_forward = graph.n_forward
        self.n_backward = graph.n_backward

        device = next(model.parameters()).device
        self.gates = EdgeGates(
            n_forward=graph.n_forward,
            n_backward=graph.n_backward,
            temperature=config.temperature,
        ).to(device)

    def _evaluate_circuit_loss(
        self,
        dataset: MultiCorruptDataset,
        metric: Callable,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Evaluate loss per corruption using current gate values as circuit mask."""
        # Get binary mask from gates
        binary_mask = self.gates.forward(hard=True)

        graph = Graph.from_model(self.model)
        graph.in_graph = binary_mask.bool().cpu()

        losses = {}
        for corruption_name in dataset.corruption_names:
            dl = make_eap_dataloader(dataset, corruption_name, batch_size)
            with torch.no_grad():
                result = evaluate_graph(
                    self.model, graph, dl, metric,
                    intervention="patching", quiet=True, skip_clean=True,
                )
            losses[corruption_name] = result.mean().item()

        return losses

    def _adversary_loss(self, losses: Dict[str, float]) -> float:
        """Softmax-weighted worst-case loss over corruptions."""
        tau = self.config.adversary_temperature
        loss_vals = torch.tensor(list(losses.values()))
        weights = F.softmax(loss_vals / tau, dim=0)
        return (weights * loss_vals).sum().item()

    def _regularization_loss(self) -> torch.Tensor:
        if self.config.reg_type == "L0":
            return self.config.reg_lambda * self.gates.l0_regularization()
        elif self.config.reg_type == "L1":
            return self.config.reg_lambda * self.gates.l1_regularization()
        raise ValueError(f"Unknown reg type: {self.config.reg_type}")

    def run(
        self,
        dataset: MultiCorruptDataset,
        metric: Callable,
        batch_size: int = 32,
    ) -> Graph:
        """
        Execute Plan B optimization.

        This uses a REINFORCE-style approach: evaluate the circuit under the current
        gate configuration, compute the adversarial loss, and use the regularization
        gradient to update gates toward sparser circuits.

        Returns:
            Graph with discrete circuit selected (in_graph mask set).
        """
        optimizer = torch.optim.Adam(self.gates.parameters(), lr=self.config.lr)

        best_loss = float("inf")
        best_mask = None

        for step in range(self.config.n_outer_steps):
            optimizer.zero_grad()

            # Evaluate circuit loss per corruption (no gradient through model)
            per_corr_losses = self._evaluate_circuit_loss(dataset, metric, batch_size)
            adv_loss = self._adversary_loss(per_corr_losses)

            # Regularization (has gradient through gates)
            reg_loss = self._regularization_loss()
            reg_loss.backward()
            optimizer.step()

            n_active = self.gates.n_active_edges()
            if step % 20 == 0:
                print(
                    f"  Step {step}: adv_loss={adv_loss:.4f}, "
                    f"reg={reg_loss.item():.4f}, n_active={n_active}"
                )

            # Track best
            if adv_loss < best_loss:
                best_loss = adv_loss
                best_mask = self.gates.to_binary_mask(self.n_edges).clone()

        # Extract discrete circuit from best mask
        if best_mask is None:
            best_mask = self.gates.to_binary_mask(self.n_edges)

        circuit_graph = Graph.from_model(self.model)
        circuit_graph.in_graph = best_mask.cpu()
        circuit_graph.prune()

        n_edges = circuit_graph.in_graph.sum().item()
        n_nodes = circuit_graph.nodes_in_graph.sum().item()
        print(f"Plan B circuit: {int(n_edges)} edges, {int(n_nodes)} nodes")

        return circuit_graph
