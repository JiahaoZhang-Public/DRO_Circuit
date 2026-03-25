"""Wraps EAP-IG attribution to score edges per corruption variant."""

import sys
from pathlib import Path
from typing import Callable, Literal

import torch

_VENDOR_EAP = str(Path(__file__).resolve().parents[2] / "vendor" / "EAP-IG" / "src")
if _VENDOR_EAP not in sys.path:
    sys.path.insert(0, _VENDOR_EAP)

from eap.attribute import attribute
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.data.eap_adapter import make_eap_dataloader
from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset
from dro_circuit.scoring.score_store import PerExampleScoreStore, ScoreStore


class PerCorruptionScorer:
    """
    Runs EAP-IG attribution independently per corruption variant.

    For each corruption c_k, calls attribute(model, graph, dataloader_k, metric, method)
    to produce per-edge scores, then stores them in a ScoreStore.
    """

    def __init__(
        self,
        model: HookedTransformer,
        method: Literal[
            "EAP", "EAP-IG-inputs", "EAP-IG-activations", "clean-corrupted"
        ] = "EAP-IG-inputs",
        ig_steps: int = 5,
        intervention: Literal["patching", "zero", "mean"] = "patching",
        aggregation: str = "sum",
        batch_size: int = 32,
        quiet: bool = False,
    ):
        self.model = model
        self.method = method
        self.ig_steps = ig_steps
        self.intervention = intervention
        self.aggregation = aggregation
        self.batch_size = batch_size
        self.quiet = quiet

    def score_all_corruptions(
        self,
        dataset: MultiCorruptDataset,
        metric: Callable,
    ) -> ScoreStore:
        """
        Score edges under each corruption variant.

        For each corruption c_k:
          1. Build DataLoader with (clean, corrupt_k, labels)
          2. Create fresh Graph (same structure, zeroed scores)
          3. Run attribute() -> fills graph.scores
          4. Store graph.scores into ScoreStore[k]

        Returns:
            ScoreStore with shape (K, n_forward, n_backward)
        """
        # Get graph dimensions from model
        graph_template = Graph.from_model(self.model)
        store = ScoreStore(
            corruption_names=dataset.corruption_names,
            n_forward=graph_template.n_forward,
            n_backward=graph_template.n_backward,
        )

        for corruption_name in dataset.corruption_names:
            if not self.quiet:
                print(f"  Scoring corruption: {corruption_name}")

            # Fresh graph for this corruption
            graph = Graph.from_model(self.model)

            # Build single-corruption DataLoader
            dataloader = make_eap_dataloader(dataset, corruption_name, self.batch_size)

            # Run EAP-IG attribution
            ig_steps = self.ig_steps if "IG" in self.method else None
            attribute(
                self.model,
                graph,
                dataloader,
                metric,
                method=self.method,
                intervention=self.intervention,
                aggregation=self.aggregation,
                ig_steps=ig_steps,
                quiet=self.quiet,
            )

            # Store results
            store.set_scores(corruption_name, graph.scores.cpu().clone())

        return store

    def score_all_corruptions_per_example(
        self,
        dataset: MultiCorruptDataset,
        metric: Callable,
    ) -> PerExampleScoreStore:
        """
        Score edges under each corruption variant, retaining per-example scores.

        Uses EAP's per_example mode to return scores of shape (N, n_forward, n_backward)
        for each corruption, stored in a PerExampleScoreStore of shape (K, N, n_fwd, n_bwd).

        Only supports method='EAP'.

        Returns:
            PerExampleScoreStore with shape (K, N, n_forward, n_backward)
        """
        if self.method != "EAP":
            raise ValueError(
                f"per-example scoring only supports method='EAP', got '{self.method}'"
            )

        graph_template = Graph.from_model(self.model)
        store = PerExampleScoreStore(
            corruption_names=dataset.corruption_names,
            n_examples=len(dataset),
            n_forward=graph_template.n_forward,
            n_backward=graph_template.n_backward,
        )

        for corruption_name in dataset.corruption_names:
            if not self.quiet:
                print(f"  Scoring corruption (per-example): {corruption_name}")

            graph = Graph.from_model(self.model)
            dataloader = make_eap_dataloader(dataset, corruption_name, self.batch_size)

            # attribute() with per_example=True returns (N, n_fwd, n_bwd) directly
            per_example_scores = attribute(
                self.model,
                graph,
                dataloader,
                metric,
                method="EAP",
                intervention=self.intervention,
                aggregation=self.aggregation,
                per_example=True,
                quiet=self.quiet,
            )

            store.set_scores(corruption_name, per_example_scores.cpu())

        return store
