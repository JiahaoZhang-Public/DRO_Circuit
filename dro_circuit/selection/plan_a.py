"""Plan A pipeline: Score -> DRO Aggregate -> Greedy/TopN selection."""

import sys
from pathlib import Path
from typing import Callable, Tuple

_VENDOR_EAP = str(Path(__file__).resolve().parents[2] / "vendor" / "EAP-IG" / "src")
if _VENDOR_EAP not in sys.path:
    sys.path.insert(0, _VENDOR_EAP)

from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.aggregation.aggregators import DROAggregator, make_aggregator
from dro_circuit.config import ExperimentConfig
from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset
from dro_circuit.scoring.per_corruption_scorer import PerCorruptionScorer
from dro_circuit.scoring.score_store import ScoreStore


class PlanAPipeline:
    """
    Plan A: Score -> Aggregate -> Greedy/TopN.

    Steps:
      1. For each corruption c_k, run EAP-IG -> per-edge scores s_e^{(k)}
      2. DRO aggregation: s_e = Agg_k(s_e^{(k)})
      3. Write aggregated scores to Graph.scores
      4. Apply greedy or topn selection
      5. Return circuit Graph with in_graph mask set
    """

    def __init__(
        self,
        model: HookedTransformer,
        scorer: PerCorruptionScorer,
        aggregator: DROAggregator,
        n_edges: int,
        selection_method: str = "topn",
        absolute: bool = True,
    ):
        self.model = model
        self.scorer = scorer
        self.aggregator = aggregator
        self.n_edges = n_edges
        self.selection_method = selection_method
        self.absolute = absolute

    def run(
        self,
        dataset: MultiCorruptDataset,
        metric: Callable,
    ) -> Tuple[Graph, ScoreStore]:
        """
        Execute the full Plan A pipeline.

        Returns:
            (graph, score_store): graph has the circuit selected (in_graph mask),
            score_store contains per-corruption scores for analysis.
        """
        # Step 1: Score per corruption
        print("Step 1: Scoring edges per corruption...")
        score_store = self.scorer.score_all_corruptions(dataset, metric)

        # Step 2: DRO aggregation
        print("Step 2: DRO aggregation over corruptions...")
        aggregated_scores = self.aggregator.aggregate(score_store.all_scores())

        # Step 3: Write to graph
        circuit_graph = Graph.from_model(self.model)
        circuit_graph.scores = aggregated_scores.to(circuit_graph.scores.device)

        # Step 4: Selection
        print(f"Step 3: Selecting top {self.n_edges} edges via {self.selection_method}...")
        if self.selection_method == "topn":
            circuit_graph.apply_topn(self.n_edges, absolute=self.absolute)
        elif self.selection_method == "greedy":
            circuit_graph.apply_greedy(self.n_edges, absolute=self.absolute)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        n_edges_in = circuit_graph.in_graph.sum().item()
        n_nodes_in = circuit_graph.nodes_in_graph.sum().item()
        print(f"Circuit: {int(n_edges_in)} edges, {int(n_nodes_in)} nodes")

        return circuit_graph, score_store

    @classmethod
    def from_config(
        cls,
        model: HookedTransformer,
        config: ExperimentConfig,
    ) -> "PlanAPipeline":
        """Build pipeline from experiment config."""
        scorer = PerCorruptionScorer(
            model=model,
            method=config.scoring.method,
            ig_steps=config.scoring.ig_steps,
            intervention=config.scoring.intervention,
            aggregation=config.scoring.aggregation,
            batch_size=config.scoring.batch_size,
        )

        agg_kwargs = {}
        if config.dro.aggregator == "cvar":
            agg_kwargs["alpha"] = config.dro.cvar_alpha
        elif config.dro.aggregator == "softmax":
            agg_kwargs["temperature"] = config.dro.softmax_temperature
        aggregator = make_aggregator(config.dro.aggregator, **agg_kwargs)

        return cls(
            model=model,
            scorer=scorer,
            aggregator=aggregator,
            n_edges=config.selection.n_edges,
            selection_method=config.selection.selection_method,
            absolute=config.selection.absolute,
        )
