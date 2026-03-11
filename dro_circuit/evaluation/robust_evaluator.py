"""Robust circuit evaluation under multiple corruption variants."""

import sys
from pathlib import Path
from typing import Callable, Dict, Literal

import torch

_VENDOR_EAP = str(Path(__file__).resolve().parents[2] / "vendor" / "EAP-IG" / "src")
if _VENDOR_EAP not in sys.path:
    sys.path.insert(0, _VENDOR_EAP)

from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.data.eap_adapter import make_eap_dataloader
from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset


@torch.no_grad()
def evaluate_robust(
    model: HookedTransformer,
    graph: Graph,
    dataset: MultiCorruptDataset,
    metric: Callable,
    intervention: Literal["patching", "zero", "mean"] = "patching",
    batch_size: int = 32,
    quiet: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a circuit under every corruption variant.

    Returns:
        Dict mapping corruption_name -> mean metric value.
    """
    results = {}
    for corruption_name in dataset.corruption_names:
        dl = make_eap_dataloader(dataset, corruption_name, batch_size)
        result = evaluate_graph(
            model,
            graph,
            dl,
            metric,
            intervention=intervention,
            quiet=True,
            skip_clean=True,
        )
        results[corruption_name] = result.mean().item()
        if not quiet:
            print(f"  {corruption_name}: {results[corruption_name]:.4f}")

    return results


def compute_robust_metrics(per_corruption_results: Dict[str, float]) -> Dict:
    """
    Compute summary statistics from per-corruption evaluation.

    Returns dict with: mean, worst, best, gap, per_corruption.
    """
    values = list(per_corruption_results.values())
    return {
        "mean": sum(values) / len(values),
        "worst": max(values),  # largest loss = worst faithfulness
        "best": min(values),
        "gap": max(values) - min(values),
        "per_corruption": per_corruption_results,
    }


def evaluate_baseline_robust(
    model: HookedTransformer,
    dataset: MultiCorruptDataset,
    metric: Callable,
    batch_size: int = 32,
    quiet: bool = False,
) -> Dict[str, float]:
    """Evaluate the full model (no ablation) under each corruption, as reference."""
    results = {}
    for corruption_name in dataset.corruption_names:
        dl = make_eap_dataloader(dataset, corruption_name, batch_size)
        result = evaluate_baseline(model, dl, metric)
        results[corruption_name] = result.mean().item()
        if not quiet:
            print(f"  baseline {corruption_name}: {results[corruption_name]:.4f}")
    return results


def compare_circuits(
    model: HookedTransformer,
    circuits: Dict[str, Graph],
    dataset: MultiCorruptDataset,
    metric: Callable,
    batch_size: int = 32,
) -> Dict[str, Dict]:
    """
    Compare multiple circuits under robust evaluation.

    Args:
        circuits: Dict mapping circuit_name -> Graph with in_graph mask set.

    Returns:
        Dict mapping circuit_name -> robust metrics dict.
    """
    all_results = {}
    for circuit_name, graph in circuits.items():
        print(f"\nEvaluating circuit: {circuit_name}")
        per_corr = evaluate_robust(
            model, graph, dataset, metric, batch_size=batch_size
        )
        all_results[circuit_name] = compute_robust_metrics(per_corr)
    return all_results
