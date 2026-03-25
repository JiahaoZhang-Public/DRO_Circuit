"""Robust circuit evaluation under multiple corruption variants."""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_VENDOR_EAP = str(Path(__file__).resolve().parents[2] / "vendor" / "EAP-IG" / "src")
if _VENDOR_EAP not in sys.path:
    sys.path.insert(0, _VENDOR_EAP)

from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.data.eap_adapter import collate_eap, make_eap_dataloader
from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset
from dro_circuit.evaluation.metrics import logit_diff


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


# ---------------------------------------------------------------------------
# Helper: per-example non-negated logit diff metric (no mean reduction)
# ---------------------------------------------------------------------------


def _per_example_logit_diff(
    logits: Tensor, clean_logits: Tensor, input_lengths: Tensor, labels: Tensor
) -> Tensor:
    """Return per-example logit diff (non-negated, non-reduced).

    Positive values indicate the model favours the correct token.
    """
    return logit_diff(logits, clean_logits, input_lengths, labels, loss=False, mean=False)


# ---------------------------------------------------------------------------
# Helper: dataset / dataloader where *both* clean and corrupt slots hold the
# corrupted strings, so ``evaluate_baseline`` runs the full model on corrupt
# inputs and produces per-example logit diff on that corrupt distribution.
# ---------------------------------------------------------------------------


class _CorruptAsCleanDataset(Dataset):
    """Wraps a MultiCorruptDataset so that the 'clean' slot is the corrupt string.

    ``evaluate_baseline`` calls ``metric(logits, corrupted_logits, …)`` where
    ``logits`` comes from the first (clean) element.  By placing the corrupt
    string in *both* slots we ensure the first forward pass uses corrupt inputs,
    giving us the full-model logit diff on the corrupt distribution.
    """

    def __init__(self, multi_dataset: MultiCorruptDataset, corruption_name: str):
        self._multi = multi_dataset
        self._corruption_name = corruption_name

    def __len__(self) -> int:
        return len(self._multi)

    def __getitem__(self, idx: int) -> Tuple[str, str, Tensor]:
        corrupt = self._multi.corrupted_strings[self._corruption_name][idx]
        return (corrupt, corrupt, self._multi.labels[idx])


def _make_corrupt_as_clean_dataloader(
    dataset: MultiCorruptDataset, corruption_name: str, batch_size: int
) -> DataLoader:
    """Create a dataloader where the clean slot contains the corrupt string."""
    ds = _CorruptAsCleanDataset(dataset, corruption_name)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_eap)


# ---------------------------------------------------------------------------
# Normalized faithfulness evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_normalized_faithfulness(
    model: HookedTransformer,
    graph: Graph,
    dataset: MultiCorruptDataset,
    intervention: Literal["patching", "zero", "mean"] = "patching",
    batch_size: int = 32,
    quiet: bool = False,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-example normalized faithfulness for every corruption.

    Normalized faithfulness is defined as::

        Faith(c; x, x̃) = (m̂ − b') / (b − b' + eps)

    where:
        - b  = full-model logit diff on clean input  (per-example)
        - b' = full-model logit diff on corrupt input (per-example)
        - m̂  = circuit-intervened logit diff          (per-example)

    Faith = 1 means the circuit fully recovers clean behaviour;
    Faith = 0 means no better than the corrupt baseline.

    Args:
        model: The ``HookedTransformer`` to evaluate.
        graph: Circuit graph with ``in_graph`` mask already set.
        dataset: ``MultiCorruptDataset`` with K corruption families.
        intervention: Ablation strategy (``'patching'``, ``'zero'``, or
            ``'mean'``).
        batch_size: Batch size for evaluation dataloaders.
        quiet: Suppress per-corruption progress messages.
        eps: Small constant to avoid division by zero.

    Returns:
        Dict mapping ``corruption_name`` to a dict with keys:

        - ``"per_example"``: ``Tensor`` of shape ``(N,)`` with per-example
          faithfulness values.
        - ``"mean"``: ``float``, mean faithfulness across examples.
        - ``"min"``: ``float``, worst single-example faithfulness.
    """
    metric = _per_example_logit_diff
    results: Dict[str, Dict[str, Any]] = {}

    for corruption_name in dataset.corruption_names:
        # --- (a) Circuit metric m̂ : per-example --------------------------
        dl = make_eap_dataloader(dataset, corruption_name, batch_size)
        circuit_metric = evaluate_graph(
            model,
            graph,
            dl,
            metric,
            intervention=intervention,
            quiet=True,
            skip_clean=True,
        )
        # evaluate_graph concatenates per-batch tensors -> shape (N,)

        # --- (b) Clean baseline b : full model on clean inputs ------------
        dl_clean = make_eap_dataloader(dataset, corruption_name, batch_size)
        clean_baseline = evaluate_baseline(model, dl_clean, metric, quiet=True)
        # evaluate_baseline with run_corrupted=False (default) calls
        # metric(clean_logits, corrupt_logits, …) -> per-example tensor

        # --- (c) Corrupt baseline b' : full model on corrupt inputs -------
        dl_corrupt = _make_corrupt_as_clean_dataloader(
            dataset, corruption_name, batch_size
        )
        corrupt_baseline = evaluate_baseline(model, dl_corrupt, metric, quiet=True)
        # The corrupt-as-clean dataloader places the corrupt string in the
        # clean slot, so evaluate_baseline's first forward pass runs on
        # corrupt inputs and metric(...) returns logit diff on those inputs.

        # --- (d) Normalized faithfulness ----------------------------------
        faith = (circuit_metric - corrupt_baseline) / (
            clean_baseline - corrupt_baseline + eps
        )

        results[corruption_name] = {
            "per_example": faith,
            "mean": faith.mean().item(),
            "min": faith.min().item(),
        }

        if not quiet:
            print(
                f"  {corruption_name}: mean_faith={results[corruption_name]['mean']:.4f}, "
                f"min_faith={results[corruption_name]['min']:.4f}"
            )

    return results


def compute_normalized_robust_metrics(
    faith_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute summary robustness metrics from normalized faithfulness.

    Args:
        faith_results: Output of :func:`evaluate_normalized_faithfulness`,
            mapping ``corruption_name`` to a dict containing
            ``"per_example"`` tensors and scalar summaries.

    Returns:
        Dict with keys:

        - ``"mean_faithfulness"``: Average faithfulness over all
          (example, corruption) pairs.  Corresponds to ``1 − R_ERM``.
        - ``"worst_group_faithfulness"``: Minimum over corruptions of per-
          corruption mean faithfulness.  Corresponds to ``1 − R_DRO_group``.
        - ``"per_example_worst_faithfulness"``: For each example *i*, take
          the min over corruptions, then average over examples.
          Corresponds to ``1 − R_DRO_local``.
        - ``"per_corruption"``: Dict of per-corruption mean faithfulness.
    """
    corruption_names = sorted(faith_results.keys())
    n_corruptions = len(corruption_names)

    # Stack per-example tensors: shape (K, N)
    per_example_tensors = torch.stack(
        [faith_results[name]["per_example"] for name in corruption_names], dim=0
    )

    # 1. mean_faithfulness: average over all (i, k) pairs
    mean_faithfulness = per_example_tensors.mean().item()

    # 2. worst_group_faithfulness: min over k of mean_k
    per_corruption_means = per_example_tensors.mean(dim=1)  # (K,)
    worst_group_faithfulness = per_corruption_means.min().item()

    # 3. per_example_worst_faithfulness: for each i, min over k, then mean over i
    per_example_worst = per_example_tensors.min(dim=0).values  # (N,)
    per_example_worst_faithfulness = per_example_worst.mean().item()

    # 4. per_corruption: dict of per-corruption mean faithfulness
    per_corruption = {
        name: faith_results[name]["mean"] for name in corruption_names
    }

    return {
        "mean_faithfulness": mean_faithfulness,
        "worst_group_faithfulness": worst_group_faithfulness,
        "per_example_worst_faithfulness": per_example_worst_faithfulness,
        "per_corruption": per_corruption,
    }
