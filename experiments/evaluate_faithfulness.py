#!/usr/bin/env python3
"""Evaluate saved circuits using normalized faithfulness.

Reports the three metrics from problem-setup.md:
  - Mean faithfulness (1 - R_ERM)
  - Worst-group faithfulness (1 - R_DRO^group)
  - Per-example worst-case faithfulness (1 - R_DRO^local)

Usage:
    python experiments/evaluate_faithfulness.py \
        --input_dir outputs/exp01_erm_vs_dro \
        --device cuda --batch_size 25 \
        --circuits dro_erm_mean dro_max dro_norm_max dro_norm_cvar_0.50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Vendor paths
_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_EAP = str(_ROOT / "vendor" / "EAP-IG" / "src")
_VENDOR_ACDC = str(_ROOT / "vendor" / "Automatic-Circuit-Discovery")
for p in [_VENDOR_EAP, _VENDOR_ACDC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from eap.graph import Graph

from dro_circuit.evaluation.robust_evaluator import (
    evaluate_normalized_faithfulness,
    compute_normalized_robust_metrics,
)
from dro_circuit.tasks.ioi import IOITask


def load_circuit_into_graph(model, mask_path: Path) -> Graph:
    """Load a saved circuit mask into a fresh Graph."""
    graph = Graph.from_model(model)
    data = torch.load(mask_path, map_location="cpu", weights_only=True)
    graph.in_graph = data["in_graph"].to(graph.in_graph.device)
    return graph


def main():
    parser = argparse.ArgumentParser(description="Evaluate circuits with normalized faithfulness")
    parser.add_argument("--input_dir", required=True, help="Experiment output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--budgets", nargs="+", type=int, default=None,
                        help="Edge budgets to evaluate (default: all found)")
    parser.add_argument("--circuits", nargs="+", default=None,
                        help="Circuit prefixes to evaluate, e.g. dro_erm_mean dro_max dro_norm_max")
    parser.add_argument("--output", default=None, help="Output JSON path (default: input_dir/faithfulness.json)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    masks_dir = input_dir / "circuit_masks"
    output_path = Path(args.output) if args.output else input_dir / "faithfulness.json"

    # Load metadata to get experiment config
    with open(input_dir / "metadata.json") as f:
        metadata = json.load(f)

    n_examples = metadata["n_examples"]
    seed = metadata["seed"]

    print(f"=== Normalized Faithfulness Evaluation ===")
    print(f"  input_dir: {input_dir}")
    print(f"  n_examples: {n_examples}, seed: {seed}")

    # Load model and dataset
    task = IOITask(n_examples=n_examples, seed=seed, device=args.device)
    model = task.load_model()
    multi_ds, _ = task.build_dataset(model.tokenizer)

    # Discover circuits to evaluate
    all_masks = sorted(masks_dir.glob("*.pt"))
    if not all_masks:
        print(f"No circuit masks found in {masks_dir}")
        return

    # Filter by budget
    budgets = args.budgets
    if budgets is None:
        budgets = sorted({int(p.stem.split("_n")[-1]) for p in all_masks})

    # Filter by circuit prefix
    circuit_prefixes = args.circuits

    results = {}
    t0 = time.time()

    for mask_path in all_masks:
        name = mask_path.stem  # e.g. dro_erm_mean_n200

        # Filter by budget
        budget = int(name.split("_n")[-1])
        if budget not in budgets:
            continue

        # Filter by prefix
        prefix = name.rsplit(f"_n{budget}", 1)[0]  # e.g. dro_erm_mean
        if circuit_prefixes and prefix not in circuit_prefixes:
            continue

        print(f"\n  Evaluating: {name}")
        graph = load_circuit_into_graph(model, mask_path)
        n_edges = int(graph.in_graph.sum().item())
        print(f"    edges: {n_edges}")

        faith_results = evaluate_normalized_faithfulness(
            model, graph, multi_ds,
            intervention="patching",
            batch_size=args.batch_size,
        )
        metrics = compute_normalized_robust_metrics(faith_results)

        results[name] = {
            "edges": n_edges,
            "mean_faithfulness": metrics["mean_faithfulness"],
            "worst_group_faithfulness": metrics["worst_group_faithfulness"],
            "per_example_worst_faithfulness": metrics["per_example_worst_faithfulness"],
            "per_corruption": metrics["per_corruption"],
        }

        print(f"    Mean Faith:          {metrics['mean_faithfulness']:.4f}")
        print(f"    Worst-Group Faith:   {metrics['worst_group_faithfulness']:.4f}")
        print(f"    Per-Ex Worst Faith:  {metrics['per_example_worst_faithfulness']:.4f}")
        for c, v in sorted(metrics["per_corruption"].items()):
            print(f"      {c}: {v:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s. Evaluated {len(results)} circuits.")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print("NORMALIZED FAITHFULNESS SUMMARY (Faith=1: perfect, Faith=0: no better than corrupt)")
    print("=" * 120)
    header = f"{'Circuit':<35} {'Edges':>6} {'Mean Faith':>12} {'Worst-Group':>12} {'Per-Ex Worst':>13}"
    print(header)
    print("-" * len(header))

    for name in sorted(results.keys()):
        r = results[name]
        print(f"{name:<35} {r['edges']:>6} {r['mean_faithfulness']:>12.4f} "
              f"{r['worst_group_faithfulness']:>12.4f} {r['per_example_worst_faithfulness']:>13.4f}")


if __name__ == "__main__":
    main()
