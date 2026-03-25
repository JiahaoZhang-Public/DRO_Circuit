#!/usr/bin/env python3
"""Evaluate saved circuits using recovery ratio (EAP-IG style).

Recovery = m̂ / b  (circuit logit_diff / full-model logit_diff on clean input)
  - Recovery = 1.0: circuit fully recovers clean behavior
  - Recovery = 0.0: circuit produces zero logit diff

No corrupt baseline needed — avoids the division-by-zero problem of
normalized faithfulness.

Usage:
    python experiments/evaluate_recovery.py \
        --input_dir outputs/exp01_erm_vs_dro \
        --device cuda --batch_size 25
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_EAP = str(_ROOT / "vendor" / "EAP-IG" / "src")
_VENDOR_ACDC = str(_ROOT / "vendor" / "Automatic-Circuit-Discovery")
for p in [_VENDOR_EAP, _VENDOR_ACDC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from functools import partial
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline

from dro_circuit.data.eap_adapter import make_eap_dataloader
from dro_circuit.evaluation.metrics import logit_diff
from dro_circuit.tasks.ioi import IOITask


def load_circuit_into_graph(model, mask_path: Path) -> Graph:
    graph = Graph.from_model(model)
    data = torch.load(mask_path, map_location="cpu", weights_only=True)
    graph.in_graph = data["in_graph"].to(graph.in_graph.device)
    return graph


@torch.no_grad()
def evaluate_recovery(model, graph, dataset, batch_size=25):
    """Compute recovery ratio per corruption.

    Returns dict: corruption_name -> {
        "recovery": float (mean m̂ / mean b),
        "circuit_mean": float,
        "baseline_mean": float,
    }
    """
    metric_fn = partial(logit_diff, loss=False, mean=False)
    results = {}

    for corruption_name in dataset.corruption_names:
        dl = make_eap_dataloader(dataset, corruption_name, batch_size)

        # b = full model logit diff on clean input
        b = evaluate_baseline(model, dl, metric_fn, quiet=True)  # (N,)
        b_mean = b.mean().item()

        # m̂ = circuit logit diff (clean input, corrupt patching)
        m_hat = evaluate_graph(
            model, graph, dl, metric_fn,
            intervention="patching", quiet=True, skip_clean=True,
        )  # (N,)
        m_hat_mean = m_hat.mean().item()

        recovery = m_hat_mean / b_mean if abs(b_mean) > 1e-8 else float("nan")

        results[corruption_name] = {
            "recovery": recovery,
            "circuit_mean": m_hat_mean,
            "baseline_mean": b_mean,
            "per_example_recovery": (m_hat / (b + 1e-8)).tolist(),
        }

    return results


def compute_summary(recovery_results):
    """Compute summary metrics from per-corruption recovery."""
    recoveries = {k: v["recovery"] for k, v in recovery_results.items()}
    values = list(recoveries.values())

    return {
        "mean_recovery": sum(values) / len(values),
        "worst_group_recovery": min(values),
        "best_group_recovery": max(values),
        "per_corruption": recoveries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--circuits", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    masks_dir = input_dir / "circuit_masks"
    output_path = Path(args.output) if args.output else input_dir / "recovery.json"

    with open(input_dir / "metadata.json") as f:
        metadata = json.load(f)

    n_examples = metadata["n_examples"]
    seed = metadata["seed"]

    print(f"=== Recovery Ratio Evaluation ===")
    print(f"  input_dir: {input_dir}")
    print(f"  n_examples: {n_examples}, seed: {seed}")

    task = IOITask(n_examples=n_examples, seed=seed, device=args.device)
    model = task.load_model()
    multi_ds, _ = task.build_dataset(model.tokenizer)

    all_masks = sorted(masks_dir.glob("*.pt"))
    if not all_masks:
        print(f"No circuit masks found in {masks_dir}")
        return

    budgets = args.budgets
    if budgets is None:
        budgets = sorted({int(p.stem.split("_n")[-1]) for p in all_masks})

    circuit_prefixes = args.circuits
    all_results = {}
    t0 = time.time()

    for mask_path in all_masks:
        name = mask_path.stem
        budget = int(name.split("_n")[-1])
        if budget not in budgets:
            continue
        prefix = name.rsplit(f"_n{budget}", 1)[0]
        if circuit_prefixes and prefix not in circuit_prefixes:
            continue

        print(f"\n  Evaluating: {name}")
        graph = load_circuit_into_graph(model, mask_path)
        n_edges = int(graph.in_graph.sum().item())
        print(f"    edges: {n_edges}")

        rec = evaluate_recovery(model, graph, multi_ds, args.batch_size)
        summary = compute_summary(rec)

        # Store without per_example_recovery (too large for JSON summary)
        stored = {}
        for k, v in rec.items():
            stored[k] = {kk: vv for kk, vv in v.items() if kk != "per_example_recovery"}

        all_results[name] = {
            "edges": n_edges,
            "mean_recovery": summary["mean_recovery"],
            "worst_group_recovery": summary["worst_group_recovery"],
            "per_corruption": summary["per_corruption"],
        }

        print(f"    Mean Recovery:       {summary['mean_recovery']:.4f}")
        print(f"    Worst-Group Recov:   {summary['worst_group_recovery']:.4f}")
        for c, v in sorted(summary["per_corruption"].items()):
            base = rec[c]["baseline_mean"]
            circ = rec[c]["circuit_mean"]
            print(f"      {c}: recovery={v:.4f}  (circuit={circ:.3f}, baseline={base:.3f})")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s. Evaluated {len(all_results)} circuits.")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Summary table
    print("\n" + "=" * 110)
    print("RECOVERY RATIO SUMMARY (1.0 = full recovery, 0.0 = no logit diff)")
    print("=" * 110)

    # Get corruption names from first result
    first = next(iter(all_results.values()))
    corruptions = sorted(first["per_corruption"].keys())
    corr_short = [c[:8] for c in corruptions]

    header = f"{'Circuit':<35} {'Edges':>5} {'Mean':>7} {'Worst':>7}"
    for cs in corr_short:
        header += f" {cs:>8}"
    print(header)
    print("-" * len(header))

    for name in sorted(all_results.keys()):
        r = all_results[name]
        line = f"{name:<35} {r['edges']:>5} {r['mean_recovery']:>7.4f} {r['worst_group_recovery']:>7.4f}"
        for c in corruptions:
            line += f" {r['per_corruption'][c]:>8.4f}"
        print(line)


if __name__ == "__main__":
    main()
