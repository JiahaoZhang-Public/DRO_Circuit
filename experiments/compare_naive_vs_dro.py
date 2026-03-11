#!/usr/bin/env python
"""
Experiment: Compare naive (single-corruption) vs DRO circuit discovery.

Design:
  - Task: IOI on GPT-2 small
  - Method: EAP (fast, single forward/backward per corruption)
  - N = 50 examples
  - Corruption families: S2_IO, IO_RAND, S_RAND

  Naive circuits: For each corruption family c_k, score edges using ONLY c_k,
                  then select top-n edges. This gives 3 naive circuits,
                  one per corruption.

  DRO circuit:   Score edges under ALL corruptions, aggregate with max over
                 corruptions, then select top-n edges. One circuit.

  Evaluation:    Evaluate ALL circuits under ALL corruptions.
                 Report per-corruption faithfulness and worst-case.

  Expected result: Each naive circuit should be most faithful on its own
                   corruption but degrade on others. The DRO circuit should
                   have a smaller worst-case gap (more uniform faithfulness).

Usage:
    python experiments/compare_naive_vs_dro.py [--n_edges 100] [--device cpu]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add vendor paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "EAP-IG" / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "Automatic-Circuit-Discovery"))
sys.path.insert(0, str(ROOT))

import torch
from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.aggregation.aggregators import MaxAggregator, make_aggregator
from dro_circuit.data.eap_adapter import make_eap_dataloader
from dro_circuit.evaluation.metrics import logit_diff_loss
from dro_circuit.tasks.ioi import IOITask


def score_single_corruption(model, dataset, corruption_name, metric, batch_size):
    """Score edges using a single corruption (naive approach)."""
    graph = Graph.from_model(model)
    dl = make_eap_dataloader(dataset, corruption_name, batch_size)
    attribute(model, graph, dl, metric, method="EAP", intervention="patching", quiet=True)
    return graph.scores.cpu().clone()


def evaluate_circuit(model, graph, dataset, metric, batch_size):
    """Evaluate a circuit under each corruption, return per-corruption results."""
    results = {}
    for cname in dataset.corruption_names:
        dl = make_eap_dataloader(dataset, cname, batch_size)
        result = evaluate_graph(
            model, graph, dl, metric, intervention="patching", quiet=True, skip_clean=True
        )
        results[cname] = result.mean().item()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--n_edges", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/naive_vs_dro")
    parser.add_argument(
        "--aggregators",
        nargs="+",
        default=["max"],
        help="DRO aggregators to test (max, cvar, softmax)",
    )
    args = parser.parse_args()

    corruptions = ["S2_IO", "IO_RAND", "S_RAND"]

    # ── Setup ──────────────────────────────────────────────────────────────
    print(f"=== Naive vs DRO Experiment ===")
    print(f"  n_examples={args.n_examples}, n_edges={args.n_edges}, device={args.device}")
    print(f"  corruptions={corruptions}")
    print()

    task = IOITask(
        n_examples=args.n_examples,
        device=args.device,
        seed=args.seed,
        corruption_families=corruptions,
    )

    print("Loading GPT-2 small...")
    t0 = time.time()
    model = task.load_model()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    print("Building multi-corruption dataset...")
    multi_ds, raw_ds = task.build_dataset(tokenizer=model.tokenizer)
    print(f"  {len(multi_ds)} examples x {multi_ds.n_corruptions} corruptions")

    metric = logit_diff_loss

    # ── Step 1: Score edges per corruption ─────────────────────────────────
    print("\n--- Scoring edges ---")
    per_corruption_scores = {}
    for cname in corruptions:
        print(f"  Scoring with corruption: {cname}...")
        t0 = time.time()
        scores = score_single_corruption(model, multi_ds, cname, metric, args.batch_size)
        per_corruption_scores[cname] = scores
        print(f"    Done in {time.time() - t0:.1f}s")

    # ── Step 2: Build circuits ─────────────────────────────────────────────
    print("\n--- Building circuits ---")
    circuits = {}

    # Naive circuits: one per corruption
    for cname in corruptions:
        graph = Graph.from_model(model)
        graph.scores = per_corruption_scores[cname].to(graph.scores.device)
        graph.apply_topn(args.n_edges, absolute=True)
        circuits[f"naive_{cname}"] = graph
        n_in = graph.in_graph.sum().item()
        print(f"  naive_{cname}: {int(n_in)} edges")

    # DRO circuits: aggregate over corruptions
    all_scores = torch.stack([per_corruption_scores[c] for c in corruptions], dim=0)  # (K, F, B)

    for agg_name in args.aggregators:
        if agg_name == "cvar":
            aggregator = make_aggregator("cvar", alpha=0.5)
        elif agg_name == "softmax":
            aggregator = make_aggregator("softmax", temperature=1.0)
        else:
            aggregator = make_aggregator("max")

        agg_scores = aggregator.aggregate(all_scores)
        graph = Graph.from_model(model)
        graph.scores = agg_scores.to(graph.scores.device)
        graph.apply_topn(args.n_edges, absolute=True)
        circuits[f"dro_{agg_name}"] = graph
        n_in = graph.in_graph.sum().item()
        print(f"  dro_{agg_name}: {int(n_in)} edges")

    # ── Step 3: Evaluate all circuits under all corruptions ────────────────
    print("\n--- Evaluating circuits ---")
    all_results = {}
    for circuit_name, graph in circuits.items():
        print(f"  Evaluating: {circuit_name}")
        per_corr = evaluate_circuit(model, graph, multi_ds, metric, args.batch_size)
        values = list(per_corr.values())
        summary = {
            "per_corruption": per_corr,
            "mean": sum(values) / len(values),
            "worst": max(values),  # highest loss = worst faithfulness
            "best": min(values),
            "gap": max(values) - min(values),
        }
        all_results[circuit_name] = summary

    # Also evaluate baseline (full model, no ablation)
    print("  Evaluating: baseline (full model)")
    baseline_results = {}
    for cname in corruptions:
        dl = make_eap_dataloader(multi_ds, cname, args.batch_size)
        result = evaluate_baseline(model, dl, metric, quiet=True)
        baseline_results[cname] = result.mean().item()
    all_results["baseline"] = {
        "per_corruption": baseline_results,
        "mean": sum(baseline_results.values()) / len(baseline_results),
        "worst": max(baseline_results.values()),
        "best": min(baseline_results.values()),
        "gap": max(baseline_results.values()) - min(baseline_results.values()),
    }

    # ── Step 4: Print results ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Header
    col_w = 14
    header = f"{'Circuit':<22}"
    for c in corruptions:
        header += f"  {c:>{col_w}}"
    header += f"  {'Mean':>{col_w}}  {'Worst':>{col_w}}  {'Gap':>{col_w}}"
    print(header)
    print("-" * len(header))

    # Print each circuit
    for circuit_name in ["baseline"] + [f"naive_{c}" for c in corruptions] + [
        f"dro_{a}" for a in args.aggregators
    ]:
        r = all_results[circuit_name]
        row = f"{circuit_name:<22}"
        for c in corruptions:
            row += f"  {r['per_corruption'][c]:>{col_w}.4f}"
        row += f"  {r['mean']:>{col_w}.4f}"
        row += f"  {r['worst']:>{col_w}.4f}"
        row += f"  {r['gap']:>{col_w}.4f}"
        print(row)

    print()
    print("Note: Values are logit_diff LOSS (negated logit diff). Lower = more faithful.")
    print("      'Worst' = max loss across corruptions (the one DRO optimizes).")
    print("      'Gap' = worst - best (sensitivity to corruption choice).")

    # ── Highlight DRO advantage ────────────────────────────────────────────
    print()
    naive_worsts = [all_results[f"naive_{c}"]["worst"] for c in corruptions]
    best_naive_worst = min(naive_worsts)
    best_naive_name = [f"naive_{c}" for c in corruptions][naive_worsts.index(best_naive_worst)]

    for agg_name in args.aggregators:
        dro_worst = all_results[f"dro_{agg_name}"]["worst"]
        dro_gap = all_results[f"dro_{agg_name}"]["gap"]
        best_naive_gap = all_results[best_naive_name]["gap"]

        if dro_worst <= best_naive_worst:
            print(
                f"DRO ({agg_name}) worst-case: {dro_worst:.4f} vs "
                f"best naive worst-case: {best_naive_worst:.4f} ({best_naive_name})"
                f" -- DRO is better by {best_naive_worst - dro_worst:.4f}"
            )
        else:
            print(
                f"DRO ({agg_name}) worst-case: {dro_worst:.4f} vs "
                f"best naive worst-case: {best_naive_worst:.4f} ({best_naive_name})"
                f" -- DRO is worse by {dro_worst - best_naive_worst:.4f}"
            )

        print(
            f"DRO ({agg_name}) gap: {dro_gap:.4f} vs "
            f"best naive gap: {best_naive_gap:.4f}"
        )

    # ── Save ───────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
