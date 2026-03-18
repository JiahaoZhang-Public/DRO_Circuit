#!/usr/bin/env python
"""
Comprehensive experiment: Naive vs DRO circuit discovery across all axes.

Grid:
  - 200 examples, 6 corruptions, 8 edge budgets, 10 aggregators
  - 128 circuits total, 774 evaluations

Three phases with checkpoint/resume:
  Phase 1: Score edges per corruption (save scores.pt)
  Phase 2: Build circuits for all (aggregator, budget) pairs (save masks)
  Phase 3: Evaluate all circuits under all corruptions (save raw_results.json)

Usage:
    python experiments/comprehensive_experiment.py \
        --n_examples 200 --device cuda --seed 42 --batch_size 25 \
        --output_dir outputs/comprehensive \
        --edge_budgets 25 50 100 200 400 800 1600 3200 \
        --resume
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Add vendor paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "EAP-IG" / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "Automatic-Circuit-Discovery"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.aggregation.aggregators import make_aggregator
from dro_circuit.data.eap_adapter import make_eap_dataloader
from dro_circuit.evaluation.metrics import logit_diff_loss
from dro_circuit.evaluation.robust_evaluator import (
    compute_robust_metrics,
    evaluate_baseline_robust,
    evaluate_robust,
)
from dro_circuit.scoring.per_corruption_scorer import PerCorruptionScorer
from dro_circuit.scoring.score_store import ScoreStore
from dro_circuit.tasks.ioi import IOITask

# ── Constants ──────────────────────────────────────────────────────────────

ALL_CORRUPTIONS = ["S2_IO", "IO_RAND", "S_RAND", "S1_RAND", "IO_S1"]
# Note: S_IO excluded due to bug in ACDC gen_flipped_prompts (UnboundLocalError)

DEFAULT_EDGE_BUDGETS = [25, 50, 100, 200, 400, 800, 1600, 3200]

# Aggregator configs: ordered from worst-case to average-case
AGGREGATOR_CONFIGS = OrderedDict([
    ("max",          ("max",     {})),
    ("cvar_0.17",    ("cvar",    {"alpha": 1 / 6})),
    ("cvar_0.33",    ("cvar",    {"alpha": 1 / 3})),
    ("cvar_0.50",    ("cvar",    {"alpha": 0.5})),
    ("cvar_0.67",    ("cvar",    {"alpha": 2 / 3})),
    ("cvar_1.00",    ("cvar",    {"alpha": 1.0})),
    ("softmax_0.01", ("softmax", {"temperature": 0.01})),
    ("softmax_0.1",  ("softmax", {"temperature": 0.1})),
    ("softmax_1.0",  ("softmax", {"temperature": 1.0})),
    ("softmax_10.0", ("softmax", {"temperature": 10.0})),
])


# ── Helpers ────────────────────────────────────────────────────────────────

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def circuit_name(method, variant, budget):
    """Generate consistent circuit name: e.g. naive_S2_IO_n100, dro_max_n200."""
    return f"{method}_{variant}_n{budget}"


# ── Phase 1: Score ─────────────────────────────────────────────────────────

def phase1_score(model, dataset, metric, batch_size, output_dir, resume=False):
    """Score edges for each corruption using EAP. Returns ScoreStore."""
    scores_path = output_dir / "scores.pt"

    if resume and scores_path.exists():
        print(f"[Phase 1] Loading existing scores from {scores_path}")
        return ScoreStore.load(str(scores_path))

    print("[Phase 1] Scoring edges per corruption...")
    t0 = time.time()

    scorer = PerCorruptionScorer(
        model, method="EAP", batch_size=batch_size, quiet=False,
    )
    store = scorer.score_all_corruptions(dataset, metric)
    store.save(str(scores_path))

    elapsed = time.time() - t0
    print(f"[Phase 1] Done in {elapsed:.1f}s. Scores saved to {scores_path}")
    return store


# ── Phase 2: Build ─────────────────────────────────────────────────────────

def phase2_build(model, score_store, edge_budgets, output_dir):
    """
    Build all circuits for every (budget, method) pair.
    Returns dict of {circuit_name: {"in_graph": tensor, "actual_edges": int, "scores": tensor}}.
    Saves in_graph masks to circuit_masks/ directory.
    """
    print("[Phase 2] Building circuits...")
    t0 = time.time()

    masks_dir = output_dir / "circuit_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    all_scores = score_store.all_scores()  # (K, n_forward, n_backward)
    corruptions = score_store.corruption_names
    circuits_info = {}

    for budget in edge_budgets:
        print(f"  Budget n={budget}:")

        # Naive circuits: one per corruption
        for cname in corruptions:
            name = circuit_name("naive", cname, budget)
            graph = Graph.from_model(model)
            graph.scores = score_store.get_scores(cname).to(graph.scores.device)
            graph.apply_topn(budget, absolute=True)
            actual = int(graph.in_graph.sum().item())

            circuits_info[name] = {
                "in_graph": graph.in_graph.cpu().clone(),
                "scores": graph.scores.cpu().clone(),
                "actual_edges": actual,
            }
            torch.save(circuits_info[name], masks_dir / f"{name}.pt")
            print(f"    {name}: {actual} edges")

        # DRO circuits: one per aggregator
        for agg_name, (agg_type, agg_kwargs) in AGGREGATOR_CONFIGS.items():
            name = circuit_name("dro", agg_name, budget)
            aggregator = make_aggregator(agg_type, **agg_kwargs)
            agg_scores = aggregator.aggregate(all_scores)

            graph = Graph.from_model(model)
            graph.scores = agg_scores.to(graph.scores.device)
            graph.apply_topn(budget, absolute=True)
            actual = int(graph.in_graph.sum().item())

            circuits_info[name] = {
                "in_graph": graph.in_graph.cpu().clone(),
                "scores": graph.scores.cpu().clone(),
                "actual_edges": actual,
            }
            torch.save(circuits_info[name], masks_dir / f"{name}.pt")
            print(f"    {name}: {actual} edges")

    elapsed = time.time() - t0
    print(f"[Phase 2] Built {len(circuits_info)} circuits in {elapsed:.1f}s")
    return circuits_info


# ── Phase 3: Evaluate ──────────────────────────────────────────────────────

def phase3_evaluate(model, circuits_info, dataset, metric, batch_size, output_dir, resume=False):
    """
    Evaluate all circuits under all corruptions.
    Saves incrementally to raw_results.json for resume safety.
    """
    print("[Phase 3] Evaluating circuits...")
    t0 = time.time()

    results_path = output_dir / "raw_results.json"

    # Load existing results if resuming
    if resume and results_path.exists():
        results = load_json(results_path)
        print(f"  Resuming: {len(results)} circuits already evaluated")
    else:
        results = {}

    # Evaluate baseline first
    if "baseline" not in results:
        print("  Evaluating: baseline (full model)")
        baseline = evaluate_baseline_robust(model, dataset, metric, batch_size=batch_size, quiet=True)
        results["baseline"] = baseline
        save_json(results, results_path)

    # Evaluate each circuit
    total = len(circuits_info)
    done = sum(1 for k in circuits_info if k in results)
    for i, (name, info) in enumerate(circuits_info.items()):
        if name in results:
            continue

        # Reconstruct Graph from saved mask
        graph = Graph.from_model(model)
        graph.in_graph = info["in_graph"].to(graph.in_graph.device)
        graph.scores = info["scores"].to(graph.scores.device)

        done += 1
        print(f"  [{done}/{total}] Evaluating: {name}")
        per_corr = evaluate_robust(model, graph, dataset, metric, batch_size=batch_size, quiet=True)
        results[name] = per_corr

        # Save after each circuit (resume safety)
        save_json(results, results_path)

    elapsed = time.time() - t0
    print(f"[Phase 3] Evaluated {total} circuits in {elapsed:.1f}s")
    return results


# ── Post-processing ────────────────────────────────────────────────────────

def compute_summary(raw_results, circuits_info):
    """Compute summary metrics for each circuit."""
    summary = {}
    for name, per_corr in raw_results.items():
        values = list(per_corr.values())
        entry = {
            "worst": max(values),
            "mean": sum(values) / len(values),
            "best": min(values),
            "gap": max(values) - min(values),
            "std": float(np.std(values)),
            "worst_corruption": max(per_corr, key=per_corr.get),
            "per_corruption": per_corr,
        }
        if name in circuits_info:
            entry["actual_edges"] = circuits_info[name]["actual_edges"]
        summary[name] = entry
    return summary


def print_main_table(summary, corruptions, edge_budgets):
    """Print concise main comparison table."""

    print("\n" + "=" * 120)
    print("MAIN RESULTS: DRO-max vs Best-Naive at each edge budget")
    print("=" * 120)

    col_w = 10
    header = f"{'Budget':<8}  {'Method':<22}  {'Edges':<6}"
    for c in corruptions:
        header += f"  {c:>{col_w}}"
    header += f"  {'Worst':>{col_w}}  {'Mean':>{col_w}}  {'Gap':>{col_w}}"
    print(header)
    print("-" * len(header))

    for budget in edge_budgets:
        # Find best naive (lowest worst-case)
        naive_names = [circuit_name("naive", c, budget) for c in corruptions]
        naive_worsts = {n: summary[n]["worst"] for n in naive_names if n in summary}
        if not naive_worsts:
            continue
        best_naive_name = min(naive_worsts, key=naive_worsts.get)

        # DRO max
        dro_name = circuit_name("dro", "max", budget)

        for name, label in [(best_naive_name, f"best-naive"), (dro_name, "dro_max")]:
            if name not in summary:
                continue
            s = summary[name]
            edges = s.get("actual_edges", "?")
            row = f"{budget:<8}  {label:<22}  {edges:<6}"
            for c in corruptions:
                v = s["per_corruption"].get(c, float("nan"))
                row += f"  {v:{col_w}.4f}"
            row += f"  {s['worst']:{col_w}.4f}  {s['mean']:{col_w}.4f}  {s['gap']:{col_w}.4f}"
            print(row)
        print()

    print("Lower = more faithful. Worst = max loss across corruptions.")


def print_aggregator_table(summary, edge_budgets):
    """Print aggregator comparison at each budget."""

    print("\n" + "=" * 100)
    print("AGGREGATOR COMPARISON: Worst-case loss at each budget")
    print("=" * 100)

    agg_names = list(AGGREGATOR_CONFIGS.keys())
    col_w = 12

    header = f"{'Budget':<8}"
    header += f"  {'best-naive':>{col_w}}"
    for agg in agg_names:
        header += f"  {agg:>{col_w}}"
    print(header)
    print("-" * len(header))

    for budget in edge_budgets:
        # Best naive worst
        naive_worsts = []
        for c in ALL_CORRUPTIONS:
            n = circuit_name("naive", c, budget)
            if n in summary:
                naive_worsts.append(summary[n]["worst"])
        best_naive = min(naive_worsts) if naive_worsts else float("nan")

        row = f"{budget:<8}  {best_naive:{col_w}.4f}"
        for agg in agg_names:
            n = circuit_name("dro", agg, budget)
            if n in summary:
                row += f"  {summary[n]['worst']:{col_w}.4f}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive naive vs DRO experiment")
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/comprehensive")
    parser.add_argument(
        "--edge_budgets", nargs="+", type=int,
        default=DEFAULT_EDGE_BUDGETS,
    )
    parser.add_argument("--resume", action="store_true", help="Skip scoring if scores.pt exists, resume eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corruptions = ALL_CORRUPTIONS

    # ── Save metadata ──────────────────────────────────────────────────────
    metadata = {
        "n_examples": args.n_examples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": args.device,
        "model": "gpt2",
        "method": "EAP",
        "corruptions": corruptions,
        "edge_budgets": args.edge_budgets,
        "aggregators": {
            name: {"type": t, **kw} for name, (t, kw) in AGGREGATOR_CONFIGS.items()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(metadata, output_dir / "metadata.json")

    # ── Setup ──────────────────────────────────────────────────────────────
    print(f"=== Comprehensive Naive vs DRO Experiment ===")
    print(f"  n_examples={args.n_examples}, device={args.device}")
    print(f"  corruptions={corruptions}")
    print(f"  edge_budgets={args.edge_budgets}")
    print(f"  aggregators={list(AGGREGATOR_CONFIGS.keys())}")
    print(f"  output_dir={output_dir}")
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

    # ── Run phases ─────────────────────────────────────────────────────────
    total_t0 = time.time()

    score_store = phase1_score(model, multi_ds, metric, args.batch_size, output_dir, resume=args.resume)
    circuits_info = phase2_build(model, score_store, args.edge_budgets, output_dir)
    raw_results = phase3_evaluate(
        model, circuits_info, multi_ds, metric, args.batch_size, output_dir, resume=args.resume
    )

    total_elapsed = time.time() - total_t0
    print(f"\nTotal experiment time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # ── Post-processing ────────────────────────────────────────────────────
    summary = compute_summary(raw_results, circuits_info)
    save_json(summary, output_dir / "summary.json")

    print_main_table(summary, corruptions, args.edge_budgets)
    print_aggregator_table(summary, args.edge_budgets)

    # ── Win/loss summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DRO-max vs Best-Naive: Win/Loss Summary")
    print("=" * 60)

    wins = 0
    for budget in args.edge_budgets:
        dro_name = circuit_name("dro", "max", budget)
        naive_worsts = []
        for c in corruptions:
            n = circuit_name("naive", c, budget)
            if n in summary:
                naive_worsts.append(summary[n]["worst"])
        if not naive_worsts or dro_name not in summary:
            continue
        best_naive_worst = min(naive_worsts)
        dro_worst = summary[dro_name]["worst"]
        win = dro_worst <= best_naive_worst
        if win:
            wins += 1
        tag = "WIN" if win else "LOSE"
        print(f"  n={budget}: DRO={dro_worst:.4f} vs Naive={best_naive_worst:.4f}  [{tag}]"
              f"  (Δ={best_naive_worst - dro_worst:.4f})")

    print(f"\nDRO-max wins {wins}/{len(args.edge_budgets)} edge budgets on worst-case metric.")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
