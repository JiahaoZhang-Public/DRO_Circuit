#!/usr/bin/env python
"""
Mixed-corruption baseline experiment.

Compares the standard practice of "mixing corruptions at the sample level"
against DRO aggregation methods.

Standard practice: each clean sample is randomly paired with ONE corruption type.
    score_mixed(e) = (1/N) Σ_i attr(e; x_i, c_i)   where c_i ~ Uniform(corruptions)

This is approximately equal to mean-aggregation over corruptions:
    score_mean(e) = (1/K) Σ_c score_c(e)

We run multiple random seeds of the assignment to measure variance, and compare:
    1. mixed (sample-level mean) — standard practice
    2. cvar_1.00 (corruption-level mean) — our mean aggregator
    3. cvar_0.50 — DRO tail-risk
    4. best-naive (single corruption) — for reference

Usage:
    python experiments/mixed_corruption_experiment.py \
        --n_examples 200 --device cuda --batch_size 25 --n_seeds 5 \
        --output_dir outputs/mixed_corruption
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "EAP-IG" / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "Automatic-Circuit-Discovery"))
sys.path.insert(0, str(ROOT))

from eap.attribute import attribute
from eap.graph import Graph

from dro_circuit.tasks.ioi import IOITask
from dro_circuit.scoring.per_corruption_scorer import PerCorruptionScorer
from dro_circuit.scoring.score_store import ScoreStore
from dro_circuit.aggregation.aggregators import make_aggregator
from dro_circuit.evaluation.metrics import logit_diff_loss
from dro_circuit.evaluation.robust_evaluator import evaluate_robust, compute_robust_metrics
from dro_circuit.data.eap_adapter import make_eap_dataloader, collate_eap


# ── Mixed Corruption DataLoader ───────────────────────────────────────────

class MixedCorruptDataset(Dataset):
    """Each sample is paired with a randomly assigned corruption.

    This simulates the standard practice of constructing a mixed-corruption
    dataset where each (clean, corrupt) pair uses a potentially different
    corruption family.
    """

    def __init__(self, multi_dataset, seed=42):
        self.multi = multi_dataset
        self.corruptions = multi_dataset.corruption_names
        K = len(self.corruptions)
        N = len(multi_dataset)

        # Randomly assign each sample to a corruption
        rng = random.Random(seed)
        self.assignments = [rng.randint(0, K - 1) for _ in range(N)]

    def __len__(self):
        return len(self.multi)

    def __getitem__(self, idx):
        c_idx = self.assignments[idx]
        c_name = self.corruptions[c_idx]
        return (
            self.multi.clean_strings[idx],
            self.multi.corrupted_strings[c_name][idx],
            self.multi.labels[idx],
        )


def make_mixed_dataloader(multi_dataset, batch_size, seed=42):
    ds = MixedCorruptDataset(multi_dataset, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_eap)


# ── Score with mixed corruption ───────────────────────────────────────────

def score_mixed(model, dataset, metric, method="EAP", batch_size=25, seed=42):
    """Run EAP on a mixed-corruption dataloader."""
    graph = Graph.from_model(model)
    dl = make_mixed_dataloader(dataset, batch_size, seed=seed)
    attribute(model, graph, dl, metric, method=method, intervention="patching",
              aggregation="sum", ig_steps=None, quiet=True)
    return graph.scores.cpu().clone()


# ── Build and evaluate circuits ───────────────────────────────────────────

def build_circuit(model, scores, budget):
    """Build a circuit from scores at the given edge budget."""
    graph = Graph.from_model(model)
    graph.scores = scores.to(graph.scores.device)
    graph.apply_topn(budget, absolute=True)
    mask = graph.in_graph.cpu().clone()
    actual = int(mask.sum().item())
    return mask, actual


def evaluate_circuit(model, mask, scores, dataset, metric, batch_size):
    """Evaluate a circuit under all corruptions."""
    graph = Graph.from_model(model)
    graph.scores = scores.to(graph.scores.device)
    graph.in_graph[:] = mask.to(graph.in_graph.device)

    per_corruption = evaluate_robust(model, graph, dataset, metric,
                                     batch_size=batch_size, quiet=True)
    return compute_robust_metrics(per_corruption)


def loss_to_faith(loss, baseline_loss):
    if baseline_loss == 0:
        return 0.0
    return (loss / baseline_loss) * 100.0


# ── Main experiment ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of random seeds for mixed-corruption assignment")
    parser.add_argument("--output_dir", default="outputs/mixed_corruption")
    parser.add_argument("--edge_budgets", type=int, nargs="+",
                        default=[25, 50, 100, 200, 400, 800, 1600, 3200])
    parser.add_argument("--scores_path", default=None,
                        help="Path to pre-computed scores.pt (skip per-corruption scoring)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ──
    print("=" * 60)
    print("  Mixed-corruption Baseline Experiment")
    print("=" * 60)

    task = IOITask(n_examples=args.n_examples, seed=args.seed, device=args.device)
    model = task.load_model()
    dataset, raw_ds = task.build_dataset(tokenizer=model.tokenizer)
    metric = logit_diff_loss
    corruptions = dataset.corruption_names
    K = len(corruptions)
    print(f"Model: gpt2, N={args.n_examples}, K={K} corruptions")
    print(f"Corruptions: {corruptions}")
    print(f"Mixed seeds: {args.n_seeds}")

    # ── Phase 1: Per-corruption scores (reuse if available) ──
    if args.scores_path and Path(args.scores_path).exists():
        print(f"\nLoading pre-computed scores from {args.scores_path}...")
        score_store = ScoreStore.load(args.scores_path)
    else:
        print("\n[Phase 1] Scoring per-corruption...")
        scorer = PerCorruptionScorer(model, method="EAP", batch_size=args.batch_size)
        score_store = scorer.score_all_corruptions(dataset, metric)
        score_store.save(output_dir / "scores.pt")

    all_scores = score_store.all_scores()  # (K, n_forward, n_backward)

    # ── Phase 2: Mixed-corruption scoring ──
    print(f"\n[Phase 2] Scoring mixed-corruption ({args.n_seeds} seeds)...")
    mixed_scores_list = []
    for seed_i in range(args.n_seeds):
        s = args.seed + seed_i * 1000
        print(f"  Mixed seed {seed_i+1}/{args.n_seeds} (seed={s})...")
        scores = score_mixed(model, dataset, metric, method="EAP",
                             batch_size=args.batch_size, seed=s)
        mixed_scores_list.append(scores)

    # ── Phase 3: Build and evaluate ──
    print(f"\n[Phase 3] Building and evaluating circuits...")

    # Compute aggregated scores
    mean_agg = make_aggregator("cvar", alpha=1.0)
    cvar50_agg = make_aggregator("cvar", alpha=0.5)
    max_agg = make_aggregator("max")

    agg_scores_mean = mean_agg.aggregate(all_scores)
    agg_scores_cvar50 = cvar50_agg.aggregate(all_scores)
    agg_scores_max = max_agg.aggregate(all_scores)

    # Evaluate baseline (full model)
    print("  Evaluating baseline (full model)...")
    graph_full = Graph.from_model(model)
    graph_full.reset(empty=False)
    baseline_pc = evaluate_robust(model, graph_full, dataset, metric,
                                  batch_size=args.batch_size, quiet=True)
    baseline_loss = float(np.mean(list(baseline_pc.values())))
    print(f"  Baseline logit_diff_loss = {baseline_loss:.4f}")

    # Results table
    results = {
        "baseline_loss": baseline_loss,
        "corruptions": corruptions,
        "edge_budgets": args.edge_budgets,
        "n_seeds": args.n_seeds,
        "per_budget": {},
    }

    for budget in args.edge_budgets:
        print(f"\n  Budget n={budget}:")
        budget_results = {}

        # --- Best naive (single corruption) ---
        best_naive_worst_faith = -float("inf")
        best_naive_name = None
        for c in corruptions:
            c_scores = score_store.get_scores(c)
            mask, actual = build_circuit(model, c_scores, budget)
            res = evaluate_circuit(model, mask, c_scores, dataset, metric,
                                   args.batch_size)
            worst_faith = loss_to_faith(res["worst"], baseline_loss)
            if worst_faith > best_naive_worst_faith:
                best_naive_worst_faith = worst_faith
                best_naive_name = c
                best_naive_res = res

        budget_results["best_naive"] = {
            "corruption": best_naive_name,
            "worst_faith": best_naive_worst_faith,
            "mean_faith": loss_to_faith(best_naive_res["mean"], baseline_loss),
            "per_corruption": {c: loss_to_faith(v, baseline_loss) for c, v in best_naive_res["per_corruption"].items()},
        }
        print(f"    best-naive({best_naive_name}): worst={best_naive_worst_faith:.1f}%")

        # --- Mixed corruption (multiple seeds) ---
        mixed_worst_faiths = []
        mixed_mean_faiths = []
        for seed_i, mixed_scores in enumerate(mixed_scores_list):
            mask, actual = build_circuit(model, mixed_scores, budget)
            res = evaluate_circuit(model, mask, mixed_scores, dataset, metric,
                                   args.batch_size)
            worst_faith = loss_to_faith(res["worst"], baseline_loss)
            mean_faith = loss_to_faith(res["mean"], baseline_loss)
            mixed_worst_faiths.append(worst_faith)
            mixed_mean_faiths.append(mean_faith)

        budget_results["mixed"] = {
            "worst_faith_mean": float(np.mean(mixed_worst_faiths)),
            "worst_faith_std": float(np.std(mixed_worst_faiths)),
            "worst_faith_all": mixed_worst_faiths,
            "mean_faith_mean": float(np.mean(mixed_mean_faiths)),
            "mean_faith_std": float(np.std(mixed_mean_faiths)),
        }
        print(f"    mixed (sample-level mean): worst={np.mean(mixed_worst_faiths):.1f}% ± {np.std(mixed_worst_faiths):.1f}%")

        # --- DRO mean (cvar_1.00 = corruption-level mean) ---
        mask, actual = build_circuit(model, agg_scores_mean, budget)
        res = evaluate_circuit(model, mask, agg_scores_mean, dataset, metric,
                               args.batch_size)
        budget_results["dro_mean"] = {
            "worst_faith": loss_to_faith(res["worst"], baseline_loss),
            "mean_faith": loss_to_faith(res["mean"], baseline_loss),
            "per_corruption": {c: loss_to_faith(v, baseline_loss) for c, v in res["per_corruption"].items()},
        }
        print(f"    DRO-mean (cvar_1.00):      worst={budget_results['dro_mean']['worst_faith']:.1f}%")

        # --- DRO CVaR(0.5) ---
        mask, actual = build_circuit(model, agg_scores_cvar50, budget)
        res = evaluate_circuit(model, mask, agg_scores_cvar50, dataset, metric,
                               args.batch_size)
        budget_results["dro_cvar_0.50"] = {
            "worst_faith": loss_to_faith(res["worst"], baseline_loss),
            "mean_faith": loss_to_faith(res["mean"], baseline_loss),
            "per_corruption": {c: loss_to_faith(v, baseline_loss) for c, v in res["per_corruption"].items()},
        }
        print(f"    DRO-CVaR(0.5):             worst={budget_results['dro_cvar_0.50']['worst_faith']:.1f}%")

        # --- DRO max ---
        mask, actual = build_circuit(model, agg_scores_max, budget)
        res = evaluate_circuit(model, mask, agg_scores_max, dataset, metric,
                               args.batch_size)
        budget_results["dro_max"] = {
            "worst_faith": loss_to_faith(res["worst"], baseline_loss),
            "mean_faith": loss_to_faith(res["mean"], baseline_loss),
            "per_corruption": {c: loss_to_faith(v, baseline_loss) for c, v in res["per_corruption"].items()},
        }
        print(f"    DRO-max:                   worst={budget_results['dro_max']['worst_faith']:.1f}%")

        results["per_budget"][str(budget)] = budget_results

    # ── Save results ──
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary table ──
    print("\n" + "=" * 90)
    print(f"  {'Budget':>8} │ {'Best-naive':>12} │ {'Mixed (±std)':>18} │ {'DRO-mean':>10} │ {'DRO-CVaR(.5)':>14} │ {'DRO-max':>10}")
    print("─" * 90)
    for b in args.edge_budgets:
        r = results["per_budget"][str(b)]
        naive_w = r["best_naive"]["worst_faith"]
        mixed_w = r["mixed"]["worst_faith_mean"]
        mixed_s = r["mixed"]["worst_faith_std"]
        mean_w = r["dro_mean"]["worst_faith"]
        cvar_w = r["dro_cvar_0.50"]["worst_faith"]
        max_w = r["dro_max"]["worst_faith"]
        print(f"  {b:>6} │ {naive_w:>10.1f}% │ {mixed_w:>8.1f}% ± {mixed_s:>4.1f}% │ {mean_w:>8.1f}% │ {cvar_w:>12.1f}% │ {max_w:>8.1f}%")
    print("─" * 90)

    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
