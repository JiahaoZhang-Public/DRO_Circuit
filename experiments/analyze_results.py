#!/usr/bin/env python
"""
Analyze and visualize results from comprehensive_experiment.py.

All metrics are expressed as **Faithfulness %** relative to the full model baseline:
    faithfulness = (logit_diff_circuit / logit_diff_baseline) × 100
    100% = perfect match to full model, 0% = zero logit difference, <0% = reversed

Generates:
  Performance plots (1-5): worst vs budget, aggregator spectrum, heatmap, gap, pareto
  Circuit visualization (6-9): edge heatmap, overlap, composition, layer density
  Tables: main comparison, full results, aggregator comparison, top edges

Usage:
    python experiments/analyze_results.py \
        --input_dir outputs/comprehensive \
        --output_dir outputs/comprehensive/figures
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add vendor paths for Graph class
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "EAP-IG" / "src"))


# ── Helpers ────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def loss_to_faithfulness(loss_val, baseline_loss):
    """Convert logit_diff_loss to faithfulness %.

    logit_diff_loss = -(correct - incorrect), so lower = better.
    faithfulness = logit_diff_circuit / logit_diff_baseline × 100
                 = (-loss_circuit) / (-loss_baseline) × 100
                 = loss_circuit / loss_baseline × 100

    Since baseline_loss < 0, a circuit with loss_circuit = baseline_loss gives 100%.
    """
    if baseline_loss == 0:
        return 0.0
    return (loss_val / baseline_loss) * 100.0


def convert_summary_to_faithfulness(summary):
    """Convert all per-corruption losses in summary to faithfulness %.

    Returns new summary dict with faithfulness values + the baseline loss used.
    """
    baseline = summary.get("baseline", {})
    # Use the mean baseline loss (same for all corruptions since full model)
    baseline_loss = baseline.get("mean", baseline.get("worst", -1.49))

    new_summary = {}
    for name, entry in summary.items():
        new_entry = dict(entry)
        pc = entry.get("per_corruption", {})
        new_pc = {}
        for corr, loss in pc.items():
            new_pc[corr] = loss_to_faithfulness(loss, baseline_loss)
        new_entry["per_corruption"] = new_pc

        # Recompute worst/mean/best/gap in faithfulness space
        # Note: worst faithfulness = min (not max, since we flipped direction)
        faith_vals = list(new_pc.values())
        if faith_vals:
            new_entry["worst_faith"] = min(faith_vals)
            new_entry["best_faith"] = max(faith_vals)
            new_entry["mean_faith"] = np.mean(faith_vals)
            new_entry["gap_faith"] = max(faith_vals) - min(faith_vals)
            new_entry["std_faith"] = np.std(faith_vals)
            # worst_corruption is the one with minimum faithfulness
            new_entry["worst_corruption_faith"] = min(new_pc, key=new_pc.get)

        new_summary[name] = new_entry

    return new_summary, baseline_loss


def parse_circuit_name(name):
    """Parse 'naive_S2_IO_n100' -> (method='naive', variant='S2_IO', budget=100)."""
    parts = name.rsplit("_n", 1)
    if len(parts) != 2:
        return None, None, None
    prefix, budget_str = parts
    budget = int(budget_str)
    if prefix.startswith("naive_"):
        return "naive", prefix[6:], budget
    elif prefix.startswith("dro_"):
        return "dro", prefix[4:], budget
    return None, None, None


def get_forward_node_name(idx, n_heads=12):
    """Map forward index -> human-readable name."""
    if idx == 0:
        return "input"
    idx -= 1
    layer = idx // (n_heads + 1)
    pos = idx % (n_heads + 1)
    if pos < n_heads:
        return f"a{layer}.h{pos}"
    else:
        return f"m{layer}"


def get_backward_node_name(idx, n_heads=12, n_layers=12):
    """Map backward index -> human-readable name."""
    total = n_layers * (3 * n_heads + 1) + 1
    if idx == total - 1 or idx == -1:
        return "logits"
    layer = idx // (3 * n_heads + 1)
    pos = idx % (3 * n_heads + 1)
    if pos < n_heads:
        return f"a{layer}.h{pos}<q>"
    elif pos < 2 * n_heads:
        return f"a{layer}.h{pos - n_heads}<k>"
    elif pos < 3 * n_heads:
        return f"a{layer}.h{pos - 2 * n_heads}<v>"
    else:
        return f"m{layer}"


def get_source_layer(fwd_idx, n_heads=12):
    """Get layer number from forward index. -1 for input, 12 for logits."""
    if fwd_idx == 0:
        return -1  # input
    fwd_idx -= 1
    return fwd_idx // (n_heads + 1)


def get_dest_layer(bwd_idx, n_heads=12, n_layers=12):
    """Get layer number from backward index. 12 for logits."""
    total = n_layers * (3 * n_heads + 1) + 1
    if bwd_idx == total - 1 or bwd_idx == -1:
        return n_layers  # logits
    return bwd_idx // (3 * n_heads + 1)


def get_edge_type(fwd_idx, bwd_idx, n_heads=12, n_layers=12):
    """Classify edge type based on source/dest node types."""
    total_bwd = n_layers * (3 * n_heads + 1) + 1

    # Source type
    if fwd_idx == 0:
        src_type = "Input"
    else:
        pos = (fwd_idx - 1) % (n_heads + 1)
        src_type = "Attn" if pos < n_heads else "MLP"

    # Dest type
    if bwd_idx == total_bwd - 1:
        dst_type = "Logits"
    else:
        pos = bwd_idx % (3 * n_heads + 1)
        if pos < 3 * n_heads:
            qkv_idx = pos // n_heads
            qkv_label = ["Q", "K", "V"][qkv_idx]
            dst_type = f"Attn({qkv_label})"
        else:
            dst_type = "MLP"

    return f"{src_type}→{dst_type}"


# ── Plot 1: Worst-case Faithfulness vs Edge Budget ────────────────────────

def plot_worst_vs_budget(faith_summary, metadata, output_dir):
    """Headline plot: worst-case faithfulness % vs edge budget."""
    budgets = metadata["edge_budgets"]
    corruptions = metadata["corruptions"]

    # Collect series
    series = {}

    # DRO variants
    series["DRO-max"] = [
        faith_summary.get(f"dro_max_n{b}", {}).get("worst_faith", np.nan)
        for b in budgets
    ]
    series["DRO-CVaR(0.5)"] = [
        faith_summary.get(f"dro_cvar_0.50_n{b}", {}).get("worst_faith", np.nan)
        for b in budgets
    ]
    series["DRO-mean"] = [
        faith_summary.get(f"dro_cvar_1.00_n{b}", {}).get("worst_faith", np.nan)
        for b in budgets
    ]

    # Best and worst naive at each budget
    best_naive = []
    worst_naive = []
    for b in budgets:
        naive_worsts = [
            faith_summary.get(f"naive_{c}_n{b}", {}).get("worst_faith", -np.inf)
            for c in corruptions
        ]
        best_naive.append(max(naive_worsts))  # max faithfulness = best
        worst_naive.append(min(naive_worsts))
    series["Best-naive"] = best_naive
    series["Worst-naive"] = worst_naive

    # Baseline = 100% everywhere
    series["Baseline (100%)"] = [100.0] * len(budgets)

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "DRO-max": {"color": "#2196F3", "marker": "o", "linewidth": 2.5, "zorder": 10},
        "DRO-CVaR(0.5)": {"color": "#FF9800", "marker": "s", "linewidth": 2.5, "zorder": 10},
        "DRO-mean": {"color": "#4CAF50", "marker": "^", "linewidth": 1.8, "linestyle": "--"},
        "Best-naive": {"color": "#F44336", "marker": "D", "linewidth": 2},
        "Worst-naive": {"color": "#9C27B0", "marker": "v", "linewidth": 1.5, "linestyle": ":"},
        "Baseline (100%)": {"color": "gray", "linewidth": 1.2, "linestyle": "-.", "marker": ""},
    }

    for name, vals in series.items():
        style = styles.get(name, {})
        ax.plot(budgets, vals, label=name, **style)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Edge Budget (requested)", fontsize=13)
    ax.set_ylabel("Worst-case Faithfulness (%)", fontsize=13)
    ax.set_title("Worst-case Faithfulness vs Circuit Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])

    # Shade regions
    ax.axhspan(ymin=ax.get_ylim()[0], ymax=0, alpha=0.05, color="red", zorder=0)
    ax.axhspan(ymin=0, ymax=100, alpha=0.05, color="green", zorder=0)

    fig.tight_layout()
    fig.savefig(output_dir / "worst_vs_budget.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "worst_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ worst_vs_budget")


# ── Plot 2: Aggregator Spectrum ───────────────────────────────────────────

def plot_aggregator_spectrum(faith_summary, metadata, output_dir, budget=200):
    """Show worst/mean faithfulness % across aggregators at a fixed budget."""
    agg_names = list(metadata["aggregators"].keys())

    worst_vals = [
        faith_summary.get(f"dro_{a}_n{budget}", {}).get("worst_faith", np.nan)
        for a in agg_names
    ]
    mean_vals = [
        faith_summary.get(f"dro_{a}_n{budget}", {}).get("mean_faith", np.nan)
        for a in agg_names
    ]
    gap_vals = [
        faith_summary.get(f"dro_{a}_n{budget}", {}).get("gap_faith", np.nan)
        for a in agg_names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Colors: blue gradient for worst, orange for mean, green for gap
    colors_worst = ["#2196F3" if v >= 0 else "#F44336" for v in worst_vals]
    colors_mean = "#FF9800"
    colors_gap = "#4CAF50"

    # Worst-case faithfulness
    axes[0].bar(range(len(agg_names)), worst_vals, color=colors_worst, alpha=0.8)
    axes[0].axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].axhline(y=100, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_title("Worst-case Faithfulness (%)", fontsize=12, fontweight="bold")

    # Mean faithfulness
    axes[1].bar(range(len(agg_names)), mean_vals, color=colors_mean, alpha=0.8)
    axes[1].axhline(y=100, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].set_title("Mean Faithfulness (%)", fontsize=12, fontweight="bold")

    # Gap
    axes[2].bar(range(len(agg_names)), gap_vals, color=colors_gap, alpha=0.8)
    axes[2].set_title("Gap (best − worst, pp)", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticks(range(len(agg_names)))
        ax.set_xticklabels(agg_names, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Aggregator Spectrum at n={budget} edges", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "aggregator_spectrum.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "aggregator_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ aggregator_spectrum")


# ── Plot 3: Per-corruption Heatmap ────────────────────────────────────────

def plot_corruption_heatmap(faith_summary, metadata, output_dir, budget=200):
    """Heatmap: circuits × corruptions, values in faithfulness %."""
    corruptions = metadata["corruptions"]

    # Select circuits to show
    circuit_labels = []
    data_rows = []

    # Baseline
    if "baseline" in faith_summary:
        circuit_labels.append("baseline (100%)")
        data_rows.append([
            faith_summary["baseline"]["per_corruption"].get(c, 100.0)
            for c in corruptions
        ])

    # Naive circuits
    for c in corruptions:
        name = f"naive_{c}_n{budget}"
        if name in faith_summary:
            circuit_labels.append(f"naive_{c}")
            data_rows.append([
                faith_summary[name]["per_corruption"].get(c2, np.nan)
                for c2 in corruptions
            ])

    # Key DRO circuits
    for agg, label in [("max", "DRO-max"), ("cvar_0.50", "DRO-CVaR(0.5)"), ("cvar_1.00", "DRO-mean")]:
        name = f"dro_{agg}_n{budget}"
        if name in faith_summary:
            circuit_labels.append(label)
            data_rows.append([
                faith_summary[name]["per_corruption"].get(c, np.nan)
                for c in corruptions
            ])

    data = np.array(data_rows)

    fig, ax = plt.subplots(figsize=(10, max(6, len(circuit_labels) * 0.6)))

    # Color scheme: red (<0%) - white (0%) - blue (100%) - dark blue (>100%)
    # Use diverging colormap centered at 50%
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=200)

    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels(corruptions, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(circuit_labels)))
    ax.set_yticklabels(circuit_labels, fontsize=10)

    # Annotate cells
    for i in range(len(circuit_labels)):
        for j in range(len(corruptions)):
            val = data[i, j]
            color = "white" if val < 10 or val > 160 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, label="Faithfulness (%)")
    cbar.ax.axhline(y=100, color="black", linewidth=1.5, linestyle="--")
    cbar.ax.axhline(y=0, color="red", linewidth=1, linestyle="-")

    ax.set_title(f"Per-corruption Faithfulness at n={budget} edges", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "heatmap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ heatmap")


# ── Plot 4: Gap vs Edge Budget ────────────────────────────────────────────

def plot_gap_vs_budget(faith_summary, metadata, output_dir):
    """Gap (best - worst faithfulness %) vs edge budget."""
    budgets = metadata["edge_budgets"]
    corruptions = metadata["corruptions"]

    series = {}
    series["DRO-max"] = [
        faith_summary.get(f"dro_max_n{b}", {}).get("gap_faith", np.nan)
        for b in budgets
    ]
    series["DRO-CVaR(0.5)"] = [
        faith_summary.get(f"dro_cvar_0.50_n{b}", {}).get("gap_faith", np.nan)
        for b in budgets
    ]

    best_naive_gap = []
    for b in budgets:
        naive_worst_faiths = {
            c: faith_summary.get(f"naive_{c}_n{b}", {}).get("worst_faith", -np.inf)
            for c in corruptions
        }
        best_c = max(naive_worst_faiths, key=naive_worst_faiths.get)
        best_naive_gap.append(
            faith_summary.get(f"naive_{best_c}_n{b}", {}).get("gap_faith", np.nan)
        )
    series["Best-naive"] = best_naive_gap

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"DRO-max": "#2196F3", "DRO-CVaR(0.5)": "#FF9800", "Best-naive": "#4CAF50"}
    for name, vals in series.items():
        ax.plot(budgets, vals, marker="o", label=name, linewidth=2, color=colors[name])

    ax.set_xscale("log")
    ax.set_xlabel("Edge Budget", fontsize=13)
    ax.set_ylabel("Faithfulness Gap (best − worst, pp)", fontsize=13)
    ax.set_title("Corruption Sensitivity vs Circuit Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])

    fig.tight_layout()
    fig.savefig(output_dir / "gap_vs_budget.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "gap_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ gap_vs_budget")


# ── Plot 5: Pareto Frontier ──────────────────────────────────────────────

def plot_pareto(faith_summary, metadata, output_dir, budget=200):
    """Pareto plot: mean faithfulness % vs worst faithfulness %."""
    agg_names = list(metadata["aggregators"].keys())
    corruptions = metadata["corruptions"]

    fig, ax = plt.subplots(figsize=(9, 7))

    # DRO aggregators
    dro_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(agg_names)))
    for idx, agg in enumerate(agg_names):
        name = f"dro_{agg}_n{budget}"
        if name not in faith_summary:
            continue
        mean_v = faith_summary[name]["mean_faith"]
        worst_v = faith_summary[name]["worst_faith"]
        ax.scatter(mean_v, worst_v, s=100, color=dro_colors[idx], zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(agg, (mean_v, worst_v), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, fontweight="bold")

    # Naive circuits
    for c in corruptions:
        name = f"naive_{c}_n{budget}"
        if name not in faith_summary:
            continue
        mean_v = faith_summary[name]["mean_faith"]
        worst_v = faith_summary[name]["worst_faith"]
        ax.scatter(mean_v, worst_v, s=80, marker="x", color="#F44336", zorder=5, linewidths=2)
        ax.annotate(f"naive_{c}", (mean_v, worst_v), textcoords="offset points",
                    xytext=(5, -10), fontsize=7, color="#F44336")

    # Reference lines
    ax.axhline(y=0, color="red", linewidth=0.8, alpha=0.4, linestyle="--")
    ax.axhline(y=100, color="gray", linewidth=0.8, alpha=0.4, linestyle="--")
    ax.axvline(x=100, color="gray", linewidth=0.8, alpha=0.4, linestyle="--")

    # Ideal point
    ax.scatter([100], [100], s=120, marker="*", color="gold", zorder=10, edgecolors="black")
    ax.annotate("ideal", (100, 100), textcoords="offset points", xytext=(-15, 8), fontsize=8, color="gray")

    ax.set_xlabel("Mean Faithfulness (%)", fontsize=13)
    ax.set_ylabel("Worst-case Faithfulness (%)", fontsize=13)
    ax.set_title(f"Pareto: Mean vs Worst-case Faithfulness at n={budget}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "pareto_front.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "pareto_front.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ pareto_front")


# ── Plot 6: Circuit Edge Heatmap ─────────────────────────────────────────

def plot_circuit_heatmap(masks_dir, output_dir, budget=200, n_heads=12, n_layers=12):
    """Show in_graph boolean mask for DRO-max vs best-naive."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, name, title in [
        (axes[0], f"dro_max_n{budget}", "DRO-max"),
        (axes[1], f"naive_IO_RAND_n{budget}", "Naive (IO_RAND)"),
    ]:
        mask_path = masks_dir / f"{name}.pt"
        if not mask_path.exists():
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
            ax.set_title(title)
            continue

        data = torch.load(mask_path, weights_only=False)
        in_graph = data["in_graph"].float().numpy()

        im = ax.imshow(in_graph, cmap="Blues", aspect="auto", interpolation="nearest")
        ax.set_xlabel("Backward index (Q/K/V/MLP dest)", fontsize=10)
        ax.set_ylabel("Forward index (src nodes)", fontsize=10)
        ax.set_title(f"{title} ({int(data['in_graph'].sum())} edges)", fontsize=12)

        # Add layer boundaries
        for layer in range(1, n_layers + 1):
            fwd_y = 1 + layer * (n_heads + 1) - 0.5
            bwd_x = layer * (3 * n_heads + 1) - 0.5
            ax.axhline(y=fwd_y, color="red", linewidth=0.3, alpha=0.5)
            ax.axvline(x=bwd_x, color="red", linewidth=0.3, alpha=0.5)

    fig.suptitle(f"Circuit Edge Masks at n={budget}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "circuit_heatmap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "circuit_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ circuit_heatmap")


# ── Plot 7: Edge Overlap (Jaccard) ───────────────────────────────────────

def plot_edge_overlap(masks_dir, metadata, output_dir, budget=200):
    """Jaccard similarity between DRO and naive circuits."""
    corruptions = metadata["corruptions"]
    agg_keys = ["max", "cvar_0.50", "cvar_1.00"]

    row_names = [f"naive_{c}" for c in corruptions]
    col_names = [f"dro_{a}" for a in agg_keys]

    jaccard_matrix = np.zeros((len(row_names), len(col_names)))

    for i, rname in enumerate(row_names):
        rpath = masks_dir / f"{rname}_n{budget}.pt"
        if not rpath.exists():
            jaccard_matrix[i, :] = np.nan
            continue
        r_mask = torch.load(rpath, weights_only=False)["in_graph"].bool()

        for j, cname in enumerate(col_names):
            cpath = masks_dir / f"{cname}_n{budget}.pt"
            if not cpath.exists():
                jaccard_matrix[i, j] = np.nan
                continue
            c_mask = torch.load(cpath, weights_only=False)["in_graph"].bool()

            intersection = (r_mask & c_mask).sum().float()
            union = (r_mask | c_mask).sum().float()
            jaccard_matrix[i, j] = (intersection / union).item() if union > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(jaccard_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=30, ha="right")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)

    for i in range(len(row_names)):
        for j in range(len(col_names)):
            ax.text(j, i, f"{jaccard_matrix[i, j]:.3f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, label="Jaccard Similarity")
    ax.set_title(f"Edge Overlap: Naive vs DRO at n={budget}", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "edge_overlap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "edge_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ edge_overlap")


# ── Plot 8: Circuit Composition ──────────────────────────────────────────

def plot_circuit_composition(masks_dir, metadata, output_dir, budget=200,
                             n_heads=12, n_layers=12):
    """Stacked bar chart of edge types for each circuit."""
    corruptions = metadata["corruptions"]
    circuits_to_show = (
        [f"naive_{c}" for c in corruptions]
        + ["dro_max", "dro_cvar_0.50", "dro_cvar_1.00"]
    )

    edge_types_order = [
        "Input→Attn(Q)", "Input→Attn(K)", "Input→Attn(V)", "Input→MLP",
        "Attn→Attn(Q)", "Attn→Attn(K)", "Attn→Attn(V)", "Attn→MLP",
        "MLP→Attn(Q)", "MLP→Attn(K)", "MLP→Attn(V)", "MLP→MLP",
        "Input→Logits", "Attn→Logits", "MLP→Logits",
    ]

    composition = {}
    for cname in circuits_to_show:
        path = masks_dir / f"{cname}_n{budget}.pt"
        if not path.exists():
            continue
        in_graph = torch.load(path, weights_only=False)["in_graph"]
        counts = defaultdict(int)
        src_idxs, dst_idxs = torch.where(in_graph)
        for s, d in zip(src_idxs.tolist(), dst_idxs.tolist()):
            etype = get_edge_type(s, d, n_heads, n_layers)
            counts[etype] += 1
        composition[cname] = counts

    if not composition:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(composition))
    labels = list(composition.keys())
    bottom = np.zeros(len(labels))

    colors = plt.cm.tab20(np.linspace(0, 1, len(edge_types_order)))
    for i, etype in enumerate(edge_types_order):
        vals = [composition[l].get(etype, 0) for l in labels]
        ax.bar(x, vals, bottom=bottom, label=etype, color=colors[i], width=0.7)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Edges", fontsize=11)
    ax.set_title(f"Circuit Composition at n={budget}", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "circuit_composition.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "circuit_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ circuit_composition")


# ── Plot 9: Layer-level Edge Density ─────────────────────────────────────

def plot_layer_density(masks_dir, output_dir, budget=200, n_heads=12, n_layers=12):
    """Layer-to-layer edge density comparison: DRO-max vs naive."""
    circuits = {
        "DRO-max": f"dro_max_n{budget}",
        "Naive (IO_RAND)": f"naive_IO_RAND_n{budget}",
    }

    fig, axes = plt.subplots(1, len(circuits), figsize=(12, 5))
    if len(circuits) == 1:
        axes = [axes]

    for ax, (title, cname) in zip(axes, circuits.items()):
        path = masks_dir / f"{cname}.pt"
        if not path.exists():
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
            ax.set_title(title)
            continue

        in_graph = torch.load(path, weights_only=False)["in_graph"]

        # Build layer-to-layer density matrix
        density = np.zeros((13, 13))
        src_idxs, dst_idxs = torch.where(in_graph)
        for s, d in zip(src_idxs.tolist(), dst_idxs.tolist()):
            sl = get_source_layer(s, n_heads) + 1
            dl = get_dest_layer(d, n_heads, n_layers)
            density[sl, dl] += 1

        im = ax.imshow(density, cmap="hot_r", aspect="auto", interpolation="nearest")
        ax.set_xlabel("Dest Layer", fontsize=10)
        ax.set_ylabel("Source Layer", fontsize=10)
        y_labels = ["inp"] + [str(i) for i in range(12)]
        x_labels = [str(i) for i in range(12)] + ["logits"]
        ax.set_xticks(range(13))
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_yticks(range(13))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_title(f"{title} ({int(in_graph.sum())} edges)", fontsize=11)
        plt.colorbar(im, ax=ax)

    fig.suptitle(f"Layer-level Edge Density at n={budget}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_density.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "layer_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ layer_density")


# ── Tables ────────────────────────────────────────────────────────────────

def save_tables(faith_summary, metadata, output_dir, baseline_loss):
    """Save CSV tables with faithfulness % values."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    corruptions = metadata["corruptions"]
    budgets = metadata["edge_budgets"]

    # Full results table
    header = ["circuit", "actual_edges", "worst_%", "mean_%", "gap_pp",
              "worst_corruption"] + [f"{c}_%" for c in corruptions]
    rows = []
    for name, s in faith_summary.items():
        row = {
            "circuit": name,
            "actual_edges": s.get("actual_edges", ""),
            "worst_%": f"{s.get('worst_faith', ''):.1f}" if "worst_faith" in s else "",
            "mean_%": f"{s.get('mean_faith', ''):.1f}" if "mean_faith" in s else "",
            "gap_pp": f"{s.get('gap_faith', ''):.1f}" if "gap_faith" in s else "",
            "worst_corruption": s.get("worst_corruption_faith", ""),
        }
        for c in corruptions:
            val = s["per_corruption"].get(c, "")
            row[f"{c}_%"] = f"{val:.1f}" if isinstance(val, (int, float)) else ""
        rows.append(row)

    with open(tables_dir / "full_results.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in header) + "\n")

    # Main comparison: best-naive vs DRO at each budget
    with open(tables_dir / "main_comparison.csv", "w") as f:
        header = ["budget", "method", "actual_edges", "worst_%", "mean_%", "gap_pp"] + [f"{c}_%" for c in corruptions]
        f.write(",".join(header) + "\n")
        for b in budgets:
            # Find best naive (highest worst faithfulness)
            naive_worst_faiths = {
                c: faith_summary.get(f"naive_{c}_n{b}", {}).get("worst_faith", -np.inf)
                for c in corruptions
            }
            best_c = max(naive_worst_faiths, key=naive_worst_faiths.get)
            for method, name in [
                (f"best-naive({best_c})", f"naive_{best_c}_n{b}"),
                ("DRO-max", f"dro_max_n{b}"),
                ("DRO-CVaR(0.5)", f"dro_cvar_0.50_n{b}"),
            ]:
                if name not in faith_summary:
                    continue
                s = faith_summary[name]
                vals = [
                    str(b), method, str(s.get("actual_edges", "")),
                    f"{s.get('worst_faith', 0):.1f}",
                    f"{s.get('mean_faith', 0):.1f}",
                    f"{s.get('gap_faith', 0):.1f}",
                ]
                for c in corruptions:
                    v = s["per_corruption"].get(c, "")
                    vals.append(f"{v:.1f}" if isinstance(v, (int, float)) else "")
                f.write(",".join(vals) + "\n")

    # Aggregator comparison
    agg_names = list(metadata["aggregators"].keys())
    with open(tables_dir / "aggregator_comparison.csv", "w") as f:
        header = ["budget"] + [f"{a}_worst%" for a in agg_names] + [f"{a}_mean%" for a in agg_names]
        f.write(",".join(header) + "\n")
        for b in budgets:
            row = [str(b)]
            for a in agg_names:
                s = faith_summary.get(f"dro_{a}_n{b}", {})
                row.append(f"{s.get('worst_faith', ''):.1f}" if "worst_faith" in s else "")
            for a in agg_names:
                s = faith_summary.get(f"dro_{a}_n{b}", {})
                row.append(f"{s.get('mean_faith', ''):.1f}" if "mean_faith" in s else "")
            f.write(",".join(row) + "\n")

    print("  ✓ tables (full_results, main_comparison, aggregator_comparison)")


# ── Top Edges ────────────────────────────────────────────────────────────

def save_top_edges(masks_dir, output_dir, budget=200, n_heads=12, n_layers=12, top_k=20):
    """Save top-K edges for DRO-max and best naive at a given budget."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    circuits = {
        "dro_max": f"dro_max_n{budget}",
        "naive_IO_RAND": f"naive_IO_RAND_n{budget}",
    }

    rows = []
    for label, cname in circuits.items():
        path = masks_dir / f"{cname}.pt"
        if not path.exists():
            continue
        data = torch.load(path, weights_only=False)
        scores = data["scores"]
        in_graph = data["in_graph"]

        # Get top edges by score magnitude
        masked_scores = scores.abs() * in_graph.float()
        flat = masked_scores.flatten()
        topk_vals, topk_idxs = flat.topk(min(top_k, int(in_graph.sum())))

        for rank, (val, idx) in enumerate(zip(topk_vals, topk_idxs)):
            fwd = idx.item() // scores.shape[1]
            bwd = idx.item() % scores.shape[1]
            src_name = get_forward_node_name(fwd, n_heads)
            dst_name = get_backward_node_name(bwd, n_heads, n_layers)
            raw_score = scores[fwd, bwd].item()
            rows.append({
                "circuit": label,
                "rank": rank + 1,
                "source": src_name,
                "destination": dst_name,
                "score": raw_score,
                "abs_score": abs(raw_score),
            })

    with open(tables_dir / "top_edges_comparison.csv", "w") as f:
        header = ["circuit", "rank", "source", "destination", "score", "abs_score"]
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    print("  ✓ top_edges_comparison")


# ── Summary Report ────────────────────────────────────────────────────────

def print_summary_report(faith_summary, metadata, baseline_loss):
    """Print a concise summary of key findings."""
    budgets = metadata["edge_budgets"]
    corruptions = metadata["corruptions"]

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY (Faithfulness %)")
    print("=" * 70)
    print(f"  Baseline logit diff loss = {baseline_loss:.4f}")
    print(f"  Faithfulness = (circuit_loss / baseline_loss) × 100%")
    print(f"  100% = perfect, 0% = random, <0% = reversed")
    print("=" * 70)

    print(f"\n{'Budget':>8} │ {'Best-naive':>12} │ {'DRO-max':>12} │ {'DRO-CVaR(.5)':>14} │ {'Δ (DRO-best vs naive)':>22}")
    print("─" * 80)
    for b in budgets:
        # Best naive worst faithfulness
        naive_worst_faiths = {
            c: faith_summary.get(f"naive_{c}_n{b}", {}).get("worst_faith", -np.inf)
            for c in corruptions
        }
        best_c = max(naive_worst_faiths, key=naive_worst_faiths.get)
        best_naive_worst = naive_worst_faiths[best_c]

        dro_max_worst = faith_summary.get(f"dro_max_n{b}", {}).get("worst_faith", np.nan)
        dro_cvar_worst = faith_summary.get(f"dro_cvar_0.50_n{b}", {}).get("worst_faith", np.nan)

        # Pick better DRO
        dro_best = max(dro_max_worst, dro_cvar_worst)
        delta = dro_best - best_naive_worst

        print(f"  {b:>6} │ {best_naive_worst:>10.1f}% │ {dro_max_worst:>10.1f}% │ {dro_cvar_worst:>12.1f}% │ {delta:>+20.1f} pp")

    print("─" * 80)
    print()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze DRO experiment results (faithfulness %)")
    parser.add_argument("--input_dir", required=True, help="Path to experiment output dir")
    parser.add_argument("--output_dir", default=None, help="Path for figures (default: input_dir/figures)")
    parser.add_argument("--budget", type=int, default=200, help="Fixed budget for per-budget plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = input_dir / "circuit_masks"

    print(f"Loading results from {input_dir}...")
    metadata = load_json(input_dir / "metadata.json")
    summary = load_json(input_dir / "summary.json")

    # Convert to faithfulness %
    print("Converting to faithfulness %...")
    faith_summary, baseline_loss = convert_summary_to_faithfulness(summary)

    # Print summary report
    print_summary_report(faith_summary, metadata, baseline_loss)

    print(f"Generating figures to {output_dir}...")

    # Performance plots (faithfulness %)
    plot_worst_vs_budget(faith_summary, metadata, output_dir)
    plot_aggregator_spectrum(faith_summary, metadata, output_dir, budget=args.budget)
    plot_corruption_heatmap(faith_summary, metadata, output_dir, budget=args.budget)
    plot_gap_vs_budget(faith_summary, metadata, output_dir)
    plot_pareto(faith_summary, metadata, output_dir, budget=args.budget)

    # Circuit visualization (structure, not metrics — unchanged)
    if masks_dir.exists():
        plot_circuit_heatmap(masks_dir, output_dir, budget=args.budget)
        plot_edge_overlap(masks_dir, metadata, output_dir, budget=args.budget)
        plot_circuit_composition(masks_dir, metadata, output_dir, budget=args.budget)
        plot_layer_density(masks_dir, output_dir, budget=args.budget)
        save_top_edges(masks_dir, output_dir, budget=args.budget)
    else:
        print("  Warning: circuit_masks/ not found, skipping circuit visualization")

    # Tables
    save_tables(faith_summary, metadata, output_dir, baseline_loss)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
