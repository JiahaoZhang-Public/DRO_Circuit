#!/usr/bin/env python
"""
Visualize discovered circuits as Graphviz diagrams using EAP's Graph.to_image().

Loads circuit masks from comprehensive experiment outputs and renders them as
publication-quality circuit diagrams showing nodes (attention heads, MLPs) and
edges (Q/K/V connections) with score-proportional edge widths.

Edge colors: Purple=Q, Green=K, Blue=V, Black=positive non-QKV, Red=negative.

Usage:
    python experiments/visualize_circuits.py \
        --input_dir outputs/comprehensive \
        --output_dir outputs/comprehensive/figures \
        --budget 50
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add vendor paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "EAP-IG" / "src"))

from eap.graph import Graph


GPT2_SMALL_CFG = {
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "parallel_attn_mlp": False,
}


def load_circuit_as_graph(mask_path: str, cfg: dict = GPT2_SMALL_CFG) -> Graph:
    """Load a circuit mask .pt file and reconstruct an EAP Graph object."""
    data = torch.load(mask_path, weights_only=False)
    graph = Graph.from_model(cfg)
    graph.scores[:] = data["scores"]
    graph.in_graph[:] = data["in_graph"].bool()

    from einops import einsum
    nodes_with_outgoing = graph.in_graph.any(dim=1)
    nodes_with_ingoing = (
        einsum(
            graph.in_graph.any(dim=0).float(),
            graph.forward_to_backward.float(),
            "backward, forward backward -> forward",
        )
        > 0
    )
    nodes_with_ingoing[0] = True
    graph.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing
    return graph


def render_circuit(
    graph: Graph,
    output_path: str,
    title: str = "",
    min_penwidth: float = 0.3,
    max_penwidth: float = 4.0,
    seed: int = 42,
):
    """Render circuit as Graphviz diagram with Q/K/V colored edges."""
    import pygraphviz as pgv
    from eap.visualization import get_color

    g = pgv.AGraph(
        directed=True,
        strict=False,  # Allow parallel edges (Q,K,V to same head)
        bgcolor="white",
        overlap="false",
        splines="true",
        rankdir="TB",
        label=title,
        labelloc="t",
        fontsize="16",
        fontname="Helvetica-Bold",
        nodesep="0.3",
        ranksep="0.4",
    )

    np.random.seed(seed)

    # Collect active nodes by layer
    layer_nodes = {}
    for node_name, node in graph.nodes.items():
        if not node.in_graph:
            continue
        if node_name == "input":
            layer = -1
        elif node_name == "logits":
            layer = graph.cfg["n_layers"]
        else:
            layer = node.layer
        layer_nodes.setdefault(layer, []).append(node_name)

    if not layer_nodes:
        return

    # Add nodes grouped by layer
    for layer in sorted(layer_nodes.keys()):
        sub = g.add_subgraph(sorted(layer_nodes[layer]), name=f"cluster_{layer}")
        sub.graph_attr["rank"] = "same"
        sub.graph_attr["style"] = "invis"

        for node_name in sorted(layer_nodes[layer]):
            if node_name == "input":
                attrs = dict(fillcolor="#C8E6C9", shape="ellipse", fontsize="11")
            elif node_name == "logits":
                attrs = dict(fillcolor="#FFE0B2", shape="doubleoctagon", fontsize="11")
            elif node_name.startswith("m"):
                attrs = dict(fillcolor="#BBDEFB", shape="box", fontsize="9")
            else:
                attrs = dict(fillcolor="#F8BBD0", shape="box", fontsize="9")

            g.add_node(
                node_name,
                style="filled,rounded",
                color="black",
                fontname="Helvetica",
                **attrs,
            )

    # Compute edge normalization from active edges
    active_scores = []
    for edge in graph.edges.values():
        if edge.in_graph:
            s = edge.score.item() if isinstance(edge.score, torch.Tensor) else edge.score
            active_scores.append(abs(s))

    if not active_scores:
        g.draw(output_path, prog="dot")
        return

    max_s = max(active_scores)
    min_s = min(active_scores)

    # Add edges with Q/K/V colors and score-proportional width
    for edge in graph.edges.values():
        if not edge.in_graph:
            continue

        score_val = edge.score.item() if isinstance(edge.score, torch.Tensor) else edge.score
        abs_score = abs(score_val)

        if max_s != min_s:
            normalized = (abs_score - min_s) / (max_s - min_s)
        else:
            normalized = 0.5
        penwidth = max(min_penwidth, normalized * max_penwidth)

        color = get_color(edge.qkv, score_val)

        # Use edge name as key to allow parallel edges
        g.add_edge(
            edge.parent.name,
            edge.child.name,
            key=edge.name,
            penwidth=str(penwidth),
            color=color,
            arrowsize="0.5",
        )

    g.draw(output_path, prog="dot")


def main():
    parser = argparse.ArgumentParser(description="Visualize circuits as Graphviz diagrams")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 200])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = input_dir / "circuit_masks"

    circuits_to_render = [
        ("dro_max", "DRO-max"),
        ("dro_cvar_0.50", "DRO-CVaR(0.5)"),
        ("naive_IO_RAND", "Naive (IO_RAND)"),
        ("naive_S_RAND", "Naive (S_RAND)"),
    ]

    for budget in args.budgets:
        print(f"\n=== Budget n={budget} ===")
        for circuit_id, label in circuits_to_render:
            mask_path = masks_dir / f"{circuit_id}_n{budget}.pt"
            if not mask_path.exists():
                print(f"  Skip {circuit_id}: mask not found")
                continue

            graph = load_circuit_as_graph(str(mask_path))
            n_edges = int(graph.in_graph.sum().item())
            n_nodes = int(graph.nodes_in_graph.sum().item())
            title = f"{label}  (n={budget}, {n_edges} edges, {n_nodes} nodes)"

            for fmt in ["png", "pdf"]:
                out_path = output_dir / f"circuit_{circuit_id}_n{budget}.{fmt}"
                render_circuit(graph, str(out_path), title=title)

            print(f"  {circuit_id}: {n_edges} edges, {n_nodes} nodes")

    print(f"\nCircuit diagrams saved to {output_dir}/")


if __name__ == "__main__":
    main()
