#!/usr/bin/env python
"""
Run DRO circuit discovery (Score-Aggregate-Select).

Usage:
    python -m dro_circuit.scripts.run --task ioi --n_edges 200 --aggregator max
    python -m dro_circuit.scripts.run --config configs/ioi.yaml
"""

import argparse
import json
from pathlib import Path

from dro_circuit.config import ExperimentConfig
from dro_circuit.evaluation.robust_evaluator import compute_robust_metrics, evaluate_robust
from dro_circuit.selection.pipeline import DROPipeline
from dro_circuit.tasks.ioi import IOITask


def main():
    parser = argparse.ArgumentParser(description="DRO Circuit Discovery")
    parser.add_argument("--task", default="ioi", choices=["ioi"])
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--n_edges", type=int, default=200)
    parser.add_argument(
        "--aggregator", default="max", choices=["max", "cvar", "softmax"]
    )
    parser.add_argument("--method", default="EAP-IG-inputs")
    parser.add_argument("--selection", default="topn", choices=["topn", "greedy"])
    parser.add_argument("--cvar_alpha", type=float, default=0.5)
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ig_steps", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="Corruption families to use (e.g., S2_IO IO_RAND S_RAND)",
    )
    args = parser.parse_args()

    # Build config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            task=args.task,
            n_examples=args.n_examples,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        config.model.device = args.device
        config.scoring.method = args.method
        config.scoring.batch_size = args.batch_size
        config.scoring.ig_steps = args.ig_steps
        config.dro.aggregator = args.aggregator
        config.dro.cvar_alpha = args.cvar_alpha
        config.dro.softmax_temperature = args.softmax_temp
        config.selection.n_edges = args.n_edges
        config.selection.selection_method = args.selection
        if args.corruptions:
            config.corruption.families = args.corruptions

    # Build task
    if config.task == "ioi":
        task = IOITask(
            n_examples=config.n_examples,
            device=config.model.device,
            seed=config.seed,
            corruption_families=config.corruption.families,
        )
    else:
        raise NotImplementedError(f"Task {config.task}")

    # Load model
    print(f"Loading model: {config.model.name}")
    model = task.load_model()

    # Build dataset
    print(f"Building dataset with {config.n_examples} examples...")
    multi_ds, raw_ds = task.build_dataset(tokenizer=model.tokenizer)
    print(
        f"  Corruptions: {multi_ds.corruption_names} "
        f"({multi_ds.n_corruptions} variants)"
    )

    # Build and run pipeline
    print(f"\nRunning DRO pipeline: {config.dro.aggregator} aggregation, "
          f"{config.selection.n_edges} edges, {config.scoring.method}")
    pipeline = DROPipeline.from_config(model, config)
    circuit_graph, score_store = pipeline.run(multi_ds, task.get_scoring_metric())

    # Evaluate
    print("\nRobust evaluation:")
    eval_results = evaluate_robust(
        model,
        circuit_graph,
        multi_ds,
        task.get_eval_metric(),
        batch_size=config.eval.batch_size,
    )
    summary = compute_robust_metrics(eval_results)

    # Save
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_graph.to_pt(str(output_dir / "circuit.pt"))
    score_store.save(str(output_dir / "scores.pt"))
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        # Save config as dict for reproducibility
        import dataclasses
        json.dump(dataclasses.asdict(config), f, indent=2)

    # Print results
    print(f"\nResults (saved to {output_dir}):")
    print(f"  Mean:  {summary['mean']:.4f}")
    print(f"  Worst: {summary['worst']:.4f}")
    print(f"  Best:  {summary['best']:.4f}")
    print(f"  Gap:   {summary['gap']:.4f}")
    for name, val in summary["per_corruption"].items():
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
