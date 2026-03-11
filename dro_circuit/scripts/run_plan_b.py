#!/usr/bin/env python
"""
Run DRO circuit discovery with Plan B (Learnable Mask + Adversary).

Usage:
    python -m dro_circuit.scripts.run_plan_b --task ioi --n_edges 200
"""

import argparse
import json
from pathlib import Path

from dro_circuit.config import PlanBConfig
from dro_circuit.evaluation.robust_evaluator import compute_robust_metrics, evaluate_robust
from dro_circuit.selection.plan_b import PlanBPipeline
from dro_circuit.tasks.ioi import IOITask


def main():
    parser = argparse.ArgumentParser(description="DRO Circuit Discovery - Plan B")
    parser.add_argument("--task", default="ioi", choices=["ioi"])
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--n_edges", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_outer_steps", type=int, default=200)
    parser.add_argument("--reg_type", default="L1", choices=["L0", "L1"])
    parser.add_argument("--reg_lambda", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--adversary_temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs_plan_b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="Corruption families to use",
    )
    args = parser.parse_args()

    # Build task
    corruption_families = args.corruptions or ["S2_IO", "IO_RAND", "S_RAND"]
    if args.task == "ioi":
        task = IOITask(
            n_examples=args.n_examples,
            device=args.device,
            seed=args.seed,
            corruption_families=corruption_families,
        )
    else:
        raise NotImplementedError(f"Task {args.task}")

    # Load model
    print(f"Loading model...")
    model = task.load_model()

    # Build dataset
    print(f"Building dataset with {args.n_examples} examples...")
    multi_ds, raw_ds = task.build_dataset(tokenizer=model.tokenizer)
    print(f"  Corruptions: {multi_ds.corruption_names}")

    # Build Plan B config
    plan_b_config = PlanBConfig(
        lr=args.lr,
        n_outer_steps=args.n_outer_steps,
        reg_type=args.reg_type,
        reg_lambda=args.reg_lambda,
        temperature=args.temperature,
        adversary_temperature=args.adversary_temperature,
    )

    # Run Plan B
    print(f"\nRunning Plan B: {args.n_edges} edges, {args.n_outer_steps} steps")
    pipeline = PlanBPipeline(model, plan_b_config, n_edges=args.n_edges)
    circuit_graph = pipeline.run(multi_ds, task.get_scoring_metric(), batch_size=args.batch_size)

    # Evaluate
    print("\nRobust evaluation:")
    eval_results = evaluate_robust(
        model,
        circuit_graph,
        multi_ds,
        task.get_eval_metric(),
        batch_size=args.batch_size,
    )
    summary = compute_robust_metrics(eval_results)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_graph.to_pt(str(output_dir / "circuit.pt"))
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

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
