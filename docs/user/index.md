# DRO Circuit Discovery

**Distributionally Robust Circuit Discovery in Transformers**

## Overview

Circuit discovery identifies sparse subgraphs (circuits) within transformers that are responsible for specific tasks. Standard methods (e.g., EAP, ACDC) score edges under a **single corruption type** and aggregate over examples, producing circuits that are faithful on average but may fail under certain interventions.

**DRO Circuit Discovery** reframes this as an **ERM vs DRO** comparison:

- **ERM (Empirical Risk Minimization)**: average scores across all corruption families — faithful on average
- **DRO (Distributionally Robust Optimization)**: emphasize worst-case corruption families — faithful uniformly

## Pipeline

```
Score (per corruption, per example)  →  Aggregate (ERM or DRO)  →  Select (top-B edges)
```

- **Score**: For each of $K$ corruption families, run EAP on all $N$ examples → per-example scores of shape $(K, N, n_\text{fwd}, n_\text{bwd})$, or corruption-averaged scores of shape $(K, n_\text{fwd}, n_\text{bwd})$
- **Aggregate**: Combine scores into a single $(n_\text{fwd}, n_\text{bwd})$ tensor using one of:
    - **ERM (Mean)**: average over corruptions
    - **Local DRO**: per-example worst-case, then average over examples
    - **Group DRO (Max / CVaR / Softmax)**: worst corruption family on average
- **Select**: Take top-$B$ edges by aggregated score → sparse circuit
- **Evaluate**: Normalized faithfulness under all corruptions, reporting mean, worst-group, and per-example worst-case metrics

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Corruption family** | A systematic way to create counterfactual inputs (e.g., swap subject and indirect object names in IOI) |
| **EAP** | Edge Attribution Patching — single forward/backward pass to score all edges |
| **Normalized faithfulness** | $\operatorname{Faith}(c; x, \tilde{x}) = (\hat{m} - b') / (b - b')$, where 1 = full recovery, 0 = no better than corrupt baseline |
| **ERM aggregation** | Mean of absolute scores across corruption families — average-case |
| **DRO aggregation** | Worst-case (Local DRO, Max, CVaR, Softmax) across corruption families |
| **Activation patching** | Evaluation method: keep clean activations for circuit edges, replace with corrupted for non-circuit edges |

## Documentation

- **[Getting Started](getting-started.md)** — Installation, first run, experiment workflow
- **[Architecture](architecture.md)** — Pipeline design, module responsibilities, data flow, IOI corruption families, extending to new tasks

For formal definitions and notation, see the research references:

- **[Problem Setup](../research/references/problem-setup.md)** — Grouped intervention dataset, faithfulness loss, ERM/DRO objectives
- **[Experiment Setup](../research/references/experiment-setup.md)** — EAP scoring, aggregation formulas, evaluation protocol
