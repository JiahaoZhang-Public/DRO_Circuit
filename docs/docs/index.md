# DRO Circuit Discovery

**Distributionally Robust Circuit Discovery in Transformers**

## Overview

Circuit discovery identifies sparse subgraphs (circuits) within transformers that are responsible for specific tasks. Standard methods (e.g., EAP-IG, ACDC) score edges under a **single corruption type**, making the discovered circuit sensitive to that particular choice.

**DRO Circuit Discovery** addresses this by:

1. Scoring edges independently under **multiple corruption families**
2. Aggregating scores with a **DRO (Distributionally Robust Optimization)** rule
3. Selecting edges that are important under the **worst-case** corruption

This produces circuits that are robust across different corruption types.

## Pipeline

```
Score (per corruption)  →  Aggregate (DRO rule)  →  Select (top-n edges)
```

- **Score**: For each of $K$ corruption families, run EAP-IG with all $N$ samples → $K$ score tensors of shape $(n_\text{forward}, n_\text{backward})$
- **Aggregate**: Combine $K$ score tensors into one using Max, CVaR($\alpha$), or Softmax($\tau$)
- **Select**: Take top-$n$ edges by aggregated score → sparse circuit

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Corruption family** | A systematic way to create counterfactual inputs (e.g., swap subject and indirect object names in IOI) |
| **EAP-IG** | Edge Attribution Patching with Integrated Gradients — single fwd/bwd pass to score all edges |
| **DRO aggregation** | Combining per-corruption scores to optimize for worst-case performance |
| **Faithfulness** | How well a circuit reproduces the full model's behavior (measured as % of logit difference preserved) |
| **Activation patching** | Evaluation method: keep clean activations for circuit edges, replace with corrupted for non-circuit edges |

## Documentation

- **[Getting Started](getting-started.md)** — Installation, first run, experiment workflow
- **[Architecture](architecture.md)** — Pipeline design, module responsibilities, data flow, IOI corruption families, extending to new tasks
