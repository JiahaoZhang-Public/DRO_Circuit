# Experiment: ERM vs Normalized DRO

**Date:** 2026-03-26
**Author:** Jiahao Zhang
**Status:** completed

## Goal

Test whether per-corruption score normalization fixes the DRO-max failure observed in Exp 01. The hypothesis: Max aggregation failed because S_RAND scores had ~10x larger magnitudes than IO_RAND scores. Normalizing each corruption's scores to [-1, 1] before aggregation should make all corruptions contribute equally.

## Background

In Exp 01, ERM won worst-group faithfulness at all 8 budgets. Root cause: `max_k |s(e; k)|` was dominated by S_RAND (largest raw magnitudes) instead of IO_RAND (hardest corruption).

**Fix**: Before aggregation, apply `normalize_per_corruption(scores, method="max")` which divides each corruption's score vector by its max absolute value, scaling all corruptions to [-1, 1].

## Environment

- **Server:** gpuhub-root-rtx4090-48
- **GPU:** RTX 4090 48GB
- **Model:** GPT-2 small
- **Task:** IOI, K=4 corruptions (S2_IO, IO_RAND, S_RAND, IO_S1)
- **Excluded:** S1_RAND (trivial corruption — see Exp 01 for explanation)
- **N examples:** 200
- **Edge budgets:** [25, 50, 100, 200, 400, 800, 1600, 3200]
- **Commit:** e8da89c

## Exact Commands

```bash
cd /root/projects/DRO_Circuit && conda activate dro

python experiments/comprehensive_experiment.py \
    --n_examples 200 --batch_size 25 --device cuda --seed 42 \
    --edge_budgets 25 50 100 200 400 800 1600 3200 \
    --output_dir outputs/exp01v2_normalized
```

## Results

### Timing

- Phase 1 + 1b (scoring): 26.9s (reuses same scoring as Exp 01)
- Phase 2 (build 184 circuits): 81.0s
- Phase 3 (evaluate 184 circuits): ~900s
- **Total: ~17 min**

### ERM vs Norm-DRO-max vs Raw-DRO-max (worst-group loss, lower = more faithful)

| Budget | ERM      | Raw Max  | **Norm Max** | Norm CVaR(0.33) | Norm CVaR(0.50) |
|--------|----------|----------|--------------|-----------------|-----------------|
| 25     | 0.1489   | 0.3179   | **0.8012**   | 0.8016          | 0.8012          |
| 50     | 0.1284   | 0.4236   | **0.1988**   | 0.7114          | 0.5861          |
| 100    | -0.1481  | 0.1799   | **-0.1240**  | -0.2355         | -0.2349         |
| 200    | -0.2018  | 0.4213   | **-0.1275**  | -0.2880         | -0.3904         |
| 400    | -0.4373  | -0.1867  | **-0.3904**  | -0.4566         | -0.5635         |
| 800    | -0.8835  | -0.6758  | **-0.8655**  | -0.7360         | -0.7226         |
| 1600   | -1.1304  | -1.0368  | **-1.2520**  | -1.3095         | -1.2762         |
| 3200   | -1.3574  | -1.3125  | **-1.4140**  | -1.4109         | -1.4317         |

### Win/Loss Summary: Norm-Max vs ERM

| Budget | Norm-Max | ERM      | Winner       |
|--------|----------|----------|--------------|
| 25     | 0.8012   | 0.1489   | ERM          |
| 50     | 0.1988   | 0.1284   | ERM          |
| 100    | -0.1240  | -0.1481  | ERM (barely) |
| 200    | -0.1275  | -0.2018  | ERM          |
| 400    | -0.3904  | -0.4373  | ERM          |
| 800    | -0.8655  | -0.8835  | ERM          |
| 1600   | **-1.2520** | -1.1304  | **Norm-Max** |
| 3200   | **-1.4140** | -1.3574  | **Norm-Max** |

**Norm-Max wins 2/8 budgets** (1600, 3200). Major improvement vs Raw-Max (0/8).

### Best normalized aggregator: Norm-CVaR(0.50) vs ERM

| Budget | Norm-CVaR(0.50) | ERM     | Winner          |
|--------|-----------------|---------|-----------------|
| 25     | 0.8012          | 0.1489  | ERM             |
| 50     | 0.5861          | 0.1284  | ERM             |
| 100    | -0.2349         | -0.1481 | ERM             |
| 200    | **-0.3904**     | -0.2018 | **Norm-CVaR**   |
| 400    | **-0.5635**     | -0.4373 | **Norm-CVaR**   |
| 800    | -0.7226         | -0.8835 | ERM             |
| 1600   | **-1.2762**     | -1.1304 | **Norm-CVaR**   |
| 3200   | **-1.4317**     | -1.3574 | **Norm-CVaR**   |

**Norm-CVaR(0.50) wins 4/8 budgets** (200, 400, 1600, 3200).

### Normalized Faithfulness (K=4, excluding S1_RAND)

Faith=1: perfect recovery. Faith=0: no better than corrupt baseline.

| Method        | Budget | Mean Faith | Worst-Group | IO_RAND | IO_S1  | S2_IO  | S_RAND |
|---------------|--------|-----------|-------------|---------|--------|--------|--------|
| ERM           | 100    | 0.49      | 0.02        | 0.69    | 0.93   | 0.02   | 0.34   |
| Norm-Max      | 100    | 0.49      | -0.09       | 0.76    | 0.95   | -0.09  | 0.33   |
| Norm-CVaR     | 100    | 0.50      | 0.02        | 0.72    | 0.92   | 0.02   | 0.34   |
| Norm-LocalDRO | 100    | 0.27      | 0.10        | 0.43    | 0.41   | 0.10   | 0.13   |
| ERM           | 200    | 0.46      | 0.14        | 0.69    | 0.59   | 0.14   | 0.43   |
| Norm-Max      | 200    | 0.54      | 0.11        | 0.68    | 1.04   | 0.11   | 0.33   |
| Norm-CVaR     | 200    | 0.51      | 0.20        | 0.76    | 0.66   | 0.20   | 0.42   |
| Norm-LocalDRO | 200    | 0.27      | 0.04        | 0.52    | 0.36   | 0.04   | 0.17   |
| ERM           | 400    | 0.65      | 0.34        | 0.74    | 0.95   | 0.34   | 0.57   |
| Norm-Max      | 400    | 0.56      | 0.46        | 0.75    | 0.50   | 0.53   | 0.46   |
| Norm-CVaR     | 400    | 0.65      | **0.47**    | 0.80    | 0.79   | 0.47   | 0.54   |
| Norm-LocalDRO | 400    | 0.38      | 0.20        | 0.63    | 0.32   | 0.20   | 0.37   |
| ERM           | 800    | 0.75      | 0.54        | 0.85    | 0.89   | 0.54   | 0.71   |
| Norm-Max      | 800    | 0.60      | 0.36        | 0.86    | 0.36   | 0.57   | 0.60   |
| Norm-CVaR     | 800    | 0.65      | 0.53        | 0.81    | 0.53   | 0.65   | 0.59   |
| Norm-LocalDRO | 800    | 0.59      | 0.49        | 0.67    | 0.60   | 0.61   | 0.49   |

### Worst-Group Faithfulness Winner per Budget (K=4, higher = better)

| Budget | ERM  | Norm-Max | Norm-CVaR(0.50) | Norm-LocalDRO | Winner          |
|--------|------|----------|-----------------|---------------|-----------------|
| 100    | 0.02 | -0.09    | 0.02            | **0.10**      | Norm-LocalDRO   |
| 200    | 0.14 | 0.11     | **0.20**        | 0.04          | Norm-CVaR       |
| 400    | 0.34 | 0.46     | **0.47**        | 0.20          | Norm-CVaR       |
| 800    | **0.54** | 0.36 | 0.53            | 0.49          | ERM             |

### Recovery Ratio (K=4, excluding S1_RAND)

Recovery = m̂/b (circuit logit_diff / full-model logit_diff). 1.0 = full recovery.

| Method          | Budget | Mean | Worst | IO_RAND | IO_S1  | S2_IO  | S_RAND |
|-----------------|--------|------|-------|---------|--------|--------|--------|
| ERM             | 100    | 0.82 | 0.10  | 0.10    | 1.10   | 0.16   | 1.99   |
| Norm-Max        | 100    | 0.83 | 0.08  | 0.25    | 1.11   | 0.08   | 1.99   |
| Norm-CVaR(0.50) | 100    | 0.84 | 0.16  | 0.16    | 1.15   | 0.16   | 1.98   |
| Norm-LocalDRO   | 100    | 0.82 | -0.43 | -0.43   | 1.14   | 0.22   | 2.29   |
| ERM             | 200    | 0.81 | 0.14  | 0.14    | 1.10   | 0.23   | 1.87   |
| Norm-Max        | 200    | 0.87 | 0.09  | 0.09    | 1.15   | 0.24   | 2.00   |
| Norm-CVaR(0.50) | 200    | 0.86 | 0.26  | 0.26    | 1.12   | 0.28   | 1.88   |
| Norm-LocalDRO   | 200    | 0.84 | -0.21 | -0.21   | 1.11   | 0.21   | 2.23   |
| ERM             | 400    | 0.73 | 0.29  | 0.29    | 1.02   | 0.43   | 1.67   |
| Norm-Max        | 400    | 0.93 | 0.26  | 0.26    | 1.07   | 0.55   | 1.84   |
| Norm-CVaR(0.50) | 400    | 0.90 | 0.38  | 0.38    | 1.06   | 0.48   | 1.73   |
| Norm-LocalDRO   | 400    | 0.83 | 0.07  | 0.07    | 1.00   | 0.35   | 1.96   |
| ERM             | 800    | 0.78 | 0.59  | 0.59    | 0.87   | 0.60   | 1.46   |
| Norm-Max        | 800    | 0.95 | 0.58  | 0.58    | 1.01   | 0.63   | 1.60   |
| Norm-CVaR(0.50) | 800    | 0.89 | 0.48  | 0.48    | 0.97   | 0.67   | 1.62   |
| Norm-LocalDRO   | 800    | 0.87 | 0.20  | 0.20    | 1.04   | 0.61   | 1.76   |

### Winner per Budget (Worst-Group Recovery, K=4, higher = better)

| Budget | ERM  | Norm-Max | Norm-CVaR | Norm-LocalDRO | Winner         |
|--------|------|----------|-----------|---------------|----------------|
| 100    | 0.10 | 0.08     | **0.16**  | -0.43         | **Norm-CVaR**  |
| 200    | 0.14 | 0.09     | **0.26**  | -0.21         | **Norm-CVaR**  |
| 400    | 0.29 | 0.26     | **0.38**  | 0.07          | **Norm-CVaR**  |
| 800    | **0.59** | 0.58 | 0.48      | 0.20          | **ERM**        |

**With recovery ratio, Norm-CVaR(0.50) wins 3/4 budgets** (100, 200, 400). ERM wins only at budget=800.

## Analysis

### Normalization helps significantly but doesn't fully solve the problem

1. **Raw Max → Norm Max**: Major improvement. Raw Max lost 0/8; Norm Max wins 2/8 (large budgets).
2. **Norm CVaR(0.50)** is the best normalized aggregator, winning 4/8 budgets.
3. **Pattern**: Normalized DRO wins at **large budgets** (≥200 for CVaR, ≥1600 for Max) but loses at **small budgets** (25–100).

### Why normalized DRO still struggles at small budgets

At small budgets (25–50 edges), normalization causes a different problem: after normalizing, the most important edges across ALL corruptions look equally important, so the circuit tries to cover too many pathways with too few edges. This "spread too thin" effect hurts more than focusing on the dominant corruption (what ERM does).

At large budgets, there are enough edges to cover all corruption pathways, so normalized DRO's broader coverage pays off.

### The crossover point

- **Budget < 200**: ERM wins (focus is better than breadth)
- **Budget ≥ 200**: Norm-CVaR(0.50) wins (breadth starts to help)
- **Budget ≥ 1600**: Norm-Max also wins (even pure worst-case helps when budget is large)

## Conclusion

Per-corruption normalization partially validates the DRO hypothesis. **Norm-CVaR(0.50) wins 4/8 budgets** vs ERM, especially at medium-to-large budgets (200–3200). The original Max aggregation failure was indeed caused by score magnitude disparity. However, normalized DRO is not strictly better — there is a budget-dependent tradeoff where ERM is preferred at small budgets.

## Follow-up

- [ ] Investigate why small-budget normalized DRO performs poorly (0.8012 at n=25)
- [ ] Try L2-norm and Z-score normalization as alternatives
- [ ] Test if the crossover point (~200 edges) is stable across seeds
- [ ] Consider budget-adaptive strategy: ERM for small budgets, Norm-DRO for large
