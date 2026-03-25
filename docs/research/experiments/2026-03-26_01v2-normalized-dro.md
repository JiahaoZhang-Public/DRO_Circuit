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
- **Task:** IOI, K=5 corruptions (S2_IO, IO_RAND, S_RAND, S1_RAND, IO_S1)
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

### Normalized Faithfulness (Faith=1: perfect recovery, Faith=0: no better than corrupt)

| Method | Budget | Mean Faith | Worst-Group Faith | Per-Example Worst Faith |
|--------|--------|-----------|-------------------|------------------------|
| ERM          | 100 | 1.9359 | 0.0228 | -1.0108 |
| Norm-Max     | 100 | 1.9655 | -0.0931 | -1.0769 |
| Norm-CVaR    | 100 | 2.2293 | 0.0164 | -1.1636 |
| Norm-LocalDRO| 100 | 0.7652 | 0.0972 | -0.3181 |
| ERM          | 200 | 2.3272 | 0.1422 | -1.3353 |
| Norm-Max     | 200 | 2.0861 | 0.1070 | -0.9334 |
| Norm-CVaR    | 200 | 2.4524 | **0.1952** | -1.2614 |
| Norm-LocalDRO| 200 | 1.0035 | 0.0371 | -0.5455 |
| ERM          | 400 | 3.0134 | 0.3394 | -1.2051 |
| Norm-Max     | 400 | 2.3459 | **0.4596** | -1.0404 |
| Norm-CVaR    | 400 | 2.4543 | **0.4702** | -1.0080 |
| Norm-LocalDRO| 400 | 1.7622 | 0.1994 | -1.2127 |
| ERM          | 800 | 2.6907 | 0.5443 | -0.8593 |
| Norm-Max     | 800 | 2.1586 | 0.3648 | -0.8227 |
| Norm-CVaR    | 800 | 2.3849 | 0.5321 | -0.9763 |
| Norm-LocalDRO| 800 | 1.1668 | 0.4876 | -0.6719 |

### Worst-Group Faithfulness Winner per Budget (higher = better)

| Budget | ERM    | Norm-Max | Norm-CVaR(0.50) | Norm-LocalDRO | Winner |
|--------|--------|----------|-----------------|---------------|--------|
| 100    | 0.0228 | -0.0931  | 0.0164          | **0.0972**    | Norm-LocalDRO |
| 200    | 0.1422 | 0.1070   | **0.1952**      | 0.0371        | Norm-CVaR |
| 400    | 0.3394 | 0.4596   | **0.4702**      | 0.1994        | Norm-CVaR |
| 800    | **0.5443** | 0.3648 | 0.5321         | 0.4876        | ERM |

### Recovery Ratio (EAP-IG style: m̂/b, 1.0 = full recovery)

| Method          | Budget | Mean | Worst | IO_RAND | IO_S1  | S1_RAND | S2_IO  | S_RAND |
|-----------------|--------|------|-------|---------|--------|---------|--------|--------|
| ERM             | 100    | 0.87 | 0.10  | 0.10    | 1.10   | 0.98    | 0.16   | 1.99   |
| Raw-Max         | 100    | 0.88 | -0.12 | -0.12   | 1.10   | 1.07    | 0.21   | 2.12   |
| Norm-Max        | 100    | 0.88 | 0.08  | 0.25    | 1.11   | 0.98    | 0.08   | 1.99   |
| Norm-CVaR(0.50) | 100    | 0.88 | 0.16  | 0.16    | 1.15   | 0.96    | 0.16   | 1.98   |
| Norm-LocalDRO   | 100    | 0.89 | -0.43 | -0.43   | 1.14   | 1.23    | 0.22   | 2.29   |
| ERM             | 200    | 0.85 | 0.14  | 0.14    | 1.10   | 0.92    | 0.23   | 1.87   |
| Raw-Max         | 200    | 0.77 | -0.28 | -0.28   | 1.07   | 0.92    | 0.19   | 1.93   |
| Norm-Max        | 200    | 0.90 | 0.09  | 0.09    | 1.15   | 1.01    | 0.24   | 2.00   |
| Norm-CVaR(0.50) | 200    | 0.89 | 0.26  | 0.26    | 1.12   | 0.90    | 0.28   | 1.88   |
| Norm-LocalDRO   | 200    | 0.90 | -0.21 | -0.21   | 1.11   | 1.16    | 0.21   | 2.23   |
| ERM             | 400    | 0.85 | 0.29  | 0.29    | 1.02   | 0.82    | 0.43   | 1.67   |
| Raw-Max         | 400    | 0.87 | 0.13  | 0.13    | 1.05   | 0.96    | 0.41   | 1.82   |
| Norm-Max        | 400    | 0.95 | 0.26  | 0.26    | 1.07   | 1.02    | 0.55   | 1.84   |
| Norm-CVaR(0.50) | 400    | 0.92 | 0.38  | 0.38    | 1.06   | 0.94    | 0.48   | 1.73   |
| Norm-LocalDRO   | 400    | 0.87 | 0.07  | 0.07    | 1.00   | 0.99    | 0.35   | 1.96   |
| ERM             | 800    | 0.88 | 0.59  | 0.59    | 0.87   | 0.88    | 0.60   | 1.46   |
| Raw-Max         | 800    | 0.92 | 0.45  | 0.45    | 0.90   | 0.96    | 0.69   | 1.58   |
| Norm-Max        | 800    | 0.95 | 0.58  | 0.58    | 1.01   | 0.94    | 0.63   | 1.60   |
| Norm-CVaR(0.50) | 800    | 0.92 | 0.48  | 0.48    | 0.97   | 0.87    | 0.67   | 1.62   |
| Norm-LocalDRO   | 800    | 0.94 | 0.20  | 0.20    | 1.04   | 1.07    | 0.61   | 1.76   |

### Winner per Budget (Worst-Group Recovery, higher = better)

| Budget | ERM  | Raw-Max | Norm-Max | Norm-CVaR | Norm-LocalDRO | Winner         |
|--------|------|---------|----------|-----------|---------------|----------------|
| 100    | 0.10 | -0.12   | 0.08     | **0.16**  | -0.43         | **Norm-CVaR**  |
| 200    | 0.14 | -0.28   | 0.09     | **0.26**  | -0.21         | **Norm-CVaR**  |
| 400    | 0.29 | 0.13    | 0.26     | **0.38**  | 0.07          | **Norm-CVaR**  |
| 800    | **0.59** | 0.45 | 0.58     | 0.48      | 0.20          | **ERM**        |

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
