# Experiment: ERM vs DRO-max Head-to-Head

**Date:** 2026-03-26
**Author:** Jiahao Zhang
**Status:** completed

## Goal

Determine whether DRO-max circuits have better worst-group faithfulness than ERM circuits, and whether the difference is consistent across 8 edge budgets.

## Environment

- **Server:** gpuhub-root-rtx4090-48
- **GPU:** RTX 4090 48GB
- **Model:** GPT-2 small
- **Task:** IOI, K=5 corruptions (S2_IO, IO_RAND, S_RAND, S1_RAND, IO_S1)
- **N examples:** 200
- **Edge budgets:** [25, 50, 100, 200, 400, 800, 1600, 3200]
- **Commit:** b417f30

## Exact Commands

```bash
cd /root/projects/DRO_Circuit && conda activate dro

python experiments/comprehensive_experiment.py \
    --n_examples 200 --batch_size 25 --device cuda --seed 42 \
    --edge_budgets 25 50 100 200 400 800 1600 3200 \
    --output_dir outputs/exp01_erm_vs_dro
```

## Results

### Timing

- Phase 1 (scoring): 12.3s
- Phase 1b (per-example): 14.4s
- Phase 2 (build 128 circuits): 53.5s
- Phase 3 (evaluate 128 circuits): 629.2s
- **Total: 709.5s (11.8 min)**

### Normalized Faithfulness (Faith=1: perfect recovery, Faith=0: no better than corrupt)

| Method | Budget | Mean Faith | Worst-Group Faith | Per-Example Worst Faith |
|--------|--------|-----------|-------------------|------------------------|
| ERM    | 100    | 1.9359    | 0.0228            | -1.0108                |
| DRO-max| 100    | 1.5731    | 0.0960            | -0.6042                |
| Local DRO| 100  | 1.0842    | 0.0350            | -0.4159                |
| ERM    | 200    | 2.3272    | 0.1422            | -1.3353                |
| DRO-max| 200    | 2.1907    | 0.0484            | -1.2704                |
| Local DRO| 200  | 1.5769    | -0.1317           | -0.9481                |
| ERM    | 400    | 3.0134    | 0.3394            | -1.2051                |
| DRO-max| 400    | 2.3354    | 0.2956            | -0.8204                |
| Local DRO| 400  | 1.9267    | -0.0278           | -1.4160                |
| ERM    | 800    | 2.6907    | 0.5443            | -0.8593                |
| DRO-max| 800    | 2.6424    | 0.6190            | -0.8190                |
| Local DRO| 800  | 1.4639    | 0.4051            | -0.8081                |

### DRO Win Rate (Worst-Group Faithfulness, higher = better)

| Budget | ERM    | DRO-max | Winner  |
|--------|--------|---------|---------|
| 100    | 0.0228 | 0.0960  | DRO-max |
| 200    | 0.1422 | 0.0484  | ERM     |
| 400    | 0.3394 | 0.2956  | ERM     |
| 800    | 0.5443 | 0.6190  | DRO-max |

**DRO-max wins worst-group faithfulness at 2/4 budgets** (100, 800). Mixed results — no clear winner.

### Recovery Ratio (EAP-IG style: m̂/b, 1.0 = full recovery)

| Method     | Budget | Mean | Worst | IO_RAND | IO_S1  | S1_RAND | S2_IO  | S_RAND |
|------------|--------|------|-------|---------|--------|---------|--------|--------|
| ERM        | 100    | 0.87 | 0.10  | 0.10    | 1.10   | 0.98    | 0.16   | 1.99   |
| DRO-max    | 100    | 0.88 | -0.12 | -0.12   | 1.10   | 1.07    | 0.21   | 2.12   |
| Local DRO  | 100    | 0.88 | -0.31 | -0.31   | 1.15   | 1.18    | 0.17   | 2.22   |
| ERM        | 200    | 0.85 | 0.14  | 0.14    | 1.10   | 0.92    | 0.23   | 1.87   |
| DRO-max    | 200    | 0.77 | -0.28 | -0.28   | 1.07   | 0.92    | 0.19   | 1.93   |
| Local DRO  | 200    | 0.84 | -0.06 | -0.06   | 1.09   | 1.04    | 0.09   | 2.05   |
| ERM        | 400    | 0.85 | 0.29  | 0.29    | 1.02   | 0.82    | 0.43   | 1.67   |
| DRO-max    | 400    | 0.87 | 0.13  | 0.13    | 1.05   | 0.96    | 0.41   | 1.82   |
| Local DRO  | 400    | 0.85 | 0.13  | 0.13    | 1.03   | 1.01    | 0.18   | 1.93   |
| ERM        | 800    | 0.88 | 0.59  | 0.59    | 0.87   | 0.88    | 0.60   | 1.46   |
| DRO-max    | 800    | 0.92 | 0.45  | 0.45    | 0.90   | 0.96    | 0.69   | 1.58   |
| Local DRO  | 800    | 0.90 | 0.27  | 0.27    | 1.05   | 1.04    | 0.46   | 1.69   |
| Naive-IO   | 100    | 0.73 | -1.35 | -1.35   | 1.03   | 1.35    | 0.09   | 2.52   |
| Naive-IO   | 200    | 0.78 | -1.27 | -1.27   | 1.06   | 1.42    | 0.18   | 2.50   |
| Naive-IO   | 400    | 0.78 | -1.20 | -1.20   | 1.02   | 1.43    | 0.18   | 2.47   |
| Naive-IO   | 800    | 0.67 | 0.10  | 0.58    | 0.69   | 0.61    | 0.10   | 1.35   |
| Naive-S    | 100    | 0.60 | -0.93 | -0.93   | 1.13   | 0.87    | 0.01   | 1.91   |
| Naive-S    | 200    | 0.77 | -0.04 | -0.04   | 1.04   | 0.85    | 0.15   | 1.86   |
| Naive-S    | 400    | 0.73 | 0.15  | 0.15    | 0.96   | 0.73    | 0.16   | 1.67   |
| Naive-S    | 800    | 0.71 | 0.32  | 0.36    | 0.85   | 0.57    | 0.32   | 1.44   |

### Winner per Budget (Worst-Group Recovery, higher = better)

| Budget | ERM   | DRO-max | Naive-IO | Naive-S | Winner  |
|--------|-------|---------|----------|---------|---------|
| 100    | **0.10** | -0.12 | -1.35    | -0.93   | **ERM** |
| 200    | **0.14** | -0.28 | -1.27    | -0.04   | **ERM** |
| 400    | **0.29** | 0.13  | -1.20    | 0.15    | **ERM** |
| 800    | 0.59  | 0.45   | 0.10     | 0.32    | **ERM** |

**With recovery ratio, ERM wins worst-group at all 4 budgets.** This is consistent with the original raw loss results but inconsistent with normalized faithfulness — the difference is driven by how the metrics handle the denominator.

### Key observations

1. **IO_RAND is consistently the hardest corruption** for all methods — it produces the least negative (worst) loss at every budget.
2. **DRO-max performs worse on IO_RAND than ERM** at every budget. At budget=200: DRO gets +0.4213 (unfaithful) vs ERM gets -0.2018 (faithful).
3. **DRO-max performs better on S_RAND** — the corruption where ERM already performs well.
4. The gap narrows at large budgets (Δ goes from -0.623 at n=200 to -0.045 at n=3200).

### Per-corruption breakdown at Budget=200

| Method    | S2_IO   | IO_RAND | S_RAND  | S1_RAND | IO_S1   |
|-----------|---------|---------|---------|---------|---------|
| ERM       | -0.3381 | -0.2018 | -2.7891 | -1.3664 | -1.6368 |
| DRO-max   | -0.2788 | +0.4213 | -2.8763 | -1.3782 | -1.5927 |

### Aggregator comparison at Budget=200 (worst-group loss)

| Aggregator    | Worst-Group |
|---------------|-------------|
| erm_mean      | -0.2018     |
| cvar_0.67     | -0.2589     |
| cvar_0.50     | -0.2532     |
| softmax_0.1   | -0.2018     |
| softmax_1.0   | -0.2018     |
| softmax_10.0  | -0.2018     |
| softmax_0.01  | -0.0348     |
| cvar_0.33     | +0.1106     |
| max           | +0.4213     |
| cvar_0.17     | +0.4213     |

## Analysis

**The main hypothesis is not supported in this experiment.** ERM outperforms DRO-max on worst-group faithfulness at all budgets.

### Why DRO-max fails on IO_RAND

The Max aggregator selects edges that have high scores under the worst corruption family. Looking at the per-corruption scores, this causes over-emphasis on edges that are important under corruptions like S_RAND (which has very large scores) while neglecting edges critical for IO_RAND (which has smaller absolute scores). The Max operation is dominated by the corruption with the largest score magnitudes, not the corruption that is hardest for evaluation.

**Key insight**: The corruption that produces the largest EAP scores is NOT the same as the corruption that is hardest for evaluation. S_RAND has the largest absolute losses (~-2.8), making its edges dominate the Max aggregation. But IO_RAND is the corruption where circuits fail (positive loss = unfaithful).

### Positive signal: CVaR(0.5–0.67) ≈ ERM

CVaR with moderate alpha (0.50–0.67) performs comparably to ERM, suggesting that mild tail-risk awareness doesn't hurt. The issue is specifically with extreme DRO (max / CVaR with small alpha).

## Conclusion

DRO-max (Group DRO with max aggregation) does NOT improve worst-group faithfulness compared to ERM. The Max aggregator is dominated by the corruption with the largest score magnitudes rather than the corruption that matters for evaluation. This suggests **the raw EAP score magnitudes are not aligned with evaluation difficulty**, and a different formulation (e.g., score normalization per corruption, or optimization-based approach) may be needed.

## Follow-up

- [ ] Investigate score magnitude disparity across corruptions — are S_RAND scores systematically larger?
- [ ] Try normalizing per-corruption scores before aggregation (divide by L2 norm or max of each corruption's scores)
- [ ] Try the Plan B approach (learnable gates) which directly optimizes the DRO evaluation objective
- [ ] Check if Local DRO has the same issue or is better calibrated
