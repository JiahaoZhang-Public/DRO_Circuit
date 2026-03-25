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

### ERM vs DRO-max (logit_diff_loss, lower = more faithful)

| Budget | ERM Edges | ERM Worst | ERM Mean  | DRO Edges | DRO Worst | DRO Mean  | Δ Worst |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|---------|
| 25     | 10        | 0.1489    | -1.3805   | 9         | 0.3179    | -1.3306   | -0.169  |
| 50     | 25        | 0.1284    | -1.3155   | 26        | 0.4236    | -1.3526   | -0.295  |
| 100    | 59        | -0.1481   | -1.2912   | 57        | 0.1799    | -1.3069   | -0.328  |
| 200    | 159       | -0.2018   | -1.2664   | 166       | 0.4213    | -1.1409   | -0.623  |
| 400    | 369       | -0.4373   | -1.2631   | 365       | -0.1867   | -1.3006   | -0.251  |
| 800    | 759       | -0.8835   | -1.3123   | 755       | -0.6758   | -1.3651   | -0.208  |
| 1600   | 1561      | -1.1304   | -1.3922   | 1561      | -1.0368   | -1.4420   | -0.094  |
| 3200   | 3167      | -1.3574   | -1.4653   | 3169      | -1.3125   | -1.4541   | -0.045  |

### DRO Win Rate

**DRO-max wins worst-group at 0/8 budgets.** ERM outperforms DRO-max on worst-group faithfulness at all budgets.

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
