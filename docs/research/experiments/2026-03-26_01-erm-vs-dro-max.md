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
- **Task:** IOI, K=4 corruptions (S2_IO, IO_RAND, S_RAND, IO_S1)
- **Excluded:** S1_RAND (trivial corruption — full model is barely affected, making normalized metrics unstable; see Note below)
- **N examples:** 200
- **Edge budgets:** [25, 50, 100, 200, 400, 800, 1600, 3200]
- **Commit:** b417f30

> **Note on S1_RAND exclusion:** S1_RAND replaces the S1 name with a random name. In IOI, the model's prediction depends primarily on IO and S2 co-occurrence — S1 has minimal influence. As a result, $M_f(x) \approx M_f(\tilde{x}_{\text{S1\_RAND}})$, causing the normalized faithfulness denominator $b - b' \approx 0$ and producing meaningless values (e.g. Faith > 7). Recovery ratio for S1_RAND is ~0.98 across all methods, confirming it is a trivial corruption that does not differentiate circuits.

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

### Normalized Faithfulness (K=4, excluding S1_RAND)

Faith=1: perfect recovery. Faith=0: no better than corrupt baseline.

| Method    | Budget | Mean Faith | Worst-Group | IO_RAND | IO_S1  | S2_IO  | S_RAND |
|-----------|--------|-----------|-------------|---------|--------|--------|--------|
| ERM       | 100    | 0.49      | 0.02        | 0.69    | 0.93   | 0.02   | 0.34   |
| DRO-max   | 100    | 0.41      | 0.10        | 0.59    | 0.70   | 0.10   | 0.25   |
| Local DRO | 100    | 0.34      | 0.04        | 0.48    | 0.68   | 0.04   | 0.17   |
| ERM       | 200    | 0.46      | 0.14        | 0.69    | 0.59   | 0.14   | 0.43   |
| DRO-max   | 200    | 0.40      | 0.05        | 0.49    | 0.67   | 0.05   | 0.39   |
| Local DRO | 200    | 0.38      | -0.13       | 0.58    | 0.77   | -0.13  | 0.30   |
| ERM       | 400    | 0.65      | 0.34        | 0.74    | 0.95   | 0.34   | 0.57   |
| DRO-max   | 400    | 0.57      | 0.30        | 0.67    | 0.82   | 0.30   | 0.48   |
| Local DRO | 400    | 0.36      | -0.03       | 0.65    | 0.45   | -0.03  | 0.38   |
| ERM       | 800    | 0.75      | 0.54        | 0.85    | 0.89   | 0.54   | 0.71   |
| DRO-max   | 800    | 0.70      | 0.62        | 0.80    | 0.74   | 0.63   | 0.62   |
| Local DRO | 800    | 0.54      | 0.41        | 0.70    | 0.52   | 0.41   | 0.54   |

### Recovery Ratio (K=4, excluding S1_RAND)

Recovery = m̂/b (circuit logit_diff / full-model logit_diff). 1.0 = full recovery.

| Method    | Budget | Mean | Worst | IO_RAND | IO_S1  | S2_IO  | S_RAND |
|-----------|--------|------|-------|---------|--------|--------|--------|
| ERM       | 100    | 0.82 | 0.10  | 0.10    | 1.10   | 0.16   | 1.99   |
| DRO-max   | 100    | 0.83 | -0.12 | -0.12   | 1.10   | 0.21   | 2.12   |
| Local DRO | 100    | 0.81 | -0.31 | -0.31   | 1.15   | 0.17   | 2.22   |
| ERM       | 200    | 0.81 | 0.14  | 0.14    | 1.10   | 0.23   | 1.87   |
| DRO-max   | 200    | 0.70 | -0.28 | -0.28   | 1.07   | 0.19   | 1.93   |
| Local DRO | 200    | 0.77 | -0.06 | -0.06   | 1.09   | 0.09   | 2.05   |
| ERM       | 400    | 0.73 | 0.29  | 0.29    | 1.02   | 0.43   | 1.67   |
| DRO-max   | 400    | 0.78 | 0.13  | 0.13    | 1.05   | 0.41   | 1.82   |
| Local DRO | 400    | 0.76 | 0.13  | 0.13    | 1.03   | 0.18   | 1.93   |
| ERM       | 800    | 0.78 | 0.59  | 0.59    | 0.87   | 0.60   | 1.46   |
| DRO-max   | 800    | 0.85 | 0.45  | 0.45    | 0.90   | 0.69   | 1.58   |
| Local DRO | 800    | 0.82 | 0.27  | 0.27    | 1.05   | 0.46   | 1.69   |

### Winner per Budget

| Budget | Norm Faith Worst-Group |        | Recovery Worst-Group |        |
|--------|------------------------|--------|----------------------|--------|
|        | Winner | Value         |        | Winner | Value        |
| 100    | DRO-max | 0.10         |        | ERM    | 0.10         |
| 200    | ERM     | 0.14         |        | ERM    | 0.14         |
| 400    | ERM     | 0.34         |        | ERM    | 0.29         |
| 800    | DRO-max | 0.62         |        | ERM    | 0.59         |

Two metrics give different conclusions:
- **Normalized faithfulness:** DRO-max wins 2/4 (100, 800)
- **Recovery ratio:** ERM wins 4/4

### Key Observations

1. **IO_RAND and S2_IO are the hardest corruptions** across all methods.
2. **DRO-max performs worse on IO_RAND than ERM** at budgets 100–400 (negative recovery = inverted logit diff).
3. **S_RAND recovery > 1.0 for all methods** — the circuit amplifies the logit diff compared to the full model under S_RAND corruption.
4. The gap narrows at large budgets.

## Analysis

**The main hypothesis is not clearly supported.** ERM outperforms raw DRO-max on worst-group recovery at all budgets. The Max aggregator is dominated by the corruption with the largest EAP score magnitudes (S_RAND), not the hardest corruption for evaluation (IO_RAND / S2_IO).

**Key insight**: The corruption that produces the largest EAP scores is NOT the same as the corruption that is hardest for evaluation. This motivates per-corruption score normalization (see Exp 01v2).

## Follow-up

- [x] ~~Try normalizing per-corruption scores before aggregation~~ → Exp 01v2
- [ ] Try the Plan B approach (learnable gates) which directly optimizes the DRO evaluation objective
