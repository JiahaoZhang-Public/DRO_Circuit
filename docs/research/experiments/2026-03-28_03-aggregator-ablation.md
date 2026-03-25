# Experiment: Aggregator Spectrum Ablation

**Date:** 2026-03-28
**Author:** Jiahao Zhang
**Status:** planned

## Goal

Compare all DRO aggregators (mean, max, CVaR×4, softmax×4, local_dro) to determine whether there exists a Pareto-optimal tradeoff between mean and worst-group faithfulness.

## Environment

- **Server:** gpuhub-root-rtx4090-48
- **GPU:** RTX 4090 48GB
- **Model:** GPT-2 small
- **Task:** IOI, K=5 corruptions
- **N examples:** 200
- **Data:** Reuses Exp 1 outputs (`outputs/exp01_erm_vs_dro/`)
- **Commit:** TBD

## Exact Commands

```bash
cd /root/projects/DRO_Circuit && conda activate dro

# Analysis only — no new compute needed
python experiments/analyze_results.py \
    --input_dir outputs/exp01_erm_vs_dro \
    --output_dir outputs/exp01_erm_vs_dro/figures \
    --budget 200
```

## Results

### Aggregator comparison at Budget=200

| Aggregator | Worst-Group Faith | Mean Faith | Gap |
|------------|-------------------|-----------|-----|
| erm_mean   |                   |           |     |
| max        |                   |           |     |
| cvar_0.17  |                   |           |     |
| cvar_0.33  |                   |           |     |
| cvar_0.50  |                   |           |     |
| cvar_0.67  |                   |           |     |
| softmax_0.01 |                 |           |     |
| softmax_0.1  |                 |           |     |
| softmax_1.0  |                 |           |     |
| softmax_10.0 |                 |           |     |
| local_dro  |                   |           |     |

### Figures

- `aggregator_spectrum.pdf` — bar chart of all aggregators
- `pareto.pdf` — mean vs worst faithfulness scatter

## Analysis

TBD — Check whether:
- CVaR(0.33–0.50) provides a good Pareto-optimal tradeoff
- Softmax(tau=0.01) ≈ Max; Softmax(tau=10.0) ≈ Mean
- Aggregators form a smooth interpolation spectrum

## Conclusion

TBD

## Follow-up

- [ ] Identify the Pareto-optimal aggregator(s)
- [ ] If CVaR is best → recommend a default alpha
