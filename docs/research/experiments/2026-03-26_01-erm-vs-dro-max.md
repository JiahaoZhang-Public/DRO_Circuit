# Experiment: ERM vs DRO-max Head-to-Head

**Date:** 2026-03-26
**Author:** Jiahao Zhang
**Status:** planned

## Goal

Determine whether DRO-max circuits have better worst-group faithfulness than ERM circuits, and whether the difference is consistent across 8 edge budgets.

## Environment

- **Server:** gpuhub-root-rtx4090-48
- **GPU:** RTX 4090 48GB
- **Model:** GPT-2 small
- **Task:** IOI, K=5 corruptions (S2_IO, IO_RAND, S_RAND, S1_RAND, IO_S1)
- **N examples:** 200
- **Edge budgets:** [25, 50, 100, 200, 400, 800, 1600, 3200]
- **Commit:** TBD

## Exact Commands

```bash
cd /root/projects/DRO_Circuit && conda activate dro

python experiments/comprehensive_experiment.py \
    --n_examples 200 --batch_size 25 --device cuda --seed 42 \
    --edge_budgets 25 50 100 200 400 800 1600 3200 \
    --output_dir outputs/exp01_erm_vs_dro

python experiments/analyze_results.py \
    --input_dir outputs/exp01_erm_vs_dro \
    --output_dir outputs/exp01_erm_vs_dro/figures
```

## Results

### Key metrics

| Budget | ERM Mean | ERM Worst-Group | DRO Mean | DRO Worst-Group | Δ Worst |
|--------|---------|-----------------|---------|-----------------|---------|
| 25     |         |                 |         |                 |         |
| 50     |         |                 |         |                 |         |
| 100    |         |                 |         |                 |         |
| 200    |         |                 |         |                 |         |
| 400    |         |                 |         |                 |         |
| 800    |         |                 |         |                 |         |
| 1600   |         |                 |         |                 |         |
| 3200   |         |                 |         |                 |         |

### DRO Win Rate

DRO-max wins worst-group at _/8 budgets.

### Figures

- `worst_vs_budget.pdf` — headline plot
- `corruption_heatmap.pdf` — per-corruption breakdown
- `gap_vs_budget.pdf` — robustness gap
- `pareto.pdf` — mean vs worst faithfulness

## Analysis

TBD

## Conclusion

TBD

## Follow-up

- [ ] If DRO wins → proceed to Exp 2 (seed robustness)
- [ ] If DRO loses → investigate which corruption drives the result
- [ ] Check if the gap narrows at large budgets (expected)
