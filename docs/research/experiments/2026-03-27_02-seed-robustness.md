# Experiment: Seed Robustness

**Date:** 2026-03-27
**Author:** Jiahao Zhang
**Status:** planned

## Goal

Verify that the ERM vs DRO-max difference is robust across different random seeds (dataset generation), not an artifact of one particular dataset.

## Environment

- **Server:** gpuhub-root-rtx4090-48
- **GPU:** RTX 4090 48GB
- **Model:** GPT-2 small
- **Task:** IOI, K=5 corruptions
- **N examples:** 200
- **Edge budgets:** [100, 200, 400, 800]
- **Seeds:** [42, 123, 456]
- **Commit:** TBD

## Exact Commands

```bash
cd /root/projects/DRO_Circuit && conda activate dro

for SEED in 42 123 456; do
    echo "=== Running seed=${SEED} ==="
    python experiments/comprehensive_experiment.py \
        --n_examples 200 --batch_size 25 --device cuda --seed ${SEED} \
        --edge_budgets 100 200 400 800 \
        --output_dir outputs/exp02_seed_${SEED}
done
```

## Results

### Key metrics (Budget=200)

| Seed | ERM Worst-Group | DRO Worst-Group | Δ |
|------|-----------------|-----------------|---|
| 42   |                 |                 |   |
| 123  |                 |                 |   |
| 456  |                 |                 |   |

### Cross-seed consistency

DRO-max wins at budget=200 for _/3 seeds.

## Analysis

TBD

## Conclusion

TBD

## Follow-up

- [ ] If consistent → the effect is real, proceed to aggregator ablation
- [ ] If inconsistent → investigate which seed fails and why
