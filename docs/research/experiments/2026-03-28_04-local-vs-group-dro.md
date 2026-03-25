# Experiment: Local DRO vs Group DRO

**Date:** 2026-03-28
**Author:** Jiahao Zhang
**Status:** planned

## Goal

Determine whether Local DRO (per-example worst-case scoring) offers advantages over Group DRO (max over corruption-averaged scores) on the per-example worst-case faithfulness metric.

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

# Extract comparison from summary.json
python -c "
import json
with open('outputs/exp01_erm_vs_dro/summary.json') as f:
    s = json.load(f)

print(f'{\"Budget\":<8} {\"Method\":<15} {\"Worst\":>10} {\"Mean\":>10} {\"Gap\":>10}')
print('-' * 65)
for budget in [25, 50, 100, 200, 400, 800, 1600, 3200]:
    for method in ['erm_mean', 'max', 'local_dro']:
        name = f'dro_{method}_n{budget}'
        if name in s:
            e = s[name]
            print(f'{budget:<8} {method:<15} {e[\"worst\"]:10.4f} {e[\"mean\"]:10.4f} {e[\"gap\"]:10.4f}')
    print()
"
```

## Results

### Key metrics

| Budget | Method    | Worst-Group | Mean   | Gap    |
|--------|-----------|-------------|--------|--------|
| 200    | erm_mean  |             |        |        |
| 200    | max       |             |        |        |
| 200    | local_dro |             |        |        |

### Local DRO vs Max win rate

Local DRO beats Max on worst-group at _/8 budgets.

## Analysis

TBD — Check whether:
- Local DRO matches or exceeds Group DRO on worst-group
- Local DRO shows improvement on per-example worst-case metric
- Mean faithfulness tradeoff is acceptable

## Conclusion

TBD

## Follow-up

- [ ] If Local DRO wins → recommend as default for strict robustness
- [ ] If similar → Group DRO (max) may be preferred for simplicity (no per-example scoring needed)
