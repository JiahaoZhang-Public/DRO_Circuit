# Experiment: Circuit Structure Analysis

**Date:** 2026-03-29
**Author:** Jiahao Zhang
**Status:** planned

## Goal

Characterize structural differences between ERM and DRO circuits: edge overlap, layer distribution, and component composition (attention heads vs MLPs).

## Environment

- **Server:** gpuhub-root-rtx4090-48
- **GPU:** RTX 4090 48GB
- **Model:** GPT-2 small
- **Task:** IOI, K=5 corruptions
- **N examples:** 200
- **Data:** Reuses Exp 1 circuit masks (`outputs/exp01_erm_vs_dro/circuit_masks/`)
- **Commit:** TBD

## Exact Commands

```bash
cd /root/projects/DRO_Circuit && conda activate dro

# Generate structure visualizations
python experiments/analyze_results.py \
    --input_dir outputs/exp01_erm_vs_dro \
    --output_dir outputs/exp01_erm_vs_dro/figures

# Compute Jaccard overlap at each budget
python -c "
import torch

budgets = [25, 50, 100, 200, 400, 800, 1600, 3200]
methods = ['erm_mean', 'max', 'local_dro']

for b in budgets:
    masks = {}
    for m in methods:
        path = f'outputs/exp01_erm_vs_dro/circuit_masks/dro_{m}_n{b}.pt'
        try:
            data = torch.load(path, map_location='cpu')
            masks[m] = data['in_graph'].bool()
        except:
            pass

    if 'erm_mean' in masks and 'max' in masks:
        erm, dro = masks['erm_mean'], masks['max']
        inter = (erm & dro).sum().item()
        union = (erm | dro).sum().item()
        jaccard = inter / union if union > 0 else 0
        erm_only = (erm & ~dro).sum().item()
        dro_only = (dro & ~erm).sum().item()
        print(f'Budget {b:>5}: overlap={inter}, Jaccard={jaccard:.3f}, ERM-only={erm_only}, DRO-only={dro_only}')
"
```

## Results

### Jaccard overlap (ERM vs DRO-max)

| Budget | Jaccard | ERM-only | DRO-only | Shared |
|--------|---------|----------|----------|--------|
| 25     |         |          |          |        |
| 50     |         |          |          |        |
| 100    |         |          |          |        |
| 200    |         |          |          |        |
| 400    |         |          |          |        |
| 800    |         |          |          |        |
| 1600   |         |          |          |        |
| 3200   |         |          |          |        |

### Figures

- `edge_overlap.pdf` — overlap matrix between methods
- `layer_density.pdf` — layer distribution of edges
- `circuit_composition.pdf` — edge type breakdown

## Analysis

TBD — Check whether:
- Jaccard < 1.0 (circuits are structurally different)
- Overlap increases with budget
- DRO circuits span more diverse layers at small budgets
- DRO circuits include more edges in early/middle layers (IO path)

## Conclusion

TBD

## Follow-up

- [ ] Compare top-20 edges unique to DRO vs ERM — which attention heads differ?
- [ ] If DRO includes known IOI circuit components → validates the robustness hypothesis
