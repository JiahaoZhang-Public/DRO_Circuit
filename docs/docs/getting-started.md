# Getting Started

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA GPU (recommended; CPU works but is slow)

## Installation

```bash
# Clone with vendor submodules
git clone --recursive https://github.com/JiahaoZhang-Public/DRO_Circuit.git
cd DRO_Circuit

# Install the package
pip install -e .
```

If you already cloned without `--recursive`, initialize submodules separately:

```bash
git submodule update --init --recursive
```

### Vendor dependencies

The project relies on two vendored libraries (included as git submodules):

| Submodule | Path | Purpose |
|-----------|------|---------|
| **EAP-IG** | `vendor/EAP-IG/` | Edge Attribution Patching — single-pass edge scoring |
| **ACDC** | `vendor/Automatic-Circuit-Discovery/` | IOI task dataset and corruption generation |

These are automatically available via `sys.path` manipulation in the package — no separate installation needed.

## First Run

Run a single DRO circuit discovery experiment on GPT-2 small / IOI:

```bash
python -m dro_circuit.scripts.run \
  --task ioi \
  --n_examples 100 \
  --n_edges 200 \
  --aggregator max \
  --device cuda \
  --output_dir outputs/first_run
```

This will:
1. Load GPT-2 small with TransformerLens hooks
2. Generate 100 IOI examples with 5 corruption variants each
3. Score all edges under each corruption via EAP-IG
4. Aggregate scores with Max (worst-case) rule
5. Select top 200 edges → sparse circuit
6. Evaluate circuit faithfulness under all corruptions

Output files in `outputs/first_run/`:

| File | Content |
|------|---------|
| `circuit.pt` | Circuit graph with `in_graph` mask |
| `scores.pt` | Per-corruption edge scores (K × n_fwd × n_bwd) |
| `results.json` | Evaluation metrics (worst, mean, gap, per-corruption) |
| `config.json` | Full experiment configuration |

## Aggregator Options

```bash
# Max (pure worst-case)
--aggregator max

# CVaR — α controls tail risk (0=max, 1=mean)
--aggregator cvar --cvar_alpha 0.5

# Softmax — τ controls temperature (0→max, ∞→mean)
--aggregator softmax --softmax_temp 1.0
```

## Comprehensive Experiment

Run the full experiment grid (8 edge budgets × 10 aggregators = 120 circuits):

```bash
python experiments/comprehensive_experiment.py \
  --n_examples 200 \
  --device cuda \
  --seed 42 \
  --batch_size 25 \
  --output_dir outputs/comprehensive \
  --resume  # skip already-completed phases
```

Phases:
1. **Score** (~12s) — EAP-IG per corruption, saves `scores.pt`
2. **Build** (~13s) — Aggregate + select for all (budget, aggregator) pairs
3. **Evaluate** (~26min) — Evaluate all 120 circuits under all corruptions

### Analyze Results

Generate figures and CSV tables:

```bash
python experiments/analyze_results.py \
  --input_dir outputs/comprehensive \
  --output_dir outputs/comprehensive/figures
```

Produces 9 figures (PDF) and 4 tables (CSV):
- Worst-case vs edge budget
- Aggregator spectrum (max → mean)
- Per-corruption heatmap
- Top-20 edge comparison
- And more

## Mixed Corruption Baseline

Compare DRO against the standard "mixed corruption" practice:

```bash
python experiments/mixed_corruption_experiment.py \
  --n_examples 200 \
  --device cuda \
  --output_dir outputs/mixed_corruption
```

## Tests

```bash
python -m pytest tests/ -v
```

30 unit tests covering aggregation, config, data loading, and score storage.

## FAQ

**Q: How long does a single run take?**
A: ~3 minutes on an RTX 5090 with n_examples=200, n_edges=200. The comprehensive experiment (120 circuits) takes ~30 minutes.

**Q: Can I use a different model?**
A: Currently only GPT-2 small is supported via the IOI task. The architecture supports extending to other TransformerLens-compatible models by implementing a new task class.

**Q: What if I don't have a GPU?**
A: Add `--device cpu`. It works but is significantly slower.
