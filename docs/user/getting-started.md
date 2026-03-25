# Getting Started

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
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
| **EAP-IG** | `vendor/EAP-IG/` | Edge Attribution Patching — single-pass edge scoring (we use the EAP method) |
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
2. Generate 100 IOI examples with $K = 5$ corruption variants each
3. Score all edges under each corruption via EAP
4. Aggregate scores with Max (worst-case Group DRO) rule
5. Select top 200 edges as the sparse circuit
6. Evaluate circuit faithfulness under all corruptions

Output files in `outputs/first_run/`:

| File | Content |
|------|---------|
| `circuit.pt` | Circuit graph with `in_graph` mask |
| `scores.pt` | Per-corruption edge scores $(K \times n_\text{fwd} \times n_\text{bwd})$ |
| `results.json` | Evaluation metrics (worst, mean, gap, per-corruption) |
| `config.json` | Full experiment configuration |

## Aggregator Options

```bash
# Mean (ERM — average-case baseline)
--aggregator mean

# Max (Group DRO — pure worst-case)
--aggregator max

# Local DRO (per-example worst-case; requires per-example scoring)
--aggregator local_dro

# CVaR — alpha controls tail risk (0=max, 1=mean)
--aggregator cvar --cvar_alpha 0.5

# Softmax — tau controls temperature (0->max, inf->mean)
--aggregator softmax --softmax_temp 1.0
```

## Comprehensive Experiment

Run the full ERM vs DRO experiment grid:

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
1. **Phase 1: Score** — EAP per corruption (aggregated), saves `scores.pt`
2. **Phase 1b: Score per-example** — EAP per-example mode, saves `scores_per_example.pt`
3. **Phase 2: Build** — Aggregate + select for all (budget, aggregator) pairs, including ERM (mean), Local DRO, and Group DRO variants (max, CVaR, softmax)
4. **Phase 3: Evaluate** — Evaluate all circuits under all corruptions

### Analyze Results

Generate figures and CSV tables:

```bash
python experiments/analyze_results.py \
  --input_dir outputs/comprehensive \
  --output_dir outputs/comprehensive/figures
```

## Tests

```bash
python -m pytest tests/ -v
```

30 unit tests covering aggregation, config, data loading, and score storage.

## FAQ

**Q: How long does a single run take?**
A: ~3 minutes on an RTX 5090 with n_examples=200, n_edges=200. The comprehensive experiment takes ~30 minutes.

**Q: Can I use a different model?**
A: Currently only GPT-2 small is supported via the IOI task. The architecture supports extending to other TransformerLens-compatible models by implementing a new task class.

**Q: What if I don't have a GPU?**
A: Add `--device cpu`. It works but is significantly slower.
