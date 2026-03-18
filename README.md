# DRO Circuit Discovery

**Distributionally Robust Circuit Discovery in Transformers**

Standard circuit discovery methods score edges under a single corruption, making them brittle to the choice of corruption type. DRO Circuit Discovery scores each edge independently under multiple corruption families, then aggregates with a DRO-style rule to find circuits that are faithful under the **worst-case** corruption.

## Method

```
                        ┌─────────────────┐
                        │  Clean + K      │
                        │  Corruptions    │
                        └────────┬────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                  ▼
        ┌──────────┐     ┌──────────┐       ┌──────────┐
        │ EAP-IG   │     │ EAP-IG   │  ...  │ EAP-IG   │
        │ c₁       │     │ c₂       │       │ cₖ       │
        └────┬─────┘     └────┬─────┘       └────┬─────┘
             │                │                   │
             ▼                ▼                   ▼
        scores(c₁)       scores(c₂)         scores(cₖ)
             │                │                   │
             └────────────────┼───────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  DRO Aggregation   │
                    │  Max / CVaR / Smx  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Top-n Selection   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Sparse Circuit    │
                    └────────────────────┘
```

**Pipeline:**
1. **Score** — For each corruption family $c_k$, run EAP-IG attribution with all $N$ samples → per-edge scores $s_e^{(k)}$
2. **Aggregate** — Apply a DRO rule over the $K$ corruption scores:
   - **Max**: $s_e = \max_k |s_e^{(k)}|$ (pure worst-case)
   - **CVaR($\alpha$)**: average of top-$\lceil \alpha K \rceil$ scores (tail-risk, $\alpha \to 0$: max, $\alpha = 1$: mean)
   - **Softmax($\tau$)**: temperature-weighted sum ($\tau \to 0$: max, $\tau \to \infty$: mean)
3. **Select** — Take top-$n$ edges by aggregated score → sparse circuit

## Installation

```bash
git clone --recursive https://github.com/JiahaoZhang-Public/DRO_Circuit.git
cd DRO_Circuit
pip install -e .
```

The `--recursive` flag pulls the two vendor submodules:
- `vendor/EAP-IG` — Edge Attribution Patching with Integrated Gradients (scoring engine)
- `vendor/Automatic-Circuit-Discovery` — ACDC (IOI task and dataset)

## Quick Start

**Single experiment** — find a 200-edge circuit using CVaR(0.5) aggregation:

```bash
python -m dro_circuit.scripts.run \
  --task ioi --n_edges 200 --aggregator cvar --cvar_alpha 0.5 \
  --device cuda --output_dir outputs/single_run
```

**Comprehensive experiment** — sweep 8 budgets × 10 aggregators (120 circuits):

```bash
python experiments/comprehensive_experiment.py \
  --n_examples 200 --device cuda --seed 42 \
  --output_dir outputs/comprehensive
```

**Analyze results** — generate figures and tables:

```bash
python experiments/analyze_results.py \
  --input_dir outputs/comprehensive \
  --output_dir outputs/comprehensive/figures
```

## Project Structure

```
dro_circuit/
├── dro_circuit/                  # Main package
│   ├── config.py                 # Experiment configuration dataclasses
│   ├── tasks/ioi.py              # IOI task: model, dataset, metrics
│   ├── corruption/
│   │   ├── base.py               # CorruptionFamily abstract interface
│   │   └── ioi.py                # 5 IOI corruption families
│   ├── data/
│   │   ├── multi_corrupt_dataset.py  # Dataset pairing clean inputs with K corruptions
│   │   └── eap_adapter.py           # Adapter for EAP-IG DataLoader format
│   ├── scoring/
│   │   ├── per_corruption_scorer.py  # EAP-IG attribution per corruption
│   │   └── score_store.py            # Score tensor storage (K, n_fwd, n_bwd)
│   ├── aggregation/
│   │   └── aggregators.py        # Max / CVaR / Softmax DRO aggregators
│   ├── selection/
│   │   └── pipeline.py           # DROPipeline: Score → Aggregate → Select
│   ├── evaluation/
│   │   ├── metrics.py            # logit_diff, KL divergence
│   │   └── robust_evaluator.py   # Evaluate circuit under all corruptions
│   └── scripts/
│       └── run.py                # CLI entry point
├── experiments/
│   ├── comprehensive_experiment.py   # Full grid experiment (120 circuits)
│   ├── compare_naive_vs_dro.py       # Quick comparison demo
│   ├── mixed_corruption_experiment.py # Mixed vs DRO baseline
│   ├── analyze_results.py            # Generate figures and tables
│   └── visualize_circuits.py         # Graphviz circuit rendering
├── tests/                        # Unit tests (pytest)
├── vendor/
│   ├── EAP-IG/                   # Edge Attribution Patching
│   └── Automatic-Circuit-Discovery/  # ACDC (IOI dataset)
├── docs/                         # Technical documentation (mkdocs)
└── presentation.html             # Interactive 17-slide presentation
```

## Documentation

See `docs/` for detailed documentation:
- [Architecture](docs/docs/architecture.md) — pipeline design, module responsibilities, data flow
- [Getting Started](docs/docs/getting-started.md) — installation, first run, experiment workflow

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT

