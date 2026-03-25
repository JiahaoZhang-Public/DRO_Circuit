# DRO Circuit Discovery

**Distributionally Robust Circuit Discovery in Transformers**

Standard circuit discovery methods score edges under a single corruption type, producing circuits that are faithful on average but may fail under certain interventions. DRO Circuit Discovery reframes this as an **ERM vs DRO** comparison — scoring each edge under multiple corruption families and aggregating with a robust rule to find circuits that are faithful under the **worst-case** corruption.

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
        │ EAP      │     │ EAP      │  ...  │ EAP      │
        │ T₁       │     │ T₂       │       │ Tₖ       │
        └────┬─────┘     └────┬─────┘       └────┬─────┘
             │                │                   │
             ▼                ▼                   ▼
        scores(T₁)       scores(T₂)         scores(Tₖ)
             │                │                   │
             └────────────────┼───────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Aggregation       │
                    │  ERM / DRO         │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Top-B Selection   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Sparse Circuit    │
                    └────────────────────┘
```

**Pipeline:**
1. **Score** — For each corruption family $T_k$, run EAP on all $N$ examples → per-edge scores $s(e; k)$
2. **Aggregate** — Apply an ERM or DRO rule over the $K$ corruption scores:
   - **ERM (Mean)**: $S(e) = \frac{1}{K} \sum_k |\bar{s}(e; k)|$ (average-case)
   - **Local DRO**: $S(e) = \frac{1}{N} \sum_i \max_k |s(e; x_i, \tilde{x}_{ik})|$ (per-example worst-case)
   - **Max (Group DRO)**: $S(e) = \max_k |\bar{s}(e; k)|$ (worst corruption family)
   - **CVaR($\alpha$)**: average of top-$\lceil \alpha K \rceil$ scores (tail-risk)
   - **Softmax($\tau$)**: temperature-weighted sum (smooth max)
3. **Select** — Take top-$B$ edges by aggregated score → sparse circuit
4. **Evaluate** — Normalized faithfulness under all corruptions (mean, worst-group, per-example worst-case)

## Installation

```bash
git clone --recursive https://github.com/JiahaoZhang-Public/DRO_Circuit.git
cd DRO_Circuit
pip install -e .
```

The `--recursive` flag pulls the two vendor submodules:
- `vendor/EAP-IG` — Edge Attribution Patching (scoring engine; we use the EAP method)
- `vendor/Automatic-Circuit-Discovery` — ACDC (IOI task and dataset)

## Quick Start

**Single experiment** — find a 200-edge circuit using Max (Group DRO) aggregation:

```bash
python -m dro_circuit.scripts.run \
  --task ioi --n_edges 200 --aggregator max \
  --device cuda --output_dir outputs/single_run
```

**Comprehensive experiment** — ERM vs DRO comparison across multiple budgets and aggregators:

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

## Aggregator Options

```bash
# ERM — average-case baseline
--aggregator mean

# Local DRO — per-example worst-case (requires per-example scoring)
--aggregator local_dro

# Max — Group DRO, pure worst-case corruption family
--aggregator max

# CVaR — tail risk, alpha controls interpolation (0→max, 1→mean)
--aggregator cvar --cvar_alpha 0.5

# Softmax — smooth max, tau controls temperature (0→max, ∞→mean)
--aggregator softmax --softmax_temp 1.0
```

## Project Structure

```
dro_circuit/
├── dro_circuit/                     # Main package
│   ├── config.py                    # Experiment configuration dataclasses
│   ├── tasks/ioi.py                 # IOI task: model, dataset, metrics
│   ├── corruption/
│   │   ├── base.py                  # CorruptionFamily abstract interface
│   │   └── ioi.py                   # 5 IOI corruption families
│   ├── data/
│   │   ├── multi_corrupt_dataset.py # Dataset pairing clean inputs with K corruptions
│   │   └── eap_adapter.py           # Adapter for EAP DataLoader format
│   ├── scoring/
│   │   ├── per_corruption_scorer.py # EAP attribution per corruption (aggregated + per-example)
│   │   └── score_store.py           # ScoreStore (K,F,B) + PerExampleScoreStore (K,N,F,B)
│   ├── aggregation/
│   │   └── aggregators.py           # Mean / LocalDRO / Max / CVaR / Softmax aggregators
│   ├── selection/
│   │   └── pipeline.py              # DROPipeline: Score → Aggregate → Select
│   ├── evaluation/
│   │   ├── metrics.py               # logit_diff, KL divergence
│   │   └── robust_evaluator.py      # Raw + normalized faithfulness evaluation
│   └── scripts/
│       └── run.py                   # CLI entry point
├── experiments/
│   ├── comprehensive_experiment.py  # Full ERM vs DRO experiment grid
│   ├── analyze_results.py           # Generate figures and tables
│   └── visualize_circuits.py        # Graphviz circuit rendering
├── tests/                           # Unit tests (pytest)
├── vendor/
│   ├── EAP-IG/                      # Edge Attribution Patching
│   └── Automatic-Circuit-Discovery/ # ACDC (IOI dataset)
└── docs/                            # Documentation
    ├── user/                        # User-facing docs (architecture, getting-started)
    └── research/references/         # Formal problem & experiment setup
```

## Documentation

See `docs/` for detailed documentation:

- [Getting Started](docs/user/getting-started.md) — installation, first run, experiment workflow
- [Architecture](docs/user/architecture.md) — pipeline design, module responsibilities, data flow, IOI corruption families
- [Problem Setup](docs/research/references/problem-setup.md) — grouped intervention dataset, faithfulness loss, ERM/DRO objectives
- [Experiment Setup](docs/research/references/experiment-setup.md) — EAP scoring, aggregation formulas, evaluation protocol

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
