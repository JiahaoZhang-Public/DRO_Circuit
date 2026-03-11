# DRO_Circuit

`dro_circuit` is a research codebase for robust circuit discovery in transformers.
The current implementation focuses on Indirect Object Identification (IOI) on GPT-2
and asks a simple question: if you score edges under several corruption families
instead of a single corruption, can you recover circuits with better worst-case
faithfulness?

The repository wraps two vendored codebases:

- `vendor/Automatic-Circuit-Discovery` for IOI data and corruption generation
- `vendor/EAP-IG` for edge attribution, graph construction, and circuit evaluation

## What is implemented

- `Plan A`: score each edge separately for each corruption, aggregate scores with a
  DRO-style rule (`max`, `cvar`, or `softmax`), then select a circuit with `topn`
  or `greedy`
- `Plan B`: an experimental learnable edge-mask pipeline with adversarial
  corruption weighting
- `Robust evaluation`: evaluate a discovered circuit under every corruption and
  report mean, worst-case, and gap

At the moment, the main supported task is `ioi`.

## Quick Start

Clone the repo with submodules, create an environment, and install the package:

```bash
git clone --recurse-submodules <repo-url>
cd dro_circuit
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Run the local test suite:

```bash
PYTHONPATH=. pytest -q tests
```

The explicit `tests` target matters because a plain `pytest` will also collect
vendor test suites, which require extra dependencies and are not part of this
repo's default smoke test.

## First Run

Run the main Plan A pipeline from the provided config:

```bash
python -m dro_circuit.scripts.run_plan_a --config configs/plan_a_ioi.yaml
```

Or run it directly from CLI flags:

```bash
python -m dro_circuit.scripts.run_plan_a \
  --task ioi \
  --n_examples 100 \
  --n_edges 200 \
  --aggregator max \
  --method EAP-IG-inputs \
  --selection topn \
  --device cuda
```

This will:

1. Load GPT-2 through TransformerLens
2. Build an IOI dataset plus several corruption variants
3. Score edges independently per corruption
4. Aggregate scores across corruptions
5. Select a circuit
6. Evaluate that circuit under each corruption
7. Save `circuit.pt`, `scores.pt`, `results.json`, and `config.json`

By default, outputs go to `outputs/` or the configured `output_dir`.

## Main Entry Points

- `python -m dro_circuit.scripts.run_plan_a`: robust score-aggregate-select pipeline
- `python -m dro_circuit.scripts.run_plan_b`: experimental learnable-mask pipeline
- `python experiments/compare_naive_vs_dro.py`: compare single-corruption circuits
  against DRO-aggregated circuits

## Repository Map

The project is much smaller than the original cookiecutter layout suggests. The
important directories are:

```text
dro_circuit/
  aggregation/   DRO aggregators over corruption-specific edge scores
  corruption/    Task-specific corruption family wrappers
  data/          Multi-corruption dataset abstractions and EAP adapters
  evaluation/    Metrics and robust evaluation helpers
  scoring/       Per-corruption attribution and score storage
  selection/     Plan A and Plan B circuit selection pipelines
  tasks/         Task-specific setup; currently IOI
  scripts/       CLI entry points

configs/
  plan_a_ioi.yaml    Example experiment config

experiments/
  compare_naive_vs_dro.py

vendor/
  Automatic-Circuit-Discovery/
  EAP-IG/
```

## How the Code Fits Together

For the main IOI workflow:

- `dro_circuit.tasks.ioi.IOITask` loads GPT-2 and builds clean/corrupted IOI data
- `dro_circuit.scoring.PerCorruptionScorer` runs EAP or EAP-IG independently for
  each corruption
- `dro_circuit.aggregation` collapses per-corruption edge scores into one robust
  score tensor
- `dro_circuit.selection.plan_a.PlanAPipeline` writes those scores into an EAP
  graph and selects the final circuit
- `dro_circuit.evaluation.robust_evaluator` evaluates the circuit under every
  corruption and summarizes worst-case behavior

## Current Caveats

- Documentation outside this README and the MkDocs onboarding pages is still
  sparse
- `Plan B` is explicitly experimental; the implementation comments note that it
  is not fully differentiable through the model/evaluation path
- The current task support is effectively IOI-only, even though some abstractions
  are generic

## Useful Commands

```bash
make requirements   # pip install -e .
make test           # python -m pytest tests
make lint           # ruff format --check && ruff check
make format         # autoformat with ruff
```

## Docs

MkDocs content lives under `docs/docs/`.

```bash
mkdocs serve -f docs/mkdocs.yml
```
