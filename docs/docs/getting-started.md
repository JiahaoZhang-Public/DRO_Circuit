# Getting Started

This page is the shortest path from clone to a real experiment run.

## What this repo does

The codebase studies robust circuit discovery: discover a circuit that remains
faithful across several corruption families instead of overfitting to one
particular clean-corrupted pair.

The current task is IOI on GPT-2. The main production path is `Plan A`:

1. generate clean IOI prompts plus several corruption variants
2. score edges independently for each corruption
3. aggregate those scores with a DRO-style rule
4. select a circuit
5. evaluate the resulting circuit under each corruption

`Plan B` exists as an experimental learnable-mask path, but the comments in the
implementation already note that it is structurally incomplete as a fully
differentiable method.

## Prerequisites

- Python 3.10 or newer
- Git submodules enabled
- A working PyTorch environment
- GPU access if you want practical run times for GPT-2 experiments

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd dro_circuit
```

If the repo is already cloned:

```bash
git submodule update --init --recursive
```

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project code imports the vendored research libraries by path, so the
submodules need to exist locally even after `pip install -e .`.

## Sanity Check

Run the local tests:

```bash
PYTHONPATH=. pytest -q tests
```

Use the explicit `tests` path. Running plain `pytest` will also collect test
suites inside `vendor/`, which are not part of the default project smoke test
and may fail on missing extras such as `pygraphviz`.

## First Experiment

The fastest way to run the main pipeline is the provided config:

```bash
python -m dro_circuit.scripts.run_plan_a --config configs/plan_a_ioi.yaml
```

That config currently uses:

- task: `ioi`
- model: `gpt2`
- corruptions: `S2_IO`, `IO_RAND`, `S_RAND`
- scoring: `EAP-IG-inputs`
- aggregator: `max`
- selection: top 200 edges with `topn`

Outputs are written under the configured `output_dir` and include:

- `circuit.pt`
- `scores.pt`
- `results.json`
- `config.json`

## CLI Variants

Run Plan A directly from flags:

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

Run the experimental Plan B pipeline:

```bash
python -m dro_circuit.scripts.run_plan_b \
  --task ioi \
  --n_examples 100 \
  --n_edges 200 \
  --device cuda
```

Compare naive single-corruption discovery against DRO aggregation:

```bash
python experiments/compare_naive_vs_dro.py --device cpu --n_examples 50 --n_edges 100
```

## Repository Layout

The directories that matter most for onboarding are:

```text
dro_circuit/
  tasks/ioi.py                  Task-specific model and dataset setup
  corruption/ioi.py             IOI corruption family wrappers
  data/multi_corrupt_dataset.py Clean-plus-many-corruptions dataset abstraction
  data/eap_adapter.py           Adapters into EAP-IG dataloaders
  scoring/per_corruption_scorer.py
  aggregation/aggregators.py
  selection/plan_a.py
  selection/plan_b.py
  evaluation/robust_evaluator.py
  scripts/run_plan_a.py
  scripts/run_plan_b.py

configs/plan_a_ioi.yaml         Example config used by the main script
experiments/compare_naive_vs_dro.py
vendor/                         ACDC and EAP-IG submodules
tests/                          Local unit tests for the wrapper code
```

## Mental Model

If you are trying to extend the repo, follow this order:

1. start in `tasks/` to see how a task loads a model and produces labels
2. read `corruption/` and `data/` to understand the clean/corrupt dataset shape
3. read `scoring/` to see how edge scores are produced per corruption
4. read `aggregation/` and `selection/plan_a.py` for the main robust pipeline
5. read `evaluation/` to understand the metrics used for comparison

## Common Pitfalls

- `README.md` and some older docs were originally cookiecutter placeholders; the
  code is a better source of truth
- the package currently assumes IOI-specific vendor code exists
- plain `pytest` will pull in vendor tests
- `Plan B` should be treated as exploratory code, not as a validated baseline
