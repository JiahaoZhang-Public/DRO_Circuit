# Architecture

## Pipeline Overview

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  IOITask     │───▶│ MultiCorrupt│───▶│PerCorruption │───▶│ ScoreStore   │
│  load model  │    │ Dataset     │    │ Scorer       │    │ (K,fwd,bwd)  │
│  build data  │    │ (N × K)     │    │ (EAP-IG ×K)  │    │              │
└──────────────┘    └─────────────┘    └──────────────┘    └──────┬───────┘
                                                                  │
                                                                  ▼
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Evaluation  │◀───│ Graph with  │◀───│  Top-n /     │◀───│ DRO          │
│  Robust      │    │ in_graph    │    │  Greedy      │    │ Aggregator   │
│  Evaluator   │    │ mask        │    │  Selection   │    │ Max/CVaR/Smx │
└──────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
```

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **Task** | `tasks/ioi.py` | Load model, build dataset, define metrics |
| **Corruption** | `corruption/ioi.py` | Generate $K$ corrupted variants of clean inputs |
| **Data** | `data/multi_corrupt_dataset.py` | Store $N$ clean $\times$ $K$ corrupted input pairs |
| **EAP Adapter** | `data/eap_adapter.py` | Convert MultiCorruptDataset to EAP-IG DataLoader format |
| **Scorer** | `scoring/per_corruption_scorer.py` | Run EAP-IG attribution $K$ times (once per corruption) |
| **Score Store** | `scoring/score_store.py` | Tensor storage for scores, shape $(K, n_\text{fwd}, n_\text{bwd})$ |
| **Aggregator** | `aggregation/aggregators.py` | Combine $K$ score tensors into one via DRO rule |
| **Pipeline** | `selection/pipeline.py` | Orchestrate Score → Aggregate → Select |
| **Evaluator** | `evaluation/robust_evaluator.py` | Evaluate circuit under all $K$ corruptions |
| **Metrics** | `evaluation/metrics.py` | logit_diff, logit_diff_loss, KL divergence |
| **Config** | `config.py` | Dataclass hierarchy for all hyperparameters |
| **CLI** | `scripts/run.py` | Command-line entry point |

## Data Flow

### 1. Dataset Construction

`IOITask.build_dataset()` creates a `MultiCorruptDataset` ($N$ samples $\times$ $K$ corruptions) containing clean inputs and $K$ corrupted variants per input, plus labels. See [IOI Task: Corruption Families](#ioi-task-corruption-families) for corruption details.

### 2. Scoring

`PerCorruptionScorer` iterates over each corruption family:

```python
for corruption_name in dataset.corruption_names:
    dataloader = make_eap_dataloader(dataset, corruption_name, batch_size)
    graph = Graph.from_model(model)
    attribute(model, graph, dataloader, metric, method="EAP-IG-inputs")
    score_store.set_scores(corruption_name, graph.scores)
```

Each call to `attribute()` runs a single forward + backward pass through the full model, computing edge importance scores for all ~32,000 edges simultaneously.

### 3. DRO Aggregation

Given per-corruption scores $s_e^{(k)}$ for edge $e$ under corruption $k \in \{1, \dots, K\}$:

**Max** (worst-case):

$$
s_e = \max_{k} \left| s_e^{(k)} \right|
$$

An edge is important if it matters under ANY corruption.

**CVaR($\alpha$)** (tail-risk):

$$
s_e = \frac{1}{\lceil \alpha K \rceil} \sum_{\text{top-}\lceil \alpha K \rceil} \left| s_e^{(k)} \right|
$$

Average the top-$\lceil \alpha K \rceil$ corruption scores. $\alpha \to 0$: max, $\alpha = 1$: mean.

**Softmax($\tau$)** (differentiable):

$$
s_e = \sum_{k} \operatorname{softmax}\!\left(\frac{|s^{(k)}|}{\tau}\right)_k \cdot \left| s_e^{(k)} \right|
$$

Soft attention over corruptions. $\tau \to 0$: max, $\tau \to \infty$: mean.

All three interpolate between **worst-case** (max) and **average-case** (mean).

### 4. Selection

After aggregation, edges are ranked by their aggregated score. Two methods:

- **Top-n**: Select the n highest-scoring edges directly
- **Greedy**: Iteratively add the edge that most improves the metric

The result is a `Graph` with an `in_graph` boolean mask marking which edges are in the circuit.

### 5. Evaluation

`evaluate_robust()` uses **activation patching** to test the circuit:

- For edges **in** the circuit: keep clean activations
- For edges **not in** the circuit: replace with corrupted activations
- Measure the logit difference between correct and incorrect answers

This is repeated for each corruption variant, producing per-corruption faithfulness scores. The key metrics are:

| Metric | Definition |
|--------|------------|
| **Faithfulness %** | $\frac{\mathcal{L}_{\text{circuit}}}{\mathcal{L}_{\text{baseline}}} \times 100\%$ |
| **Worst-case** | $\min_{k} \text{Faithfulness}(c_k)$ |
| **Mean** | $\frac{1}{K}\sum_{k} \text{Faithfulness}(c_k)$ |
| **Gap** | $\text{Worst} - \text{Best}$ (lower is more uniform) |

## Configuration Hierarchy

```
ExperimentConfig
├── ModelConfig         (model name, device, dtype)
├── CorruptionConfig    (which corruption families to use)
├── ScoringConfig       (EAP method, IG steps, batch size)
├── DROConfig           (aggregator type, CVaR α, Softmax τ)
├── SelectionConfig     (n_edges, topn/greedy, absolute)
├── EvalConfig          (intervention type, batch size)
├── task                (str: "ioi")
├── n_examples          (int)
├── seed                (int)
└── output_dir          (str)
```

Load from YAML:
```python
config = ExperimentConfig.from_yaml("configs/ioi.yaml")
```

## IOI Task: Corruption Families

The **Indirect Object Identification (IOI)** task tests whether a model can predict the correct indirect object in sentences like:

> "When **Mary** and **John** went to the store, **John** gave a drink to **\_\_\_**"

The model should predict **Mary** (the indirect object, IO) rather than **John** (the subject, S). The sentence has four named positions:

| Position | Notation | Example | Role |
|----------|----------|---------|------|
| First name | **IO** | Mary | Indirect object (correct answer) |
| Second name | **S1** | John | Subject, first occurrence |
| Repeated name | **S2** | John | Subject, duplicate (= S1) |
| Answer | — | Mary | Token to predict |

### Corruption design principle

Each corruption type systematically alters one or more name positions. The clean input and corrupted input are fed together during activation patching: the model processes the clean input, but edges **outside** the circuit receive activations from the corrupted run. A corruption that changes a specific name position tests whether the circuit correctly handles information flow through that position.

### The 5 corruption families ($K=5$)

Concrete example — clean sentence:

> "When **Mary** and **John** went to the store, **John** gave a drink to"

| Corruption | What changes | Corrupted sentence | Why it matters |
|------------|-------------|-------------------|----------------|
| **S2_IO** | Swap S2 ↔ IO | "When **John** and **Mary** went to the store, **Mary** gave a drink to" | Tests if circuit distinguishes subject from IO via position, not just name identity |
| **IO_RAND** | IO → random name | "When **Alice** and **John** went to the store, **John** gave a drink to" | Tests if circuit tracks IO identity; with IO replaced, circuit must rely on stored IO information |
| **S_RAND** | S1, S2 → random name | "When **Mary** and **Bob** went to the store, **Bob** gave a drink to" | Tests if circuit uses subject identity; subject path disrupted but IO path intact |
| **S1_RAND** | S1 → random name | "When **Mary** and **Alice** went to the store, **John** gave a drink to" | Tests if circuit relies on S1-S2 match for duplicate token detection |
| **IO_S1** | Swap IO ↔ S1 | "When **John** and **Mary** went to the store, **John** gave a drink to" | Tests if circuit distinguishes IO from S1 when both appear exactly once |

### Implementation

Each corruption wraps ACDC's `IOIDataset.gen_flipped_prompts(flip_type, seed)`:

```python
# dro_circuit/corruption/ioi.py

# Pre-defined corruption families
IOI_CORRUPTIONS = {
    "S2_IO":   IOICorruptionFamily(("S2", "IO"), seed=42),
    "IO_RAND": IOICorruptionFamily(("IO", "RAND"), seed=42),
    "S_RAND":  IOICorruptionFamily(("S", "RAND"), seed=42),
    "S1_RAND": IOICorruptionFamily(("S1", "RAND"), seed=42),
    "IO_S1":   IOICorruptionFamily(("IO", "S1"), seed=42),
}

# ACDC default: triple flip (used by standard ACDC experiments)
ACDC_DEFAULT_CORRUPTION = IOIComposedCorruption(
    [("IO", "RAND"), ("S", "RAND"), ("S1", "RAND")], seed=42
)
```

`IOICorruptionFamily` applies a single flip; `IOIComposedCorruption` applies a sequence of flips.

### Which corruption is hardest?

Different corruptions stress different sub-circuits. **IO_RAND** tends to be the hardest for naive circuits because it directly removes the IO name, forcing the circuit to rely entirely on stored IO information from earlier layers. A circuit discovered under only S2_IO corruption may fail catastrophically under IO_RAND — this is exactly the failure mode that DRO aggregation addresses.

### Adding corruptions for a new task

To define corruption families for a task other than IOI, implement `CorruptionFamily`:

```python
class MyCorruption(CorruptionFamily):
    def name(self) -> str:
        return "my_corruption_name"

    def generate(self, clean_dataset, **kwargs) -> CorruptionResult:
        corrupted = [apply_corruption(s) for s in clean_dataset.sentences]
        return CorruptionResult(corrupted, self.name())
```

The key design principle: each corruption should target a **distinct information pathway** in the circuit, so that DRO aggregation over them finds edges robust to all pathways.

## Extending to New Tasks

To add a new task (e.g., Greater-Than, Docstring):

### 1. Define corruption families

```python
# dro_circuit/corruption/my_task.py
from dro_circuit.corruption.base import CorruptionFamily, CorruptionResult

class MyCorruption(CorruptionFamily):
    def name(self) -> str:
        return "my_corruption"

    def generate(self, clean_dataset, **kwargs) -> CorruptionResult:
        corrupted = [corrupt(s) for s in clean_dataset]
        return CorruptionResult(corrupted, self.name())
```

### 2. Define the task

```python
# dro_circuit/tasks/my_task.py
class MyTask:
    def load_model(self) -> HookedTransformer:
        model = HookedTransformer.from_pretrained(...)
        model.set_use_attn_result(True)
        model.set_use_split_qkv_input(True)
        model.set_use_hook_mlp_in(True)
        return model

    def build_dataset(self, tokenizer) -> Tuple[MultiCorruptDataset, Any]:
        # Build clean data + K corruption variants
        # Return (MultiCorruptDataset, raw_dataset)
        ...

    def get_scoring_metric(self) -> Callable:
        # Return metric(logits, clean_logits, input_lengths, labels) -> scalar
        ...

    def get_eval_metric(self) -> Callable:
        return self.get_scoring_metric()
```

### 3. Register in the CLI

Add a new choice in `scripts/run.py`:

```python
parser.add_argument("--task", choices=["ioi", "my_task"])
```

The rest of the pipeline (scoring, aggregation, selection, evaluation) works unchanged.

## Vendor Dependencies

| Library | Path | Used For |
|---------|------|----------|
| **EAP-IG** | `vendor/EAP-IG/src/eap/` | `attribute()` for edge scoring, `Graph` for circuit representation, `evaluate_graph()` for activation patching |
| **ACDC** | `vendor/Automatic-Circuit-Discovery/` | `IOIDataset` for generating IOI data, `gen_flipped_prompts()` for corruption variants |

Key EAP-IG interfaces:
- `Graph.from_model(model)` — create edge graph from HookedTransformer
- `attribute(model, graph, dataloader, metric, method)` — compute edge scores
- `evaluate_graph(model, graph, dataloader, metric)` — activation patching evaluation
- `graph.apply_topn(n, absolute)` — select top-n edges
- `graph.in_graph` — boolean mask of selected edges
- `graph.scores` — edge score tensor (n_forward, n_backward)

## Future: Pluggable Scoring & Evaluation

The current pipeline is tightly coupled to EAP-IG at the scoring and evaluation layers. To support alternative circuit discovery methods (e.g., ACDC iterative pruning, gradient saliency, activation patching variants), the following abstractions are planned:

### Planned abstractions

```
Scorer (ABC)                    Circuit (ABC)                  Evaluator (ABC)
├── EAPScorer (current)         ├── EAPGraph (current Graph)   ├── EAPEvaluator (current)
├── ACDCScorer                  ├── ACDCCircuit                ├── ACDCEvaluator
└── CustomScorer                └── CustomCircuit              └── CustomEvaluator
```

**`Scorer`** — abstract interface for edge scoring:

```python
class Scorer(ABC):
    @abstractmethod
    def score_all_corruptions(self, model, dataset, metric) -> ScoreStore:
        """Score all edges under each corruption. Returns (K, n_fwd, n_bwd) scores."""
        ...
```

**`Circuit`** — abstract circuit representation:

```python
class Circuit(ABC):
    @abstractmethod
    def select(self, scores: Tensor, n_edges: int, method: str) -> None:
        """Select top-n edges from aggregated scores."""
        ...

    @property
    @abstractmethod
    def in_graph(self) -> BoolTensor:
        """Boolean mask of edges in the circuit."""
        ...
```

**`Evaluator`** — abstract evaluation:

```python
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model, circuit, dataloader, metric) -> Tensor:
        """Evaluate circuit faithfulness via activation patching."""
        ...
```

### Current coupling points

| Component | Coupled to | What would change |
|-----------|-----------|-------------------|
| `PerCorruptionScorer` | `Graph`, `attribute()` | Wrap in `EAPScorer(Scorer)` |
| `DROPipeline.run()` | `Graph.from_model()`, `apply_topn()` | Accept `Scorer` + `Circuit` factory |
| `robust_evaluator.py` | `evaluate_graph()`, `evaluate_baseline()` | Wrap in `EAPEvaluator(Evaluator)` |
| `eap_adapter.py` | EAP DataLoader format | Move into `EAPScorer` as internal detail |
| `ScoringConfig` | EAP method enum | Generalize to `method: str` + `method_kwargs: dict` |

### What stays the same

These components are already method-agnostic and require no changes:

- `DROAggregator` (Max / CVaR / Softmax) — pure tensor operations
- `ScoreStore` — generic $(K, n_\text{fwd}, n_\text{bwd})$ storage
- `MultiCorruptDataset` — generic data structure
- `CorruptionFamily` — generic corruption interface
