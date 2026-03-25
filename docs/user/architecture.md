# Architecture

## Pipeline Overview

```
┌──────────────┐    ┌─────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  IOITask     │───>│ MultiCorrupt│───>│  PerCorruption       │───>│ ScoreStore (K,F,B)   │
│  load model  │    │ Dataset     │    │  Scorer              │    │ PerExampleScoreStore  │
│  build data  │    │ (N x K)     │    │  (EAP x K)           │    │ (K,N,F,B)            │
└──────────────┘    └─────────────┘    └──────────────────────┘    └──────────┬───────────┘
                                                                               │
                                                                               v
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────────────────────┐
│  Normalized  │<───│ Graph with  │<───│  Top-B       │<───│ Aggregator                    │
│  Faithfulness│    │ in_graph    │    │  Selection   │    │ Mean/LocalDRO/Max/CVaR/Softmax│
│  Evaluation  │    │ mask        │    │              │    │                               │
└──────────────┘    └─────────────┘    └──────────────┘    └───────────────────────────────┘
```

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **Task** | `tasks/ioi.py` | Load model, build dataset, define metrics |
| **Corruption** | `corruption/ioi.py` | Generate $K$ corrupted variants of clean inputs |
| **Data** | `data/multi_corrupt_dataset.py` | Store $N$ clean $\times$ $K$ corrupted input pairs |
| **EAP Adapter** | `data/eap_adapter.py` | Convert MultiCorruptDataset to EAP DataLoader format |
| **Scorer** | `scoring/per_corruption_scorer.py` | Run EAP attribution $K$ times; supports aggregated and per-example modes |
| **Score Store** | `scoring/score_store.py` | `ScoreStore` $(K, F, B)$ and `PerExampleScoreStore` $(K, N, F, B)$ |
| **Aggregator** | `aggregation/aggregators.py` | Mean (ERM), LocalDRO, Max, CVaR, Softmax |
| **Pipeline** | `selection/pipeline.py` | Orchestrate Score -> Aggregate -> Select |
| **Evaluator** | `evaluation/robust_evaluator.py` | Raw metric evaluation and normalized faithfulness |
| **Metrics** | `evaluation/metrics.py` | logit_diff, logit_diff_loss, KL divergence |
| **Config** | `config.py` | Dataclass hierarchy for all hyperparameters |
| **CLI** | `scripts/run.py` | Command-line entry point |

## Data Flow

### 1. Dataset Construction

`IOITask.build_dataset()` creates a `MultiCorruptDataset` ($N$ samples $\times$ $K$ corruptions) containing clean inputs and $K$ corrupted variants per input, plus labels. See [IOI Task: Corruption Families](#ioi-task-corruption-families) for corruption details.

### 2. Scoring

`PerCorruptionScorer` supports two scoring modes:

**Aggregated mode** (`score_all_corruptions`) — returns `ScoreStore` of shape $(K, n_\text{fwd}, n_\text{bwd})$:

```python
for corruption_name in dataset.corruption_names:
    dataloader = make_eap_dataloader(dataset, corruption_name, batch_size)
    graph = Graph.from_model(model)
    attribute(model, graph, dataloader, metric, method="EAP")
    score_store.set_scores(corruption_name, graph.scores)  # (n_fwd, n_bwd)
```

**Per-example mode** (`score_all_corruptions_per_example`) — returns `PerExampleScoreStore` of shape $(K, N, n_\text{fwd}, n_\text{bwd})$:

```python
for corruption_name in dataset.corruption_names:
    dataloader = make_eap_dataloader(dataset, corruption_name, batch_size)
    graph = Graph.from_model(model)
    per_example_scores = attribute(model, graph, dataloader, metric,
                                   method="EAP", per_example=True)  # (N, n_fwd, n_bwd)
    store.set_scores(corruption_name, per_example_scores)
```

Per-example mode is needed for Local DRO aggregation. It uses a modified EAP backward hook that preserves the batch dimension in the einsum.

### 3. Aggregation

Given per-corruption scores, aggregators reduce to a single $(n_\text{fwd}, n_\text{bwd})$ tensor. Using notation from [Experiment Setup](../research/references/experiment-setup.md):

**Mean (ERM)** — average over corruption families (operates on aggregated scores):

$$S_{\text{ERM}}(e) = \frac{1}{K} \sum_{k=1}^{K} |\bar{s}(e;\, k)|$$

**Local DRO** — per-example worst-case (requires per-example scores):

$$S_{\text{DRO}}^{\text{local}}(e) = \frac{1}{N} \sum_{i=1}^{N} \max_{k \in [K]} |s(e;\, x_i, \tilde{x}_{ik})|$$

**Max (Group DRO)** — worst corruption family on average:

$$S_{\text{DRO}}^{\max}(e) = \max_{k \in [K]} |\bar{s}(e;\, k)|$$

**CVaR($\alpha$)** — tail-risk average of top-$\lceil \alpha K \rceil$ corruptions:

$$S_{\text{DRO}}^{\text{CVaR}}(e) = \frac{1}{\lceil \alpha K \rceil} \sum_{k \in \text{top-}\lceil \alpha K \rceil} |\bar{s}(e;\, k)|$$

$\alpha \to 0$: approaches max. $\alpha = 1$: equals mean.

**Softmax($\tau$)** — temperature-controlled weighted sum:

$$S_{\text{DRO}}^{\text{softmax}}(e) = \sum_{k} \frac{\exp(|\bar{s}_k| / \tau)}{\sum_{k'} \exp(|\bar{s}_{k'}| / \tau)} \cdot |\bar{s}(e;\, k)|$$

$\tau \to 0$: approaches max. $\tau \to \infty$: approaches mean.

All aggregators take absolute values by default, since EAP scores can be negative and we care about magnitude.

### 4. Selection

After aggregation, edges are ranked by their aggregated score. Two methods:

- **Top-B**: Select the $B$ highest-scoring edges directly
- **Greedy**: Iteratively add the edge that most improves the metric

The result is a `Graph` with an `in_graph` boolean mask marking which edges are in the circuit.

### 5. Evaluation

The evaluation layer provides two interfaces:

**Raw evaluation** (`evaluate_robust`) — returns per-corruption metric values (logit_diff_loss) for backward compatibility.

**Normalized faithfulness** (`evaluate_normalized_faithfulness`) — the primary evaluation metric, as defined in [Problem Setup](../research/references/problem-setup.md):

$$\operatorname{Faith}(c;\, x, \tilde{x}) = \frac{\hat{m} - b'}{b - b'}$$

where:
- $b = M_f(x)$: full model logit diff on clean input
- $b' = M_f(\tilde{x})$: full model logit diff on corrupted input
- $\hat{m} = M_{f_c}(x;\, \tilde{x})$: circuit-intervened model logit diff

$\operatorname{Faith} = 1$ means the circuit fully recovers the clean model's behavior; $\operatorname{Faith} = 0$ means no better than the corrupted baseline.

`compute_normalized_robust_metrics()` summarizes faithfulness into three metrics:

| Metric | Formula | Corresponds to |
|--------|---------|---------------|
| **Mean faithfulness** | mean over all $(i, k)$ pairs | $1 - R_{\text{ERM}}(c)$ |
| **Worst-group faithfulness** | min over $k$ of per-corruption means | $1 - R_{\text{DRO}}^{\text{group}}(c)$ |
| **Per-example worst faithfulness** | for each $i$, min over $k$, then mean over $i$ | $1 - R_{\text{DRO}}^{\text{local}}(c)$ |

## Configuration Hierarchy

```
ExperimentConfig
├── ModelConfig         (model name, device, dtype)
├── CorruptionConfig    (which corruption families to use)
├── ScoringConfig       (EAP method, IG steps, batch size)
├── DROConfig           (aggregator: max/mean/local_dro/cvar/softmax, alpha, temperature)
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
| **S2_IO** | Swap S2 <-> IO | "When **John** and **Mary** went to the store, **Mary** gave a drink to" | Tests if circuit distinguishes subject from IO via position, not just name identity |
| **IO_RAND** | IO -> random name | "When **Alice** and **John** went to the store, **John** gave a drink to" | Tests if circuit tracks IO identity; with IO replaced, circuit must rely on stored IO information |
| **S_RAND** | S1, S2 -> random name | "When **Mary** and **Bob** went to the store, **Bob** gave a drink to" | Tests if circuit uses subject identity; subject path disrupted but IO path intact |
| **S1_RAND** | S1 -> random name | "When **Mary** and **Alice** went to the store, **John** gave a drink to" | Tests if circuit relies on S1-S2 match for duplicate token detection |
| **IO_S1** | Swap IO <-> S1 | "When **John** and **Mary** went to the store, **John** gave a drink to" | Tests if circuit distinguishes IO from S1 when both appear exactly once |

### Implementation

Each corruption wraps ACDC's `IOIDataset.gen_flipped_prompts(flip_type, seed)`:

```python
# dro_circuit/corruption/ioi.py

IOI_CORRUPTIONS = {
    "S2_IO":   IOICorruptionFamily(("S2", "IO"), seed=42),
    "IO_RAND": IOICorruptionFamily(("IO", "RAND"), seed=42),
    "S_RAND":  IOICorruptionFamily(("S", "RAND"), seed=42),
    "S1_RAND": IOICorruptionFamily(("S1", "RAND"), seed=42),
    "IO_S1":   IOICorruptionFamily(("IO", "S1"), seed=42),
}
```

### Which corruption is hardest?

Different corruptions stress different sub-circuits. **IO_RAND** tends to be the hardest for ERM circuits because it directly removes the IO name, forcing the circuit to rely entirely on stored IO information from earlier layers. A circuit discovered under only S2_IO corruption may fail catastrophically under IO_RAND — this is exactly the failure mode that DRO aggregation addresses.

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
| **EAP-IG** | `vendor/EAP-IG/src/eap/` | `attribute()` for edge scoring (EAP method), `Graph` for circuit representation, `evaluate_graph()` for activation patching |
| **ACDC** | `vendor/Automatic-Circuit-Discovery/` | `IOIDataset` for generating IOI data, `gen_flipped_prompts()` for corruption variants |

Key EAP interfaces:
- `Graph.from_model(model)` — create edge graph from HookedTransformer
- `attribute(model, graph, dataloader, metric, method)` — compute edge scores
- `attribute(..., per_example=True)` — per-example scores, shape $(N, n_\text{fwd}, n_\text{bwd})$
- `evaluate_graph(model, graph, dataloader, metric)` — activation patching evaluation
- `graph.apply_topn(n, absolute)` — select top-n edges
- `graph.in_graph` — boolean mask of selected edges
- `graph.scores` — edge score tensor $(n_\text{fwd}, n_\text{bwd})$
