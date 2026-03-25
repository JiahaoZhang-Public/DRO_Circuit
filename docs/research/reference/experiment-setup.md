# Experiment Setup

## Overview

Our goal is to compare **ERM-style** and **DRO-style** circuit discovery under a unified experimental framework. The framework is defined over three axes:

- **Circuit discovery method** (e.g. EAP, ACDC),
- **Model** (e.g. GPT-2 small),
- **Task** (e.g. IOI).

We first illustrate the framework using a concrete running example — **EAP on GPT-2 small for IOI** — and then discuss how it generalizes.

## Running Example

We instantiate the framework using:

- **EAP** (Edge Attribution Patching) as the circuit discovery method,
- **GPT-2 small** as the base model $f$,
- **Indirect Object Identification (IOI)** as the task.

This setting is a natural starting point because:

1. GPT-2 small is a standard model in mechanistic interpretability,
2. IOI is a canonical benchmark task for circuit analysis,
3. EAP provides edge-level importance scores that can be naturally aggregated under both ERM and DRO formulations.

## Dataset Construction

### Clean Dataset

We construct a set of clean IOI examples $\mathcal{X} = \{x_i\}_{i=1}^{N}$. Each example $x_i$ contains:

- a sentence template with multiple names (IO, S1, S2),
- a well-defined indirect object prediction target (IO token),
- and a corresponding distractor token (S token).

### Grouped Interventions

For each clean example $x_i$, we apply $K$ IOI-specific intervention operators $T_1, \dots, T_K$. Concretely, for IOI the $K = 5$ intervention families are:

| Operator | Description |
|---|---|
| $T_1$ (S2\_IO) | Swap the S2 and IO names |
| $T_2$ (IO\_RAND) | Replace IO with a random name |
| $T_3$ (S\_RAND) | Replace S with a random name |
| $T_4$ (S1\_RAND) | Replace S1 with a random name |
| $T_5$ (IO\_S1) | Swap the IO and S1 names |

Following the grouped intervention dataset defined in [Problem Setup](problem-setup.md), this yields:

$$
\mathcal{D}_{\text{IOI}} = \left\{(x_i,\; \mathcal{G}_i)\right\}_{i=1}^{N}, \qquad \mathcal{G}_i = \{\tilde{x}_{i1}, \dots, \tilde{x}_{iK}\}.
$$

Following standard practice in EAP-IG and related work, the same dataset is used for both edge scoring and faithfulness evaluation.

## Edge-Level Scoring

Given a clean-corrupt pair $(x_i, \tilde{x}_{ik})$, EAP assigns each edge $e$ an importance score $s(e;\, x_i, \tilde{x}_{ik})$ by computing the product of activation differences and metric gradients in a single forward-backward pass.

We run EAP in **per-example mode**: for each corruption family $k$, EAP returns per-example scores

$$
s(e;\, x_i, \tilde{x}_{ik}), \qquad i \in [N],\; k \in [K],
$$

giving a score tensor of shape $(K, N, n_{\text{fwd}}, n_{\text{bwd}})$.

The per-corruption average score is then:

$$
\bar{s}(e;\, k) \;=\; \frac{1}{N} \sum_{i=1}^{N} s(e;\, x_i, \tilde{x}_{ik}).
$$

The key design choice in our work is **how to aggregate** these scores across the $K$ intervention families. Having per-example scores enables both group-level and per-example aggregation strategies.

## ERM vs DRO Aggregation of Scores

### ERM Aggregation

The ERM version aggregates scores by averaging over all corruption families:

$$
S_{\text{ERM}}(e) = \frac{1}{K} \sum_{k=1}^{K} |\bar{s}(e;\, k)|.
$$

The ERM circuit is obtained by selecting the top-$B$ edges according to $S_{\text{ERM}}(e)$.

### DRO Aggregation

The DRO version emphasizes worst-case behavior across corruption families. We support two levels of DRO aggregation.

**Local DRO (per-example worst-case)** — for each example, take the worst corruption, then average over examples. This requires per-example scores:

$$
S_{\text{DRO}}^{\text{local}}(e) = \frac{1}{N} \sum_{i=1}^{N} \max_{k \in [K]} |s(e;\, x_i, \tilde{x}_{ik})|.
$$

**Max (Group DRO)** — selects the worst-case corruption family on average:

$$
S_{\text{DRO}}^{\max}(e) = \max_{k \in [K]} |\bar{s}(e;\, k)|.
$$

**CVaR** — averages the top-$\lceil \alpha K \rceil$ corruption families:

$$
S_{\text{DRO}}^{\text{CVaR}(\alpha)}(e) = \frac{1}{\lceil \alpha K \rceil} \sum_{k \in \text{top-}\lceil \alpha K \rceil} |\bar{s}(e;\, k)|.
$$

When $\alpha \to 0$, CVaR approaches Max; when $\alpha = 1$, CVaR equals the ERM mean.

**Softmax** — uses a temperature-controlled weighted sum:

$$
S_{\text{DRO}}^{\text{softmax}(\tau)}(e) = \sum_{k=1}^{K} \frac{\exp(|\bar{s}(e;\, k)| / \tau)}{\sum_{k'} \exp(|\bar{s}(e;\, k')| / \tau)} \; |\bar{s}(e;\, k)|.
$$

When $\tau \to 0$, softmax approaches Max; when $\tau \to \infty$, it approaches the ERM mean.

The DRO circuit is constructed by selecting the top-$B$ edges under the chosen aggregation rule.

### Why Absolute Values

All aggregation rules operate on absolute scores rather than raw scores. This is because EAP scores can be negative (indicating that an edge reduces the metric), and for circuit selection we care about the **magnitude** of each edge's contribution regardless of sign.

### Note on Aggregation Levels

Local DRO operates on the full per-example score tensor $(K, N, n_{\text{fwd}}, n_{\text{bwd}})$, taking $\max$ over $K$ before averaging over $N$. Group-level aggregators (Max, CVaR, Softmax) operate on the corruption-averaged scores $\bar{s}(e;\, k)$ of shape $(K, n_{\text{fwd}}, n_{\text{bwd}})$. ERM also operates at the group level.

### Controlled Comparison

In all comparisons, we keep fixed:

- the base model $f$,
- the task and its intervention families,
- the discovery dataset,
- the pair-level scoring method (EAP),
- and the circuit budget $B$.

The **only** difference between the ERM and DRO versions is the aggregation rule.

## Task Metric

For IOI, we use the task metric $M$ defined in [Problem Setup](problem-setup.md). Concretely, the **logit difference** is:

$$
M_f(x) = \mathrm{logit}(\text{IO} \mid x) - \mathrm{logit}(\text{S} \mid x).
$$

For EAP edge scoring, we use the negated logit difference as a loss (so that EAP's gradient-based attribution aligns with maximizing the metric).

## Evaluation Protocol

We evaluate circuit faithfulness using the normalized faithfulness $\operatorname{Faith}(c;\, x, \tilde{x})$ defined in [Problem Setup](problem-setup.md):

$$
\operatorname{Faith}(c;\, x, \tilde{x}) = \frac{\hat{m} - b'}{b - b'}
$$

where $b = M_f(x)$ is the full model metric on the clean input, $b' = M_f(\tilde{x})$ is the full model metric on the corrupted input, and $\hat{m} = M_{f_c}(x;\, \tilde{x})$ is the circuit-intervened model metric.

For grouped intervention data, we report three evaluation metrics (using the faithfulness loss $\ell = 1 - \operatorname{Faith}$):

- **Average faithfulness** — corresponds to $R_{\text{ERM}}(c)$: mean loss across all $(i, k)$ pairs,
- **Worst-group faithfulness** — corresponds to $R_{\text{DRO}}^{\text{group}}(c)$: the corruption family with the highest average loss,
- **Per-example worst-case faithfulness** — corresponds to $R_{\text{DRO}}^{\text{local}}(c)$: for each example, take the worst corruption, then average over examples. This metric is computed at evaluation time by retaining per-example metric values.

## Comparison Protocol

For each circuit budget $B$, we construct:

- an **ERM circuit** $c_{\text{ERM}}^{(B)}$ using $S_{\text{ERM}}$,
- a **Local DRO circuit** $c_{\text{DRO-local}}^{(B)}$ using $S_{\text{DRO}}^{\text{local}}$,
- and a family of **Group DRO circuits** $c_{\text{DRO}}^{(B)}$ using $S_{\text{DRO}}^{\max}$, $S_{\text{DRO}}^{\text{CVaR}}$, or $S_{\text{DRO}}^{\text{softmax}}$.

We compare them using:

1. average faithfulness across all interventions,
2. worst-group faithfulness across intervention families,
3. per-example worst-case faithfulness.

This protocol allows us to test whether DRO-style aggregation yields circuits that are more stable under structured interventions, while maintaining competitive average-case performance.

## Scalability

Although the running example focuses on EAP, GPT-2 small, and IOI, the same protocol generalizes immediately:

- Replacing **EAP** with another circuit discovery method changes only the pair-level scoring rule $s(e;\, x, \tilde{x})$.
- Replacing **GPT-2 small** with another model changes only the base computation graph and activations.
- Replacing **IOI** with another task changes only the task dataset, intervention families, and task metric $M$.

The running example should be viewed as a concrete illustration of a more general ERM-vs-DRO framework for circuit discovery.
