# Experiment Setup

## Overview

Our goal is to compare **ERM-style** and **DRO-style** circuit discovery under a unified experimental framework. The framework is defined over three axes:

- **Circuit discovery method** (e.g. EAP-IG, ACDC),
- **Model** (e.g. GPT-2 small),
- **Task** (e.g. IOI).

We first illustrate the framework using a concrete running example — **EAP-IG on GPT-2 small for IOI** — and then discuss how it generalizes.

## Running Example

We instantiate the framework using:

- **EAP-IG** as the circuit discovery method,
- **GPT-2 small** as the base model $f$,
- **Indirect Object Identification (IOI)** as the task.

This setting is a natural starting point because:

1. GPT-2 small is a standard model in mechanistic interpretability,
2. IOI is a canonical benchmark task for circuit analysis,
3. EAP-IG provides edge-level importance scores that can be naturally aggregated under both ERM and DRO formulations.

## Dataset Construction

### Clean Dataset

We construct a set of clean IOI examples $\mathcal{X} = \{x_i\}_{i=1}^{N}$. Each example $x_i$ contains:

- a sentence template with multiple names,
- a well-defined indirect object prediction target,
- and a corresponding distractor token.

### Grouped Interventions

For each clean example $x_i$, we apply $K$ IOI-specific intervention operators $T_1, \dots, T_K$. These interventions may include:

- name substitution (e.g. S2 $\leftrightarrow$ IO swap),
- distractor manipulation (e.g. random IO replacement),
- template or positional variation,
- or other task-preserving perturbations.

Following the grouped intervention dataset defined in [Problem Setup](problem-setup.md), this yields:

$$
\mathcal{D}_{\text{IOI}} = \left\{(x_i,\; \mathcal{G}_i)\right\}_{i=1}^{N}, \qquad \mathcal{G}_i = \{\tilde{x}_{i1}, \dots, \tilde{x}_{iK}\}.
$$

We split this dataset into:

- a **discovery set** for computing circuit scores,
- a **validation set** for selecting hyperparameters (e.g. circuit budget $B$),
- and a **test set** for final faithfulness evaluation.

## Edge-Level Scoring

Given a clean-corrupt pair $(x_i, \tilde{x}_{ik})$, EAP-IG assigns each edge $e$ an importance score

$$
s(e;\, x_i, \tilde{x}_{ik}),
$$

which quantifies how much that edge contributes to the target task behavior. The key design choice in our work is **how to aggregate** these pair-level scores across the grouped intervention dataset.

## ERM vs DRO Aggregation of Scores

### ERM Aggregation

The ERM version aggregates scores by averaging over all examples and interventions:

$$
S_{\text{ERM}}(e) = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} s(e;\, x_i, \tilde{x}_{ik}).
$$

The ERM circuit is obtained by selecting the top-$B$ edges according to $S_{\text{ERM}}(e)$.

### DRO Aggregation

The DRO version emphasizes worst-case intervention behavior.

**Local DRO** takes the per-example worst-case score:

$$
S_{\text{DRO}}^{\text{local}}(e) = \frac{1}{N} \sum_{i=1}^{N} \max_{k \in [K]} s(e;\, x_i, \tilde{x}_{ik}).
$$

**Group DRO** takes the worst intervention family on average:

$$
S_{\text{DRO}}^{\text{group}}(e) = \max_{k \in [K]} \frac{1}{N} \sum_{i=1}^{N} s(e;\, x_i, \tilde{x}_{ik}).
$$

The DRO circuit is constructed using the same budget $B$.

### Controlled Comparison

In all comparisons, we keep fixed:

- the base model $f$,
- the task and its intervention families,
- the discovery dataset,
- the pair-level scoring method (EAP-IG),
- and the circuit budget $B$.

The **only** difference between the ERM and DRO versions is the aggregation rule.

## Task Metric

For IOI, we use the task metric $M$ defined in [Problem Setup](problem-setup.md). Concretely, the **logit difference** is:

$$
M_f(x) = \mathrm{logit}(\text{correct IO} \mid x) - \mathrm{logit}(\text{distractor} \mid x).
$$

## Evaluation Protocol

We evaluate circuit faithfulness using the normalized faithfulness $\operatorname{Faith}(c;\, x, \tilde{x})$ defined in [Problem Setup](problem-setup.md).

For grouped intervention data, we report three metrics:

- **Average faithfulness** — corresponds to $R_{\text{ERM}}(c)$,
- **Per-example worst-case faithfulness** — corresponds to $R_{\text{DRO}}^{\text{local}}(c)$,
- **Worst-group faithfulness** — corresponds to $R_{\text{DRO}}^{\text{group}}(c)$.

## Comparison Protocol

For each circuit budget $B$, we construct:

- an **ERM circuit** $c_{\text{ERM}}^{(B)}$,
- and a **DRO circuit** $c_{\text{DRO}}^{(B)}$.

We compare them on the held-out test set using:

1. average faithfulness across all interventions,
2. per-example worst-case faithfulness,
3. worst-group faithfulness across intervention families.

This protocol allows us to test whether DRO-style aggregation yields circuits that are more stable under structured interventions, while maintaining competitive average-case performance.

## Scalability

Although the running example focuses on EAP-IG, GPT-2 small, and IOI, the same protocol generalizes immediately:

- Replacing **EAP-IG** with another circuit discovery method changes only the pair-level scoring rule $s(e;\, x, \tilde{x})$.
- Replacing **GPT-2 small** with another model changes only the base computation graph and activations.
- Replacing **IOI** with another task changes only the task dataset, intervention families, and task metric $M$.

The running example should be viewed as a concrete illustration of a more general ERM-vs-DRO framework for circuit discovery.
