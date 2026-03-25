# Problem Setup

## Background and Motivation

Existing circuit discovery methods, such as the EAP family, treat a language model as a computation graph composed of nodes and edges. They typically:

1. assign an importance score to each node or edge,
2. aggregate these scores across a dataset using mean or sum,
3. select a sparse subgraph (circuit) by retaining the top-$B$ edges, where $B$ is the circuit budget.

This paradigm is naturally aligned with **empirical risk minimization (ERM)**: the selected circuit is the one that performs best on average over the evaluation data.

However, for **faithfulness**, average-case performance may be insufficient. A circuit may appear faithful because it performs well on average, while still failing systematically under certain structured perturbations. If the goal is to identify a circuit that truly captures the underlying task computation, then faithfulness should also account for **worst-case** or **group-wise** failures under task-preserving interventions.

## Grouped Intervention Dataset

### Prior Pair-Level Organization

We focus on tasks such as **Indirect Object Identification (IOI)**, where prior work already includes multiple manually designed intervention patterns. These patterns are typically used to generate a collection of clean-corrupt pairs for attribution or patching analysis.

Formally, prior datasets are organized as a set of independent pairs:

$$
\mathcal{D}_{\text{pair}} = \{(x_j^{\text{clean}},\; x_j^{\text{corrupt}})\}_{j=1}^{M}
$$

where each pair is treated as an independent data point. This organization is suitable for pairwise attribution methods, but it does not directly support reasoning about **worst-case faithfulness around the same underlying example**.

### Reorganized Structure

Our key idea is to reorganize the data from a pair-level dataset into a **grouped intervention dataset**.

Let

$$
\mathcal{X} = \{x_i\}_{i=1}^{N}
$$

denote a set of clean task instances. For each clean sample $x_i$, we define $K$ intervention operators $T_1, T_2, \dots, T_K$ and construct $K$ corrupted variants:

$$
\tilde{x}_{ik} = T_k(x_i), \qquad k \in [K].
$$

This gives a grouped dataset of the form:

$$
\mathcal{D}_{\text{grouped}} = \left\{\left(x_i,\; \mathcal{G}_i\right)\right\}_{i=1}^{N}, \qquad \mathcal{G}_i = \{\tilde{x}_{i1}, \dots, \tilde{x}_{iK}\}
$$

where $\mathcal{G}_i$ is the **intervention set** associated with clean example $x_i$.

Under this formulation, each clean sample is no longer associated with a single corrupt counterpart, but with a **local corruption neighborhood** induced by multiple structured interventions.

### Conceptual Shift

In prior work, interventions are primarily used to generate **independent clean-corrupt pairs** for attribution aggregation.

In our setting, interventions define a **structured uncertainty set** around each clean example. This allows faithfulness to be studied not only at the level of individual pairs, but also at the level of **sets of semantically related perturbations**.

As a result, faithfulness becomes a question of whether a circuit remains faithful **uniformly across multiple intervention modes applied to the same underlying task instance**.

## Candidate Circuits

Let $f$ denote the full model, and let $c$ denote a candidate sparse circuit extracted from the full computation graph. Let $f_c$ denote the model induced by retaining only the edges/nodes in $c$.

The goal is to evaluate whether $f_c$ faithfully reproduces the relevant behavior of $f$ on the target task.

## Faithfulness

Following prior work, we say that a circuit is **faithful** to a model's task behavior if all edges outside the circuit can be corrupted while the model's task behavior remains essentially unchanged.

Operationally, we run the model on a clean input $x$, but for every edge not included in the circuit, we replace its activation with the corresponding activation obtained from a corrupted input $\tilde{x}$. If the resulting model still exhibits the same task-relevant behavior as the full model on $x$, then the circuit is faithful.

More concretely, let $v$ be a node, and let $E_v$ be the set of edges entering $v$. For an edge $e = (u, v)$, let $I_e \in \{0, 1\}$ indicate whether $e$ is included in the circuit. Let $z_u(x)$ denote the output of node $u$ on clean input $x$, and let $z_u(\tilde{x})$ denote the corresponding output on corrupted input $\tilde{x}$. Then the intervened input to node $v$ is

$$
\sum_{e=(u,v) \in E_v} I_e \, z_u(x) + (1 - I_e) \, z_u(\tilde{x}).
$$

If all edges are retained, this reduces to the full model on the clean input. If no edges are retained, this reduces to the corrupted computation. A circuit is faithful when retaining only its edges is sufficient to preserve the original task behavior despite corrupting the rest of the graph.

## Task Metric and Normalized Faithfulness

Let $M_f(x)$ denote a task-specific metric that captures the relevant model behavior when the full model $f$ runs on input $x$. For IOI, a standard choice is the **logit difference** between the correct indirect object token and the distractor token. More generally, $M$ may be a logit difference, a probability difference, or another task-specific contrast.

For a clean-corrupt pair $(x, \tilde{x})$, define:

- $b = M_f(x)$: the full model's task metric on the clean input,
- $b' = M_f(\tilde{x})$: the full model's task metric on the corrupted input,
- $\hat{m} = M_{f_c}(x; \tilde{x})$: the circuit-intervened model's task metric, where all edges outside $c$ carry corrupted activations.

We define the **normalized faithfulness** of circuit $c$ on $(x, \tilde{x})$ as

$$
\operatorname{Faith}(c;\, x, \tilde{x}) = \frac{\hat{m} - b'}{b - b'}.
$$

This metric equals $1$ when the circuit fully recovers the clean model behavior, and equals $0$ when it performs no better than the corrupted baseline.

## Faithfulness Loss

To optimize or aggregate across interventions, we define a faithfulness loss

$$
\ell(c;\, x, \tilde{x}) = 1 - \operatorname{Faith}(c;\, x, \tilde{x}).
$$

In the IOI setting, the most natural primary metric is normalized faithfulness defined using the IOI logit difference.

## ERM-Style Faithfulness

Under the grouped intervention dataset, the standard average-case notion of faithfulness corresponds to minimizing the average loss over all corrupted variants:

$$
R_{\text{ERM}}(c) = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \ell(c;\, x_i, \tilde{x}_{ik}).
$$

All intervention instances are flattened and treated equally. This objective captures **average-case faithfulness**.

## DRO-Style Faithfulness

### Per-Example Worst-Case (Local DRO)

To capture robustness of the circuit across structured perturbations, we define a local DRO-style objective:

$$
R_{\text{DRO}}^{\text{local}}(c) = \frac{1}{N} \sum_{i=1}^{N} \max_{k \in [K]} \ell(c;\, x_i, \tilde{x}_{ik}).
$$

This objective evaluates the worst-performing intervention for each clean example. It captures the following intuition:

> A circuit should not be considered faithful for an example if there exists a natural intervention of that example under which the circuit fails badly.

### Worst-Group (Group DRO)

Because each $T_k$ corresponds to a specific intervention family, we can also define a group-wise worst-case notion:

$$
R_{\text{DRO}}^{\text{group}}(c) = \max_{k \in [K]} \frac{1}{N} \sum_{i=1}^{N} \ell(c;\, x_i, \tilde{x}_{ik}).
$$

This objective identifies whether a circuit systematically fails on a specific intervention type. It captures **worst-group faithfulness** across intervention families.

## Evaluation Views

Our formulation supports two complementary evaluation views.

**Pair View.** Each pair $(x_i, \tilde{x}_{ik})$ is treated as an independent unit, matching prior work. This view enables direct comparison with existing circuit discovery methods.

**Grouped View.** Each clean example $x_i$ together with its full intervention set $\mathcal{G}_i$ is treated as one unit. This view enables evaluation of average-case, per-example worst-case, and worst-group faithfulness.

## Problem Statement

Given:

- a full model $f$,
- a target task (e.g. IOI) with task metric $M$,
- a set of clean examples $\mathcal{X} = \{x_i\}_{i=1}^{N}$,
- $K$ structured intervention operators $\{T_k\}_{k=1}^{K}$,
- and a circuit budget $B$,

construct the grouped intervention dataset $\mathcal{D}_{\text{grouped}}$ and evaluate candidate circuits $c$ under both:

1. **ERM-style average faithfulness** $R_{\text{ERM}}(c)$, and
2. **DRO-style worst-case faithfulness** $R_{\text{DRO}}^{\text{local}}(c)$ and $R_{\text{DRO}}^{\text{group}}(c)$.

The central question is:

> Does a circuit that is faithful on average also remain faithful across all structured interventions of the same underlying example?

## Main Hypothesis

We hypothesize that:

- **ERM-style circuit selection** tends to favor circuits that are small and perform well on average, but may rely on brittle shortcuts and fail under certain interventions;
- **DRO-style circuit selection** tends to favor circuits that are more stable across structured perturbations, and therefore better reflect the true computation underlying the task.
