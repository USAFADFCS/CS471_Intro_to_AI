#  L20: Bayesian Networks

:::{admonition} Lesson Objectives
:class: note

* Represent probabilistic systems as graphs.

* Identify dependencies and conditional independence.

* Interpret network structure (parents, children, ancestors, and descendants).
:::

Full joint probability tables scale exponentially with the number of operational variables ($2^n$ entries for $n$ binary variables). On a modern battlefield with hundreds of sensors, weather systems, and enemy platforms, maintaining an unconstrained joint distribution table is computationally intractable.

A {term}`Bayesian Network` solves this combinatorial explosion by using a {term}Directed Acyclic Graph (DAG) to represent direct dependencies and conditional independencies between variables.

## Representing Probabilistic Systems as Graphs

A Bayesian network consists of two components:

A Directed Acyclic Graph (DAG): Nodes represent random variables, and directed edges represent direct statistical or causal dependencies.

A Set of {term}Conditional Probability Table (CPT)s: Each node $X_i$ maintains a local distribution conditioned strictly on its immediate {term}Parent (Bayesian Network)s: $P(X_i \mid Parents(X_i))$.

Consider an autonomous theater air-strike planning system with four variables:

$W$: Weather in target sector (Severe vs. Clear)

$V$: Visual Sensor Visibility (Good vs. Poor)

$C$: SATCOM Communications Reliability (Reliable vs. Disrupted)

$M$: Strike Mission Success (Success vs. Abort)

The graph structure is defined by the following directed edges:


$$W \to V, \quad W \to C, \quad V \to M, \quad C \to M$$

             ┌─────────────┐
             │ Weather (W) │
             └──────┬──────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
     ┌──────────────┐┌──────────────┐
     │Visibility (V)││  SATCOM (C)  │
     └──────┬───────┘└──────┬───────┘
            │               │
            └───────┬───────┘
                    ▼
             ┌─────────────┐
             │ Mission (M) │
             └─────────────┘


## Interpreting Network Structure

Reading family relationships within the DAG reveals which operational parameters directly control or influence downstream outcomes:

Parents: The immediate upstream nodes with directed edges pointing into a node.

Example: For $M$, the incoming edges originate from $V$ and $C$. Therefore, $Parents(M) = \{V, C\}$.

Children: The immediate downstream nodes that receive directed edges from a node.

Example: For $W$, directed edges point outward to $V$ and $C$. Therefore, $Children(W) = \{V, C\}$.

Ancestors: All nodes reachable by traversing upstream along directed paths.

Can we trace an upstream path from $M$ to $W$? Yes ($M \leftarrow V \leftarrow W$ or $M \leftarrow C \leftarrow W$). Therefore, $W$ is an ancestor of $M$ ($Ancestors(M) = \{V, C, W\}$).

Descendants: All nodes reachable by traversing downstream along directed edges.

Example: The descendants of $W$ are $\{V, C, M\}$.

## Identifying Dependencies and Conditional Independence

The primary power of a Bayesian network lies in its ability to encode {term}Conditional Independence directly through graphical topology.

### The Local Markov Assumption

In any Bayesian network, each variable is conditionally independent of its non-descendants given its parents:


$$X_i \perp \text{NonDescendants}(X_i) \mid Parents(X_i)$$

### Common Cause (The Diverging Fork)

Look at the sub-graph $V \leftarrow W \to C$:

* Unconditioned (Marginal): If weather $W$ is unknown, $V$ and $C$ are dependent. Observing poor visibility suggests bad weather, which increases the likelihood that satellite communications are also disrupted.

* Conditioned on the Common Cause: Once we receive a meteorological satellite report verifying the exact weather condition $W$, knowing the optical visibility $V$ gives zero additional information about the SATCOM radio status $C$.

$V$ and $C$ are conditionally independent given $W$:


$$V \perp C \mid W \iff P(V, C \mid W) = P(V \mid W)P(C \mid W)$$

### Factoring the Full Joint Distribution

By applying the chain rule of Bayesian networks, the full joint probability distribution factors into the product of local CPTs:

$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i \mid Parents(X_i))$$

For the mission model:

$$P(W, V, C, M) = P(W) \cdot P(V \mid W) \cdot P(C \mid W) \cdot P(M \mid V, C)$$

A standard unconstrained joint table for 4 binary variables requires $2^4 - 1 = 15$ independent probability values. By exploiting the graph's conditional independence, the CPTs only require:

$P(W)$: $1$ parameter

$P(V \mid W)$: $2$ parameters

$P(C \mid W)$: $2$ parameters

$P(M \mid V, C)$: $4$ parameters

Total: $1 + 2 + 2 + 4 = 9$ independent parameters, providing significant memory savings and robust parameter estimation under limited combat reconnaissance data.

## Knowledge Check & Practice Questions

1. Consider a Bayesian network modeling drone strike operations: $W \to V$, $W \to C$, $V \to M$, $C \to M$, where $W$ is weather, $V$ is visibility, $C$ is communications, and $M$ is mission success. What are the immediate parents of $M$?
A) $W$
B) $V$ only
C) $V$ and $C$
D) $W, V$, and $C$

2. In the Bayesian network $W \to V$, $W \to C$, $V \to M$, $C \to M$, which conditional independence relationship is guaranteed by the graphical structure?
A) $W \perp M \mid V$
B) $V \perp C \mid W$
C) $V \perp M \mid C$
D) $W \perp C \mid M$

3. In the directed network $W \to V$, $W \to C$, $V \to M$, $C \to M$, is the weather node $W$ considered an ancestor of mission success $M$?
A) No, because $W$ does not have a direct edge pointing into $M$.
B) Yes, because a directed path exists from $W$ to $M$ through intermediate nodes.
C) No, because $W$ is separated from $M$ by two parallel paths.
D) Yes, but only when $C$ is unobserved.

4. An autonomous naval sensor grid has 5 binary variables. Without independence assumptions, the full joint probability table requires 31 independent parameters ($2^5 - 1$). If the variables form a simple chain Bayesian network $X_1 \to X_2 \to X_3 \to X_4 \to X_5$, how many independent parameters are needed across all CPTs?
A) 5
B) 9
C) 15
D) 25
