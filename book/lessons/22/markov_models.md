# L22: Markov Models

:::{admonition} Lesson Objectives
:class: note

* Represent systems using Markov state transitions.

* Compute multi-step state probabilities.

* Explain assumptions of Markov models (the Markov property and stationarity).
:::

In Lesson 20 and Lesson 21, we modeled static operational environments where random variables were arranged in a Directed Acyclic Graph (DAG) without explicit temporal ordering. However, modern command and control (C2) operations unfold dynamically across time:

"How will target sector weather evolve over the next three hours before our strike package reaches the objective?"

"Given active cyber jamming, what is the probability that our primary radar remains operational two time steps from now?"

To model dynamic, time-series phenomena, an autonomous agent employs a {term}`Markov Model` (often called a {term}`Markov Chain`). A Markov model is a probabilistic model of a system that changes discrete states sequentially over time.

## Representing Systems Using Markov State Transitions

A discrete-time Markov model consists of three core components:

A Discrete State Space ($S$): The set of all mutually exclusive configurations the operational system can occupy:


$$S = \{s_1, s_2, \dots, s_N\}$$

An {term}Initial State Distribution ($\mathbf{p}_0$): A probability vector specifying the system's state distribution at time $t=0$:


$$\mathbf{p}_0 = \langle P(X_0 = s_1), P(X_0 = s_2), \dots, P(X_0 = s_N) \rangle, \quad \sum_{i=1}^N P(X_0 = s_i) = 1.0$$

A {term}Transition Matrix ($T$): A square matrix where entry $T_{ij}$ defines the single-step conditional transition probability of moving from state $s_i$ at time $t$ to state $s_j$ at time $t+1$:


$$T_{ij} = P(X_{t+1} = s_j \mid X_t = s_i)$$

Because the system must transition to some state in $S$, each row of the transition matrix forms a valid probability distribution and must sum to exactly $1.0$:


$$\sum_{j=1}^N T_{ij} = 1.0 \quad \text{for all rows } i$$

Tactical State Transition Topology

Consider an autonomous theater weather monitoring station that tracks target airspace conditions over hourly time steps:

$S = \{\text{Clear } (C), \text{Cloudy } (K), \text{Storm } (R)\}$

The transition probabilities between these three environmental states are represented graphically as a state transition diagram:

```{mermaid}
flowchart TD
    C["Clear (C)"]
    K["Cloudy (K)"]
    R["Storm (R)"]

    %% Self loops
    C -->|0.70| C
    R -->|0.70| R

    %% Transitions between Clear and Cloudy
    C -->|0.10| K
    K -->|0.30| C

    %% Transitions between Clear and Storm
    C -->|0.20| R
    R -->|0.10| C

    %% Transitions between Cloudy and Storm
    K -->|0.10| R
    R -->|0.20| K
```

## Transition Matrix Formulation

The corresponding stochastic transition matrix $T$ is structured with rows as current states ($X_t$) and columns as next states ($X_{t+1}$):

$$T = \begin{pmatrix}
P(C \mid C) & P(K \mid C) & P(R \mid C) \\
P(C \mid K) & P(K \mid K) & P(R \mid K) \\
P(C \mid R) & P(K \mid R) & P(R \mid R)
\end{pmatrix}
=
\begin{pmatrix}
0.70 & 0.10 & 0.20 \\
0.30 & 0.60 & 0.10 \\
0.10 & 0.20 & 0.70
\end{pmatrix}$$

Operational Verification: Notice that each row sums to $1.0$:

Row 1 (Clear): $0.70 + 0.10 + 0.20 = 1.0$

Row 2 (Cloudy): $0.30 + 0.60 + 0.10 = 1.0$

Row 3 (Storm): $0.10 + 0.20 + 0.70 = 1.0$

## Assumptions of Markov Models

A Markov model relies on two foundational mathematical assumptions that allow an AI agent to efficiently reason over long time horizons without tracking infinite operational history:

**Assumption 1:** The First-Order Markov Assumption

The {term}First-Order Markov Assumption states that the conditional probability distribution of the future state $X_{t+1}$ depends strictly and solely on the current state $X_t$, and is conditionally independent of all past states:

$$P(X_{t+1} \mid X_t, X_{t-1}, X_{t-2}, \dots, X_0) = P(X_{t+1} \mid X_t)$$

Operational Meaning: To predict whether tomorrow's theater sector will experience a storm, an autonomous planner only needs to know today's weather state. Knowing the sequence of weather states over the preceding three months provides zero additional mathematical predictive power once today's state is verified.

**Assumption 2:** The Stationary Assumption (Time-Homogeneity)

The {term}Stationary Assumption (Time-Homogeneity) asserts that the underlying laws governing state transitions remain constant across all time steps $t$:

$$P(X_{t+1} = s_j \mid X_t = s_i) = P(X_1 = s_j \mid X_0 = s_i) \quad \text{for all } t$$

Operational Meaning: The probability of transitioning from Clear to Storm is identical whether the mission occurs at $t=0$ (0600 hrs) or at $t=10$ (1600 hrs). The transition matrix $T$ does not vary with time.

## Joint Trajectory Factorization

Under these two assumptions, the full joint probability of an entire temporal sequence of states factors into the initial prior multiplied by the chain of single-step transitions:

$$P(X_0 = x_0, X_1 = x_1, \dots, X_t = x_t) = P(X_0 = x_0) \prod_{i=1}^t P(X_i = x_i \mid X_{i-1} = x_{i-1})$$

Tactical Scenario: An air tasking order initiates on a Clear day ($P(X_0 = \text{Clear}) = 1.0$). What is the probability that the sector remains Clear on Day 1, transitions to Cloudy on Day 2, and then Storms on Day 3?

$$P(X_0 = C, X_1 = C, X_2 = K, X_3 = R) = P(X_0 = C) \cdot P(C \mid C) \cdot P(K \mid C) \cdot P(R \mid K)$$

$$P(X_0 = C, X_1 = C, X_2 = K, X_3 = R) = 1.0 \times 0.70 \times 0.10 \times 0.10 = 0.0070 \text{ (or } 0.70\%)$$

## Computing Multi-Step State Probabilities

When commanders plan multi-day operations, the intermediate states are rarely observed in advance. We must compute the marginal probability distribution of states multiple steps into the future.

**Method 1:** Marginalization Over Intermediate States (Mini-Forward Algorithm)

To calculate the probability distribution at $t=1$, we multiply the initial distribution $\mathbf{p}_0$ by the transition matrix:

$$P(X_1 = j) = \sum_{i \in S} P(X_0 = i) \cdot P(X_1 = j \mid X_0 = i)$$

To propagate forward to time $t=2$, we marginalize over all possible intermediate states $X_1$:

$$P(X_2 = j) = \sum_{i \in S} P(X_1 = i) \cdot P(X_2 = j \mid X_1 = i)$$

Substituting $P(X_1 = i)$ from the previous step:

$$P(X_2 = j) = \sum_{i \in S} \left( \sum_{k \in S} P(X_0 = k) \cdot P(X_1 = i \mid X_0 = k) \right) P(X_2 = j \mid X_1 = i)$$

**Method 2:** Matrix Multiplication

Because marginalization over intermediate states is identical to row-vector matrix multiplication, forward propagation over $t$ time steps is computed compactly:

$$\mathbf{p}_1 = \mathbf{p}_0 T$$

$$\mathbf{p}_2 = \mathbf{p}_1 T = (\mathbf{p}_0 T) T = \mathbf{p}_0 T^2$$

In general, for any forward time horizon $k$:

$$\mathbf{p}_k = \mathbf{p}_0 T^k$$

Where the $(i, j)$-th entry of the matrix power $T^k$ represents the {term}Multi-Step Transition Probability of starting in state $s_i$ and arriving in state $s_j$ exactly $k$ steps later: $(T^k)_{ij} = P(X_{t+k} = s_j \mid X_t = s_i)$.

## Detailed Walkthrough: Two-Step Prediction

Intelligence confirms that on Day 0, target sector weather is Clear with 100% certainty:

$$\mathbf{p}_0 = \langle P(C)=1.0, \quad P(K)=0.0, \quad P(R)=0.0 \rangle$$

Using the transition matrix $T$:

$$T = \begin{pmatrix} 0.70 & 0.10 & 0.20 \\ 0.30 & 0.60 & 0.10 \\ 0.10 & 0.20 & 0.70 \end{pmatrix}$$

**Step 1:** Compute state probabilities at Day 1 ($\mathbf{p}_1$)

$$\mathbf{p}_1 = \mathbf{p}_0 T = \begin{pmatrix} 1.0 & 0.0 & 0.0 \end{pmatrix} \begin{pmatrix} 0.70 & 0.10 & 0.20 \\ 0.30 & 0.60 & 0.10 \\ 0.10 & 0.20 & 0.70 \end{pmatrix}$$

$P(X_1 = C) = 1.0(0.70) + 0.0(0.30) + 0.0(0.10) = 0.70$

$P(X_1 = K) = 1.0(0.10) + 0.0(0.60) + 0.0(0.20) = 0.10$

$P(X_1 = R) = 1.0(0.20) + 0.0(0.10) + 0.0(0.70) = 0.20$

$$\mathbf{p}_1 = \langle 0.70, 0.10, 0.20 \rangle$$

**Step 2:** Compute state probabilities at Day 2 ($\mathbf{p}_2$)

Multiply the Day 1 distribution $\mathbf{p}_1$ by the transition matrix $T$:

$$\mathbf{p}_2 = \mathbf{p}_1 T = \begin{pmatrix} 0.70 & 0.10 & 0.20 \end{pmatrix} \begin{pmatrix} 0.70 & 0.10 & 0.20 \\ 0.30 & 0.60 & 0.10 \\ 0.10 & 0.20 & 0.70 \end{pmatrix}$$

Calculate each state component individually:

Probability of Clear on Day 2 ($P(X_2 = C)$):

$$P(X_2 = C) = \sum_{i \in \{C, K, R\}} P(X_1 = i) \cdot P(C \mid i)$$

$$P(X_2 = C) = (0.70 \times 0.70) + (0.10 \times 0.30) + (0.20 \times 0.10)$$

$$P(X_2 = C) = 0.4900 + 0.0300 + 0.0200 = 0.5400 \text{ (or } 54.0\%)$$

Probability of Cloudy on Day 2 ($P(X_2 = K)$):


$$P(X_2 = K) = (0.70 \times 0.10) + (0.10 \times 0.60) + (0.20 \times 0.20)$$

$$P(X_2 = K) = 0.0700 + 0.0600 + 0.0400 = 0.1700 \text{ (or } 17.0\%)$$

Probability of Storm on Day 2 ($P(X_2 = R)$):


$$P(X_2 = R) = (0.70 \times 0.20) + (0.10 \times 0.10) + (0.20 \times 0.70)$$

$$P(X_2 = R) = 0.1400 + 0.0100 + 0.1400 = 0.2900 \text{ (or } 29.0\%)$$

$$\mathbf{p}_2 = \langle 0.5400, 0.1700, 0.2900 \rangle$$


Check that the probabilities sum to $1.0$:

$$0.5400 + 0.1700 + 0.2900 = 1.0000$$

## Stationary Distributions

As time advances ($t \to \infty$), many Markov chains reach an equilibrium where the state probability vector stops changing from one step to the next. This invariant distribution is called the {term}Stationary Distribution ($\mathbf{\pi}$):

$$\mathbf{\pi} = \mathbf{\pi} T, \quad \text{subject to } \sum_{i=1}^N \pi_i = 1.0$$

Operational Meaning: Regardless of whether a campaign begins in Clear weather or a violent Storm, after many weeks, the probability of finding the theater in a Storm on any random day converges to a fixed climatic baseline ($\pi_R$).


## Knowledge Check & Practice Questions

1. What mathematical property guarantees that in a discrete-time Markov chain, the probability distribution over state $X_{t+1}$ is conditionally independent of $X_{t-1}$ given knowledge of state $X_t$? <br>
A) The Stationary Transition Property<br>
B) The First-Order Markov Assumption<br>
C) Ergodic Convergence<br>
D) Laplace Smoothing<br>

<br>

2. An autonomous military communications node operates with two states: Operational ($O$) and Jammed ($J$). The single-step transition matrix is:

$$T = \begin{pmatrix} 0.80 & 0.20 \\ 0.40 & 0.60 \end{pmatrix}$$

If the network begins in the Operational state at $t=0$ ($\mathbf{p}_0 = \langle 1.0, 0.0 \rangle$), what is the probability that the node is Jammed at time $t=2$ ($P(X_2 = J)$)?<br>
A) $0.20$<br>
B) $0.28$<br>
C) $0.32$<br>
D) $0.40$<br>

<br>

3. In a discrete Markov model with $N$ states, what mathematical condition must every valid transition matrix $T$ strictly satisfy?<br>
A) The sum of every column must equal $0.0$.<br>
B) The determinant of $T$ must equal $1.0$.<br>
C) All entries must be non-negative, and each row must sum to exactly $1.0$.<br>
D) The diagonal entries must all be equal.<br>

<br>

4. A combat simulation team models adversary armor movements. The analysts observe that the adversary changes doctrine depending on whether it is daytime or nighttime, causing transition probabilities between sectors to change every 12 hours. Which core assumption of standard discrete Markov models does this violate?<br>
A) The First-Order Markov Assumption<br>
B) The Completeness Principle<br>
C) The Stationary Assumption (Time-Homogeneity)<br>
D) The Law of Total Probability<br>