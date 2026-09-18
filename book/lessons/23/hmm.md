# L23: Hidden Markov Models


In autonomous defense operations, an AI agent rarely has direct access to ground truth. An automated Electronic Support Measures (ESM) receiver cannot directly inspect the operating mode of an adversary radar; it only detects intercepted pulses. Similarly, a naval sonar processor cannot directly observe a submarine's propulsion machinery; it only hears acoustic emissions.

:::{admonition} Lesson Objectives
:class: note

* Distinguish observable vs hidden state models.

* Interpret how observations update beliefs.

* Analyze sequences to infer system behavior.
:::


A {term}`Hidden Markov Model (HMM)` solves this challenge by modeling the physical world as a sequence of unobserved {term}`Hidden States` that emit noisy, observable {term}`Observation (Emission)`s.

```{mermaid}
flowchart TD
    subgraph Hidden["Hidden States"]
        direction LR
        X0((X_0)) --> X1((X_1))
        X1 --> X2((X_2))
        X2 --> Xt((X_t))
    end

    subgraph Observed["Emissions"]
        direction LR
        E1[E_1] ~~~ E2[E_2] ~~~ Et[E_t]
    end

    X1 --> E1
    X2 --> E2
    Xt --> Et

    classDef hidden fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef emission fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;

    class X0,X1,X2,Xt hidden;
    class E1,E2,Et emission;
```


## Distinguishing Observable vs. Hidden State Models

The core distinction between standard Markov models and Hidden Markov Models lies in state observability:

* **Observable State Models** ({term}`Markov Chains`): The state $X_t$ is directly measurable at every time step. If an air base tracks weather $X_t \in \{\text{Clear}, \text{Storm}\}$, the flight controller looks outside and records the true state. Inference reduces to projecting transitions forward: $P(X_t \mid X_{t-1})$.

* **Hidden State Models** ({term}`Hidden Markov Model (HMM)`s): The true physical state $X_t$ is hidden behind the "fog of war." The agent only receives indirect evidence $E_t$ produced according to a sensor likelihood function.

## Foundational Independence Assumptions

### An HMM enforces two structural Markov assumptions:

**State Transition Markov Property:** The current hidden state depends strictly on the immediately preceding hidden state:
$$P(X_t \mid X_{0:t-1}, E_{1:t-1}) = P(X_t \mid X_{t-1})$$

**Sensor (Emission) Independence Property:** The current evidence emission depends strictly on the current hidden state:
$$P(E_t \mid X_{0:t}, E_{1:t-1}) = P(E_t \mid X_t)$$

An adversary multi-function phased-array radar transitions between two hidden operational modes: $X_t \in \{\text{Search (S)}, \text{Track (T)}\}$. An airborne ESM receiver cannot see the operator's display. It only intercepts pulse repetition frequencies (PRF): $E_t \in \{\text{Low PRF (L)}, \text{High PRF (H)}\}$.

```{mermaid}
flowchart LR
    subgraph HiddenProcess["Unobserved Tactical State"]
        X0((X_0)) --> X1((X_1))
        X1 --> X2((X_2))
    end
    subgraph ObservedProcess["Intercepted RF Emissions"]
        X1 --> E1[E_1: High PRF]
        X2 --> E2[E_2: Low PRF]
    end
```

## Interpreting How Observations Update Beliefs (Filtering)

The primary online operational task in an HMM is {term}`Filtering (Monitoring)` or tracking the agent's probability distribution over the current hidden state given the entire stream of incoming sensor readings:

$$\quad B(X_t) = P(X_t \mid e_{1:t})$$

Filtering executes as an interleaved two-step recursive loop using the {term}`Forward Algorithm (Filtering)`:

**Step 1: Predict (Time Update / Passage of Time)**

Before processing the new sensor emission at time $t$, the AI projects the previous belief state forward across the state transition model $P(X_t \mid X_{t-1})$:

$$\bar{B}(X_t) = P(X_t \mid e_{1:t-1}) = \sum_{x_{t-1}} P(X_t \mid x_{t-1}) B(x_{t-1})$$

This step increases entropy (uncertainty grows because the adversary may have changed modes).

**Step 2: Update (Measurement / Evidence Update)**

When new evidence $e_t$ arrives, the AI conditions on the emission using Bayes' Rule:

$$B(X_t) = P(X_t \mid e_{1:t}) = \alpha P(e_t \mid X_t) \bar{B}(X_t)$$

where $\alpha = \frac{1}{\sum_{x_t} P(e_t \mid x_t) \bar{B}(x_t)}$ is the {term}Normalization Constant ($\alpha$). This step decreases entropy (evidence focuses probability on states likely to produce $e_t$).

### Step-by-Step Walkthrough: Radar Mode Tracking

Operational Parameters:
* Prior Base Belief at $t=0$: 

$$B(X_0) = \langle P(S)=0.80, \, P(T)=0.20 \rangle$$

* Transition Model $P(X_t \mid X_{t-1})$:

$$P(S \mid S) = 0.70, \quad P(T \mid S) = 0.30$$

$$P(S \mid T) = 0.10, \quad P(T \mid T) = 0.90$$

* Sensor Emission Model 

$P(E_t \mid X_t)$:

$$P(L \mid S) = 0.80, \quad P(H \mid S) = 0.20$$

$$P(L \mid T) = 0.10, \quad P(H \mid T) = 0.90$$

**Mission Event:** At time $t=1$, the ESM pod intercepts a High PRF pulse: $E_1 = H$. What is the updated probability that the radar is currently in Track mode?

**Phase 1:** Time Prediction ($\bar{B}(X_1)$)

$$\bar{B}(S) = P(S \mid S)B(S_0) + P(S \mid T)B(T_0) = (0.70)(0.80) + (0.10)(0.20) = 0.56 + 0.02 = 0.58$$

$$\bar{B}(T) = P(T \mid S)B(S_0) + P(T \mid T)B(T_0) = (0.30)(0.80) + (0.90)(0.20) = 0.24 + 0.18 = 0.42$$


**Phase 2:** Evidence Update with $E_1 = H$

Compute unnormalized values:

$$P(X_1 = S, E_1 = H) = P(H \mid S) \bar{B}(S) = 0.20 \times 0.58 = 0.116$$

$$P(X_1 = T, E_1 = H) = P(H \mid T) \bar{B}(T) = 0.90 \times 0.42 = 0.378$$

Compute the total evidence probability:

$$P(E_1 = H) = 0.116 + 0.378 = 0.494$$

Normalize to find the filtered belief $B(X_1)$:

$$B(S_1) = \frac{0.116}{0.494} \approx 0.2348 \text{ (23.48\%)}$$

$$B(T_1) = \frac{0.378}{0.494} \approx 0.7652 \text{ (76.52\%)}$$

Despite the radar beginning with an 80% prior probability of being in Search, intercepting a single High PRF pulse immediately drives our operational belief to 76.52% in favor of Track mode.

## Analyzing Sequences to Infer System Behavior

When analyzing streams of tactical intelligence over time, operators perform three distinct inferential tasks:

| Inference Task | Mathematical Formulation | Operational Application
| {term}`Filtering (Monitoring)` | $P(X_t \mid e_{1:t})$ |  "Is that radar tracking me right now?" | 
| {term}`Smoothing (HMM)` | $P(X_k \mid e_{1:t}), \, k < t$ | "Where was the submarine at 0200 hours given all data collected through 0600?" | 
| {term}`Decoding (HMM)` | $\arg\max_{x_{1:t}} P(x_{1:t} \mid e_{1:t})$ | "What exact sequence of maneuvers did the adversary execute?" |

### Sequence Decoding via the Viterbi Algorithm

Finding the most probable sequence of hidden states is not equivalent to picking the individual maximum of each time step's filtered belief. An individually likely state might belong to an impossible or highly improbable transition pair.

The {term}`Viterbi Algorithm` uses dynamic programming to find the optimal path through the state trellis:

$$v_t(j) = \max_i \left[ v_{t-1}(i) \cdot P(X_t = j \mid X_{t-1} = i) \right] \cdot P(e_t \mid X_t = j)$$

By maintaining back-pointers to the maximizing previous state $i$, the agent backtracks to extract the single most probable sequence of military actions.

---


## Knowledge Check & Practice Questions

1. In a Hidden Markov Model monitoring an autonomous underwater vehicle (AUV), the true engine state $X_t$ transitions between Stealth and Cavitating, while a passive sonobuoy records acoustic emissions $E_t$. Which of the following conditional independence statements is guaranteed by the HMM architecture?<br>

A) $E_t \perp E_{t-1} \mid X_{t-1}$<br>
B) $E_t \perp X_{t-1} \mid X_t$<br>
C) $X_t \perp X_{t-1} \mid E_t$<br>
D) $X_{t+1} \perp X_t \mid E_t$<br>

<br>

2. An AI defense tracking system performs a two-step filtering cycle at each time step. What effect does the Time Prediction step (passage of time) have on the system's certainty regarding the hidden state before sensor data is incorporated?<br>

A) It concentrates probability onto the single most likely state.<br>
B) It eliminates all false positives.<br>
C) It increases uncertainty (entropy) by diffusing probability across possible transitions.<br>
D) It resets the belief state to a uniform distribution.<br>

<br>

3. A cyber defense team analyzes an Advanced Persistent Threat (APT) intrusion using an HMM. At 0800, the team has accumulated evidence from 0000 to 0800. If the team computes $P(X_{0300} \mid e_{0000:0800})$ to determine whether lateral movement occurred at 0300, which inference task are they performing?<br>

A) Filtering<br>
B) Prediction<br>
C) Smoothing<br>
D) Viterbi Decoding<br>

<br>

4. Why can an operator NOT determine the most probable sequence of hidden states $\arg\max_{x_{1:t}} P(x_{1:t} \mid e_{1:t})$ by simply selecting the maximum probability state from the filtered belief $B(X_t)$ at each individual time step?<br>

A) The filtered belief incorporates future evidence that distorts the path.<br>
B) The sequence of pointwise argmax states may contain transitions that have zero probability ($P(X_t \mid X_{t-1}) = 0$).<br>
C) Filtering requires the Viterbi trellis to be calculated first.<br>
D) Pointwise selection fails to normalize probabilities to 1.0.<br>