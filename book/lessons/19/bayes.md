# L19: Bayes' Rule

:::{admonition} Lesson Objectives
:class: note

* Apply Bayes' rule to update beliefs.
* Interpret posterior probabilities.
* Reason about evidence and base rates.
:::

When an AI system operates in an uncertain battlefield, it begins with an initial assumption about the environment. As new sensor intelligence or evidence arrives, the system must mathematically update its assumptions to reflect reality. We accomplish this using Bayes' Rule.

## Applying Bayes' Rule to Update Beliefs

{term}`Bayes' Rule` is a mathematical theorem that defines how to calculate a {term}`Posterior Probability` by reversing conditional probabilities.

$$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$$

* $P(A)$: {term}`Prior Probability`. What we believed about hypothesis $A$ before seeing any new evidence.
* $P(B \mid A)$: {term}`Likelihood`. The probability of observing evidence $B$ if hypothesis $A$ is true.
* $P(B)$: **Marginal Likelihood**. The total probability of observing evidence $B$ under any circumstance.
* $P(A \mid B)$: **Posterior Probability**. Our newly updated belief about hypothesis $A$ after observing evidence $B$.

A satellite subsystem component has a prior failure probability of $P(F) = 0.10$. The onboard diagnostic detector will trigger an alert ($A$) 80% of the time if a failure actually occurs ($P(A \mid F) = 0.80$). The total probability of the detector triggering an alert for any reason is $P(A) = 0.20$.

If the detector alerts, the AI calculates the updated posterior probability of an actual failure:

$$P(F \mid A) = \frac{P(A \mid F)P(F)}{P(A)} = \frac{(0.80)(0.10)}{0.20} = 0.40$$



### Computing Bayesian Probability

To fully compute and interpret Bayes' Rule in decision-making, an AI agent operates with five distinct mathematical probabilities:

```
                  ┌────────────────────────────────────────┐
                  │ 1. Prior Probability: P(H)             │
                  │    Baseline threat rate before evidence│
                  └──────────────────┬─────────────────────┘
                                     │
                  ┌──────────────────▼─────────────────────┐
                  │ 2. Likelihood: P(E | H)                │
                  │    Sensor sensitivity given ground truth│
                  └──────────────────┬─────────────────────┘
                                     │ (Product Rule)
                  ┌──────────────────▼─────────────────────┐
                  │ 3. Joint Probability: P(H, E)          │
                  │    P(H, E) = P(E | H) * P(H)           │
                  └──────────────────┬─────────────────────┘
                                     │ (Law of Total Probability)
                  ┌──────────────────▼─────────────────────┐
                  │ 4. Marginal Evidence: P(E)             │
                  │    P(E) = ∑ P(E | H=h) * P(H=h)        │
                  └──────────────────┬─────────────────────┘
                                     │ (Bayes' Rule Quotient)
                  ┌──────────────────▼─────────────────────┐
                  │ 5. Posterior Probability: P(H | E)     │
                  │    P(H | E) = P(H, E) / P(E)           │
                  └────────────────────────────────────────┘

```

An Airborne Early Warning & Control (AEW&C) radar evaluates whether an unidentified contact in contested airspace is a **Hostile Inbound Missile ($H$)** based on a **Radar Threat Alert ($E$)**:

* **Hypothesis states:** $H$ (Hostile missile present) vs. $\neg H$ (No missile / clear airspace).
* **Evidence states:** $E$ (Radar alarms) vs. $\neg E$ (Radar remains silent).

### Prior Probability — $P(H)$

The baseline probability of the hypothesis occurring across all operational flights *before* evaluating the current sensor reading.

* **Given Intelligence:** Hostile missiles are present in 5% of all radar sweeps.

$$P(H) = 0.05 \quad \implies \quad P(\neg H) = 1 - 0.05 = 0.95$$


### Likelihood (Sensor Sensitivity & False Positive Rate) — $P(E \mid H)$

The conditional probability that the sensor produces evidence $E$ given that the true operational state is known.

* **True Positive Rate (Sensitivity):** If a hostile missile is present, the radar alerts 90% of the time:

$$P(E \mid H) = 0.90 \quad \implies \quad P(\neg E \mid H) = 0.10 \text{ (Missed detection)}$$


* **False Positive Rate (False Alarm):** If the airspace is clear, environmental clutter causes a false alarm 10% of the time:

$$P(E \mid \neg H) = 0.10 \quad \implies \quad P(\neg E \mid \neg H) = 0.90 \text{ (True negative)}$$


### Joint Probability — $P(H, E)$

The probability that a specific hypothesis state and a specific evidence outcome occur simultaneously. It is computed via the **Product Rule**:

$$P(H, E) = P(E \mid H) \cdot P(H)$$

* **Hostile Target and Alert:**

$$P(H, E) = 0.90 \times 0.05 = 0.0450 \text{ (or } 4.50\%)$$


* **Clear Airspace and False Alert:**

$$P(\neg H, E) = P(E \mid \neg H) \cdot P(\neg H) = 0.10 \times 0.95 = 0.0950 \text{ (or } 9.50\%)$$


### Marginal Probability of Evidence — $P(E)$

The total, unconditional probability of observing the evidence across all possible operational states. It is computed using the **Law of Total Probability** (marginalizing over all hypotheses):

$$P(E) = \sum_{h \in \{H, \neg H\}} P(h, E) = P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)$$

* **Total Alarm Probability:**

$$P(E) = 0.0450 + 0.0950 = 0.1400 \text{ (or } 14.00\%)$$

### Posterior Probability — $P(H \mid E)$

The updated probability that the hypothesis is true *after* observing the sensor evidence. It is computed using **Bayes' Rule**:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)} = \frac{P(H, E)}{P(E)}$$

* **Substitution & Solution:**

$$P(H \mid E) = \frac{0.0450}{0.1400} = \frac{45}{140} \approx 0.3214 \text{ (or } 32.14\%)$$


* **Interpretation:** Even though the radar has a 90% detection likelihood, when an alert triggers, there is only a **32.14% probability** that an actual hostile missile is present. Because hostile targets are rare ($P(H) = 0.05$), the false alarms generated from the larger non-hostile population ($0.0950$) outnumber the true positive detections ($0.0450$).


| Probability Type | Mathematical Notation | Formula | Computed Value | Operational Interpretation |
| --- | --- | --- | --- | --- |
| **Prior** | $P(H)$ | Given base rate | $0.0500$ ($5.0\%$) | Baseline threat frequency in sector |
| **Likelihood** | $P(E \mid H)$ | Diagnostic sensitivity | $0.9000$ ($90.0\%$) | Radar detection rate on actual targets |
| **Joint (True Alert)** | $P(H, E)$ | $P(E \mid H)P(H)$<br> | $0.0450$ ($4.5\%$) | Simultaneous presence of threat & alert |
| **Joint (False Alarm)** | $P(\neg H, E)$ | $P(E \mid \neg H)P(\neg H)$ | $0.0950$ ($9.5\%$) | Simultaneous clean air & false alarm |
| **Marginal Evidence** | $P(E)$ | $P(H,E) + P(\neg H,E)$ | $0.1400$ ($14.0\%$) | Overall rate of radar alarm triggers |
| **Posterior** | $P(H \mid E)$ | $\frac{P(E \mid H)P(H)}{P(E)}$<br> | $0.3214$ ($32.14\%$) | Updated threat probability given an alarm |



## Interpreting Posterior Probabilities

A posterior probability is not a guarantee; it is an updated expectation. If a satellite component has a calculated $0.80$ probability of remaining operational, this means that across many comparable mission segments, the component would be expected to remain operational about 80% of the time, though the outcome of any single segment remains uncertain.

## Reasoning About Evidence and Base Rates

A common error is the Base Rate Fallacy—assuming that highly sensitive sensors guarantee high posterior probabilities.

The {term}`Base Rate` is the underlying prior probability of an event in the population. If a target or event is extremely rare, even highly accurate sensors will produce numerous false positives.

Suppose a sensor detector identifies 95% of actual critical failures ($P(\text{Alert} \mid \text{Failure}) = 0.95$), but those physical failures occur in only 1% of missions (Base Rate $P(\text{Failure}) = 0.01$).

Why might an alert still yield a posterior probability of failure ($P(\text{Failure} \mid \text{Alert})$) far below the detector's 95% detection rate?

Because failures are so rare, the false alerts generated from the massive no-failure population vastly outnumber the valid alerts generated from the tiny failure population. The 95% detection rate represents the *likelihood* ($P(\text{Alert} \mid \text{Failure})$), which is fundamentally different from the *posterior* ($P(\text{Failure} \mid \text{Alert})$).

---

## Knowledge Check & Practice Questions

1. Suppose the probability of a drone's primary sensor failing is $P(A)=0.30$, and the probability of it failing given the backup sensor has already failed is $P(A\vert{}B)=0.30$. Are these failure events independent or dependent?<br>
A) Independent<br>
B) Dependent<br>
C) Mutually Exclusive<br>
D) Conditionally Dependent<br>
<br>
2. For a battlefield reconnaissance system, adequate visibility occurs with probability $P(V)=0.75$, and reliable communication given adequate visibility occurs with probability $P(C\vert{}V)=0.80$. What is the joint probability $P(V, C)$?<br>
A) 0.94<br>
B) 0.80<br>
C) 0.60<br>
D) 0.15<br>
<br>
3. An AI diagnostics system tracks a hardware failure prior of $P(F)=0.10$, a sensor likelihood of $P(A\vert{}F)=0.80$, and an overall alert probability of $P(A)=0.20$. What is the posterior probability $P(F\vert{}A)$?<br>
A) 0.08<br>
B) 0.16<br>
C) 0.40<br>
D) 0.80<br>
<br>
4. An automated early-warning detector successfully identifies 95% of actual incoming threats, but these threats occur in only 1% of all radar sweeps. Why might a triggered alert still yield a posterior probability of a threat far below 95%?<br>
A) The detector's likelihood rate is artificially inflated by the base rate.<br>
B) False alerts from the much larger no-threat population outnumber true detections.<br>
C) The events are statistically independent.<br>
D) The algorithm assumes a uniform distribution across all sweeps.<br>



