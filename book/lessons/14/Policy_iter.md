# L14 — Reinforcement Learning II (Policy Iteration)

:::{admonition} Lesson Objectives
:class: note

* Explain and define policy extraction, policy evaluation, and policy iteration.
* Apply policy iteration to a simple decision problem.
* Compare policy iteration and value iteration.
* Explain relationship to reinforcement learning.
:::

## Introduction to Policy Operations

In the previous lesson, we used Value Iteration to calculate the optimal numerical utility of every state on a map. However, our tactical agents ultimately need a playbook—a policy ($\pi$)—to tell them exactly which action to execute.

To bridge the gap between mathematical state-values and actionable commands, we rely on three distinct operational concepts: **Policy Extraction**, **Policy Evaluation**, and **Policy Iteration**. All three of these algorithms are variations of the core Bellman updates, relying on fragments of one-step expectimax lookaheads. They differ only in whether we plug in a fixed policy or maximize over all possible actions.

### 1. Policy Evaluation

Sometimes an AI agent is simply handed a standard operating procedure (a fixed policy) and needs to calculate how effective it is. **Policy Evaluation** takes a specific, fixed policy and computes the state values strictly for that chosen policy.

Because the agent's actions are already decided by the playbook, we remove the $\max_a$ operator from the Bellman equation, turning the complex search into a straightforward calculation:

$$V_{k+1}^\pi(s) \leftarrow \sum_{s'} T(s, \pi(s), s') [R(s, \pi(s), s') + \gamma V_k^\pi(s')]$$

### 2. Policy Extraction

Once an agent knows the optimal values ($V^*$) or the evaluated values of a specific policy ($V^\pi$), it needs to determine its next move. Because deciding how to act based solely on state-values is not obvious, the agent must perform a mini-expectimax search (a one-step lookahead) to find the best immediate action.

This process of converting state-values into an actionable playbook is called **Policy Extraction**. It utilizes the $\arg\max_a$ operator, which returns the actual action that yields the highest score, rather than the score itself:

$$\pi^*(s) = \arg\max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]$$

### 3. Policy Iteration

**Policy Iteration** is an alternative algorithm to Value Iteration that directly searches for the optimal playbook. The algorithm asks a simple question: *"I have picked a policy; is it optimal?"*.

It answers this by looping through two steps:

1. Evaluate the current policy using Policy Evaluation.
2. Pick the next policy using Policy Extraction to see if a mathematically better move exists.

If the newly extracted policy is identical to the old one, the algorithm knows it has found the optimal solution and exits. Mathematically, generating the new policy looks like this:

$$\pi_{i+1}(s) = \arg\max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^{\pi_i}(s')]$$

## Applying Policy Iteration

Let us apply this to a simple tactical scenario involving an autonomous sentry deciding whether to patrol or recharge.

```{mermaid}
graph LR
    S((Base<br>State)) -- "Patrol" --> P[Reward: +5]
    S -- "Recharge" --> R[Reward: 0]

    classDef default fill:#f9f9f9,stroke:#888,stroke-width:2px,color:#000;
    classDef base fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    linkStyle default stroke:#a0a0a0,stroke-width:2px;
    class S base;
```

**Step 1: Initialization**
The sentry is initialized with a flawed, random policy: $\pi_0 = \text{"Always Recharge"}$.

**Step 2: Policy Evaluation**
The AI evaluates $\pi_0$. Because recharging yields no tactical advantage, the state value converges to $V^{\pi_0} = 0$.

**Step 3: Policy Extraction (Improvement)**
The AI runs a one-step lookahead against the environment. It sees that selecting the "Patrol" action yields an expected return of $+5$. Because $+5 > 0$, the AI updates its playbook to $\pi_1 = \text{"Always Patrol"}$.

**Step 4: Convergence Verification**
The AI evaluates $\pi_1$, yielding a high state value. It runs extraction again. Because "Patrol" remains the highest yielding action, the extracted policy ($\pi_2$) is perfectly identical to $\pi_1$. The algorithm halts, having successfully found the optimal policy.

## Comparing Value Iteration and Policy Iteration

While both algorithms solve Markov Decision Processes, they take different operational paths:

* **Value Iteration:** Focuses entirely on calculating raw numerical values until they mathematically converge, extracting the policy only once at the very end.
* **Policy Iteration:** Focuses on the playbook. It extracts intermediate policies continuously, resulting in an algorithm that often converges in far fewer overall steps, even though the math inside each individual step is computationally heavier.

## The Bridge to Reinforcement Learning

Solving these models with Value Iteration or Policy Iteration is defined as Offline Planning. In offline planning, the AI determines all quantities purely through computation because it has perfect knowledge of the environmental details ($T$ and $R$). The AI does not actually play the game or take physical actions in the world to figure out the solution.

However, in real-world combat scenarios, the rules frequently change, or the environment is entirely unknown. When the AI cannot solve the environment through offline computation, it must transition to Online Planning, also known as Reinforcement Learning (RL).

In Reinforcement Learning, the agent is placed directly into the environment and must learn purely based on observed samples of outcomes. The agent receives feedback in the form of raw rewards, and its utility is entirely defined by that reward function. It must learn to act to maximize its expected rewards.

Because the agent learns via physical trial and error, it must navigate distinct RL challenges:

* **Exploration:** The agent must try unknown actions to gain new tactical information.
* **Exploitation:** Eventually, the agent must leverage the knowledge it has gained to maximize its score.
* **Regret:** Because the agent is learning through trial and error, it will inevitably make mistakes along the way.

## Knowledge Check & Practice Questions

1. In Policy Iteration, how does the AI agent definitively know when to stop running its loops and exit the algorithm?
2. An AI commander must calculate the exact expected utility of following an established Standard Operating Procedure (SOP). Which specific algorithm should the commander use, and why is the $\max_a$ operator removed from the math?
3. Explain the fundamental difference between Offline Planning (like Policy Iteration) and Reinforcement Learning regarding how the agent interacts with the environment.
