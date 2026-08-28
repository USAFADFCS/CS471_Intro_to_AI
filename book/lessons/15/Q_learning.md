# Lesson 15 - Reinforcement Learning III (Q-Learning)

:::{admonition} Lesson Objectives
:class: note
* Explain Q-learning and Q-values
* Compare model-based and model-free learning
* Interpret learning from experience
:::

## Reinforcement Learning in Tactical Environments

When deploying autonomous agents—such as unmanned aerial vehicles (UAVs) navigating mountainous terrain or cyber-defense algorithms securing a 5 Gbps tactical network—the environment is rarely fully known in advance. Agents must learn optimal policies through direct interaction, receiving feedback in the form of rewards.

### Model-Based vs. Model-Free Learning

In offline planning, all parameters of the Markov Decision Process (MDP) are known. In online learning, the agent must experiment.

*   **{term}`Model-Based Learning`**: The agent estimates the transition probabilities and reward functions from observed samples. Once the model is approximated, the agent solves for the optimal policy using techniques like value iteration. Think of this as a UAV mapping a hostile radar environment before planning its permanent route.
*   **{term}`Model-Free Learning`**: The agent directly estimates the values (or Q-values) of states without constructing an explicit model of the transitions or rewards. This is highly effective for rapidly shifting electronic warfare environments where storing transition probabilities is too computationally expensive.

### Learning from Experience: Temporal Difference

{term}`Temporal Difference Learning` (TD) updates the value of a state each time the agent experiences a transition. The agent moves its running average value toward the value of whatever successor state actually occurs. 

### Q-Learning and Exploration

{term}`Q-Learning` is an active reinforcement learning method where the agent learns the optimal policy by sampling Q-value iteration. Q-learning is an off-policy algorithm, meaning it converges to the optimal policy even if the agent is acting suboptimally during the learning phase. 

To learn effectively, the agent must balance:
*   **Exploration:** Trying unknown actions to gather intelligence. A common method is $\epsilon$-greedy, where the agent acts randomly with a small probability $\epsilon$.
*   **Exploitation:** Utilizing known information to maximize rewards.

```{mermaid}
graph LR
    S((Base)) -- "Deploy UAV" --> A((Waypoint Alpha))
    A -- "Safe Route" --> B((Target))
    A -- "Radar Jamming" --> C((Hostile Zone))
    C -- "Evasive Maneuver" --> B
    classDef default fill:#f9f9f9,stroke:#888,stroke-width:2px,color:#000;
    classDef highlight fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    linkStyle default stroke:#a0a0a0,stroke-width:2px;
```
