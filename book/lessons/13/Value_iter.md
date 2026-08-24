# Lesson 13 — Reinforcement Learning I (Value Iteration)

:::{admonition} Lesson Objectives
:class: note

* Explain value iteration.
* Apply the Bellman update.
* Interpret convergence of value functions.
:::

## Introduction to Finding Optimal Utilities

In our previous lesson, we defined what makes a policy optimal. However, knowing that an optimal policy ($\pi^*$) exists is different from actually calculating it. To navigate complex tactical environments, an AI agent must compute the optimal quantities for every possible state. 

The expected utility of starting in state $s$ and acting optimally thereafter is defined as the optimal state-value, $V^*(s)$. Similarly, the expected utility of starting in state $s$, taking a specific action $a$, and acting optimally thereafter is defined as the optimal action-value, $Q^*(s, a)$. 

## The Bellman Equations

To calculate these optimal values, we rely on a fundamental recursive relationship. The definition of optimal utility can be expressed via an expectimax recurrence, providing a simple one-step lookahead relationship. This is known as the Bellman Equation.

The optimal value of a state is mathematically defined as the maximum expected return achievable from that state across all possible actions:

$$V^*(s) = \max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]$$

### Bellman Equation Terminology

To understand the algorithm, we must break down its mathematical syntax. The table below defines each symbol and operation within a tactical context:

| Symbol / Term | Name | Plain Language Definition |
| :--- | :--- | :--- |
| **$s$** | Current State | The agent's current tactical situation or location. |
| **$a$** | Action | A specific maneuver or decision the agent can execute. |
| **$s'$** | Next State | The resulting situation after an action is taken. |
| **$V^*(s)$** | Optimal State-Value | The maximum expected reward from this state onward, assuming perfect future decisions. |
| **$Q^*(s,a)$** | Optimal Action-Value | The expected reward of taking a specific action *right now*, and then making perfect decisions afterward. |
| **$\max_a$** | Maximize over Actions | The mathematical operation of testing every possible action and choosing the one that yields the highest score. |
| **$\arg\max_a$** | Argument of the Maximum | Returns the *actual action* (the command itself) that produced the highest score, rather than the numerical score. |
| **$\sum_{s'}$** | Sum over Next States | Adding up the possibilities of all potential outcomes to calculate an average, because the environment is unpredictable (stochastic). |
| **$T(s,a,s')$** | Transition Probability | The percentage chance (0.0 to 1.0) that taking an action will actually result in the intended next state. |
| **$R(s,a,s')$** | Reward Function | The immediate tactical payoff (or penalty) received during the transition. |
| **$\gamma$** | Discount Factor | A multiplier (between 0.0 and 1.0) that makes immediate rewards mathematically more valuable than distant, uncertain future rewards. |
| **$V_k(s)$** | Value at Iteration $k$ | The agent's current estimate of a state's value, looking exactly $k$ steps into the future. |
| **$\pi^*(s)$** | Optimal Policy | The ultimate "playbook": the exact best action to take in the current state. |

## Value Iteration and Convergence

Value Iteration is a dynamic programming algorithm used to systematically compute these optimal Bellman values. 

The algorithm functions by iteratively updating its estimates of state values:
1.  **Initialization:** Start with $V_0(s) = 0$ for all states. This represents having zero time steps left, meaning the expected reward sum is zero.
2.  **The Bellman Update:** Given the vector of $V_k(s)$ values, the algorithm performs one ply of expectimax from each state to calculate the next step:
    $$V_{k+1}(s) \leftarrow \max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V_k(s')]$$
3.  **Convergence:** Repeat this update process until the values converge. 

Because the algorithm iteratively refines its approximations toward the true optimal values, it is mathematically guaranteed to converge to a unique optimal value for every state. Interestingly, the optimal policy (the actual tactical decisions) often converges long before the numerical values themselves fully stabilize.

## Policy Extraction and Evaluation

Once the optimal values ($V^*$) or action-values ($Q^*$) are calculated, the AI must translate them into an actionable playbook. 

*   **Policy Extraction:** This is the process of deriving the optimal policy implied by the calculated values. If the agent only has $V^*(s)$, it must perform a one-step mini-expectimax calculation to find the best action: 
    $\pi^*(s) = \arg\max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]$
    However, if the agent has already calculated the optimal Q-values, action selection is completely trivial: $\pi^*(s) = \arg\max_a Q^*(s,a)$.
*   **Policy Evaluation:** Sometimes an agent needs to evaluate a non-optimal, fixed policy (like a standard operating procedure). By removing the $\max_a$ operator, the Bellman update turns into a simple linear system that calculates the exact expected return of following that specific, fixed policy.

<br>
<hr width="100%" size="4" color="black">

## Knowledge Check & Practice Questions

1. Why does the Value Iteration algorithm initialize all states with a value of $V_0(s) = 0$?
<details>
 <summary>Show solution</summary>
 <div style="border: 1px solid #b3b3b3; border-radius: 6px; overflow: hidden; font-family: system-ui, sans-serif; max-width: 800px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
  <!-- Header Bar -->
  <div style="background-color: #b3b3b3; color: #000000; padding: 4px 12px; font-size: 0.95em; font-style: italic;">
    Solution
  </div>
  <!-- Body Content -->
  <div style="background-color: #fcfcfc; padding: 12px 16px; color: #111111; line-height: 1.5;">

 **Step 1:** Consider the definition of $V_k(s)$, where $k$ represents the number of time steps remaining.

 **Step 2:** At initialization ($k=0$), there are no time steps left for the agent to act.

 **Answer:** With zero time steps remaining, the agent cannot take any actions, meaning the expected sum of future rewards is zero.
  </div>
</div>

</details>
<br>

2. An autonomous routing system has successfully run Value Iteration and possesses the optimal value $V^*(s)$ for every sector on the map. To determine its next exact maneuver, does it simply look at the highest value of the adjacent sectors? Explain based on Policy Extraction.
<details>
 <summary>Show solution</summary>
 <div style="border: 1px solid #b3b3b3; border-radius: 6px; overflow: hidden; font-family: system-ui, sans-serif; max-width: 800px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
  <!-- Header Bar -->
  <div style="background-color: #b3b3b3; color: #000000; padding: 4px 12px; font-size: 0.95em; font-style: italic;">
    Solution
  </div>
  <!-- Body Content -->
  <div style="background-color: #fcfcfc; padding: 12px 16px; color: #111111; line-height: 1.5;">
**Step 1:** Recall how Policy Extraction calculates $\pi^*(s)$ from $V^*(s)$.

**Step 2:** The agent cannot just look at the raw value of the next state; it must account for transition probabilities and immediate rewards.

**Answer:** No, it is not completely trivial. The agent must perform a mini-expectimax calculation (a one-step lookahead) to weigh the transition probabilities ($T$) and immediate rewards ($R$) against the discounted future values ($\gamma V^*$) to accurately extract the optimal policy.
 </div>
</div>
</details>


3. During the execution of Value Iteration, an AI commander notices that its planned operational actions ($\pi$) stopped changing at Iteration 50, but the mathematical state values ($V$) kept incrementally changing until Iteration 120. Is this a system error?
<details>
 <summary>Show solution</summary>
 <div style="border: 1px solid #b3b3b3; border-radius: 6px; overflow: hidden; font-family: system-ui, sans-serif; max-width: 800px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
  <!-- Header Bar -->
  <div style="background-color: #b3b3b3; color: #000000; padding: 4px 12px; font-size: 0.95em; font-style: italic;">
    Solution
  </div>
  <!-- Body Content -->
  <div style="background-color: #fcfcfc; padding: 12px 16px; color: #111111; line-height: 1.5;">
 <br>
 <b>Step 1:</b> Evaluate the relationship between policy convergence and value convergence in dynamic programming.<br>
 <b>Step 2:</b> Reference the mathematical properties of the Bellman update over multiple iterations.<br>
 <b>Answer:</b> No, this is normal behavior. In Markov Decision Processes, the agent's policy will often converge and stabilize long before the exact numerical state-values mathematically converge to their final fixed points.
  </div>
</div>
</details>
<br>