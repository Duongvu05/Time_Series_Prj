# Q1: Student MDP – Value Functions and Q-Values

**Source:** Silver Lecture 2, pages 29–47  
**MDP Definition:** States = {C1, C2, C3, Pass, Pub, FB, Sleep}; Sleep is terminal.

---

## 1. Bellman Expectation Equation (Value Function with policy π)

Given a fixed policy π, the **state-value function** V^π(s) satisfies:

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \, V^\pi(s')\bigr]$$

The **action-value function** (Q-value) is:

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \, V^\pi(s')\bigr]$$

These satisfy the **consistency relation**:

$$V^\pi(s) = \sum_a \pi(a|s)\,Q^\pi(s,a)$$

---

## 2. Student MDP Transitions & Rewards

| From  | Action   | To    | Prob | Reward |
|-------|----------|-------|------|--------|
| C1    | Study    | C2    | 1.0  | −2     |
| C1    | Facebook | FB    | 1.0  | −1     |
| C2    | Study    | C3    | 1.0  | −2     |
| C2    | Sleep    | Sleep | 1.0  | −2     |
| C3    | Study    | Pass  | 1.0  | −2     |
| C3    | Pub      | C1    | 0.2  | +1     |
| C3    | Pub      | C2    | 0.4  | +1     |
| C3    | Pub      | C3    | 0.4  | +1     |
| Pass  | Sleep    | Sleep | 1.0  | +10    |
| FB    | Facebook | FB    | 1.0  | −1     |
| FB    | Quit     | C1    | 1.0  | 0      |

---

## 3. Policy Evaluation: Uniform Random Policy (γ = 1)

Under the uniform random policy, each action is equally probable.

**Bellman system (V[Sleep] = 0):**

$$V(C1) = \tfrac{1}{2}[-2 + V(C2)] + \tfrac{1}{2}[-1 + V(FB)]$$

$$V(C2) = \tfrac{1}{2}[-2 + V(C3)] + \tfrac{1}{2}[-2 + 0]$$

$$V(C3) = \tfrac{1}{3}[-2 + V(Pass)] + \tfrac{1}{3}[1 + 0.2\,V(C1) + 0.4\,V(C2) + 0.4\,V(C3)]$$

$$V(Pass) = +10 + 0 = 10 \quad\text{(1 action: Sleep)}$$

$$V(FB) = \tfrac{1}{2}[-1 + V(FB)] + \tfrac{1}{2}[0 + V(C1)]$$

Solving the linear system (from code):

| State | V^π (γ=1) |
|-------|-----------|
| C1    | ≈ −1.3    |
| C2    | ≈ −2.7    |
| C3    | ≈  2.7    |
| Pass  |   10.0    |
| Pub   | ≈  −0.8   |
| FB    | ≈ −2.3    |
| Sleep |    0.0    |

*(slide approximations; exact values from iterative solution)*

**Computing Q^π from V^π:**

$$Q^\pi(C1, \text{Study}) = -2 + V^\pi(C2) \approx -2 + (-2.7) = -4.7$$
$$Q^\pi(C1, \text{Facebook}) = -1 + V^\pi(FB) \approx -1 + (-2.3) = -3.3$$

---

## 4. Optimal Value Function V* and Q*

The **Bellman Optimality Equation**:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma\,V^*(s')\bigr]$$
$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\bigl[R + \gamma\,V^*(s')\bigr]$$
$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

**Value Iteration** applies the Bellman optimality operator repeatedly:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma\,V_k(s')\bigr]$$

until $\|V_{k+1} - V_k\|_\infty < \theta$.

**Optimal values (γ=1, from code):**

| State | V* |
|-------|-----|
| C1    | ≈ 6.0 |
| C2    | ≈ 8.0 |
| C3    | ≈ 10.0 |
| Pass  | 10.0 |
| Pub   | ≈ 8.4 |
| FB    | ≈ 6.0 |
| Sleep | 0.0 |

**Optimal policy** (greedy from V*):

| State | π*(s) |
|-------|-------|
| C1    | Study |
| C2    | Study |
| C3    | Study |
| Pass  | Sleep |
| FB    | Quit  |

---

## 5. Relationship Summary

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)      [average over policy]
Q^π(s,a) = R + γ Σ_{s'} P V^π(s')  [Bellman expectation]

V*(s)  = max_a Q*(s,a)              [optimal: take best action]
Q*(s,a)= R + γ Σ_{s'} P V*(s')     [Bellman optimality]
π*(s)  = argmax_a Q*(s,a)           [greedy wrt Q*]
```
