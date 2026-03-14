# Q1: Student MDP – Value Functions and Q-Values

**Source:** Câu 1 - ML2.pdf
**MDP Definition:** States = {C1, C2, C3, FB, Sleep}; Sleep is terminal.

---

## 1. Bellman Expectation Equation (Value Function with policy π)

Given a fixed policy $\pi$, the **state-value function** $V^\pi(s)$ satisfies the Bellman expectation equation:

$$V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \, V^{\pi}(s')\bigr]$$

The **action-value function** (Q-value) is:

$$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \, V^{\pi}(s')\bigr]$$

Relationship: $V^{\pi}(s) = \sum_a \pi(a|s)\,Q^{\pi}(s,a)$.

---

## 2. Student MDP Definition (PDF Version)

**States:** {C1, C2, C3, FB} are non-terminal. {Sleep} is terminal.

| From  | Action   | To    | Prob | Reward |
|-------|----------|-------|------|--------|
| C1    | Study    | C2    | 1.0  | −2     |
| C1    | Facebook | FB    | 1.0  | −1     |
| C2    | Study    | C3    | 1.0  | −2     |
| C2    | Sleep    | Sleep | 1.0  | 0      |
| C3    | Study    | Sleep | 1.0  | +10    |
| C3    | Pub      | C1    | 0.2  | +1     |
| C3    | Pub      | C2    | 0.4  | +1     |
| C3    | Pub      | C3    | 0.4  | +1     |
| FB    | Facebook | FB    | 1.0  | −1     |
| FB    | Quit     | C1    | 1.0  | 0      |

---

## 3. Policy Evaluation: Uniform Random Policy ($\gamma = 1, \pi(a|s) = 0.5$)

Each decision state has 2 actions, so $\pi(a|s) = 0.5$.

**Bellman Equations:**

1.  $v(F) = 0.5 [-1 + v(F)] + 0.5 [0 + v(C1)] \implies v(F) = v(C1) - 1$
2.  $v(C1) = 0.5 [-1 + v(F)] + 0.5 [-2 + v(C2)] \implies v(C1) = v(C2) - 4$
3.  $v(C2) = 0.5 [-2 + v(C3)] + 0.5 [0 + v(S)] \implies v(C2) = 0.5 v(C3) - 1$
4.  $v(C3) = 0.5 [10 + v(S)] + 0.5 [1 + 0.2 v(C1) + 0.4 v(C2) + 0.4 v(C3)]$

**Calculated Values ($\gamma=1$):**

| State | $V^\pi$ | Action-Value Function ($Q^\pi$) |
|-------|---------|--------------------------------|
| Facebook | $-2.3$ | Facebook: $-3.3$, Quit: $-1.3$ |
| C1    | $-1.3$    | Facebook: $-3.3$, Study: $0.7$  |
| C2    | $2.7$     | Sleep: $0$, Study: $5.4$        |
| C3    | $7.4$     | Study: $10$, Pub: $4.78$        |
| Sleep | $0.0$     | -                              |

---

## 4. Optimal Value Function $V^{\ast}$ and $Q^{\ast}$

$V^{\ast}(s) = \max_a Q^{\ast}(s,a) = \max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma\,V^{\ast}(s')\bigr]$.

**Optimal Values (γ=1):**

| State | $V^{\ast}$ | $\pi^{\ast}(s)$ | Reasoning |
|-------|-------|-------|-----------|
| Facebook | $6.0$ | Quit | $Q^*(F, \text{Quit}) = 0 + 6 = 6$; $Q^*(F, F) = -1 + 6 = 5$. |
| C1    | $6.0$   | Study | $Q^*(C1, \text{Study}) = -2 + 8 = 6$; $Q^*(C1, FB) = -1 + 6 = 5$. |
| C2    | $8.0$   | Study | $Q^*(C2, \text{Study}) = -2 + 10 = 8$; $Q^*(C2, \text{Sleep}) = 0$. |
| C3    | $10.0$  | Study | $Q^*(C3, \text{Study}) = 10 + 0 = 10$; $Q^*(C3, \text{Pub}) = 1 + 0.2(6) + 0.4(8) + 0.4(10) = 9.4$. |
| Sleep | 0.0   | Stay  | Terminal state. |

---

## 5. Relationship Summary

$$V^{\pi}(s) = \sum_{a} \pi(a|s) Q^{\pi}(s,a) \quad [\text{average over policy}]$$
$$Q^{\pi}(s,a) = R + \gamma \sum_{s'} P(s'|s,a) V^{\pi}(s') \quad [\text{Bellman expectation}]$$

$$V^{\ast}(s) = \max_{a} Q^{\ast}(s,a) \quad [\text{optimal: take best action}]$$
$$Q^{\ast}(s,a) = R + \gamma \sum_{s'} P(s'|s,a) V^{\ast}(s') \quad [\text{Bellman optimality}]$$
$$\pi^{\ast}(s) = \arg\max_{a} Q^{\ast}(s,a) \quad [\text{greedy wrt } Q^{\ast}]$$
