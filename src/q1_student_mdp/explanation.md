# Q1: Student MDP – Value Functions and Q-Values

**Source:** Silver Lecture 2, pages 29–47  
**MDP Definition:** States = {C1, C2, C3, Pass, Pub, FB, Sleep}; Sleep is terminal.

---

## 1. Bellman Expectation Equation (Value Function with policy π)

Given a fixed policy $\pi$, the **state-value function** $V^\pi(s)$ satisfies the Bellman expectation equation:

$$V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \, V^{\pi}(s')\bigr]$$

The **action-value function** (Q-value) is:

$$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \, V^{\pi}(s')\bigr]$$

Relationship: $V^{\pi}(s) = \sum_a \pi(a|s)\,Q^{\pi}(s,a)$.

---

## 2. Student MDP Definition (Slide 29 Version)

**States:** {C1, C2, C3, Pass, FB} are non-terminal. {Sleep} is terminal.

| From  | Action   | To    | Prob | Reward |
|-------|----------|-------|------|--------|
| C1    | Study    | C2    | 1.0  | −2     |
| C1    | Facebook | FB    | 1.0  | −1     |
| C2    | Study    | C3    | 1.0  | −2     |
| C2    | Sleep    | Sleep | 1.0  | 0      |
| C3    | Study    | Pass  | 1.0  | −2     |
| C3    | Pub      | C1    | 0.2  | +1     |
| C3    | Pub      | C2    | 0.4  | +1     |
| C3    | Pub      | C3    | 0.4  | +1     |
| Pass  | Sleep    | Sleep | 1.0  | +10    |
| FB    | Facebook | FB    | 1.0  | −1     |
| FB    | Quit     | C1    | 1.0  | 0      |

---

## 3. Policy Evaluation: Uniform Random Policy ($\gamma = 1, \pi(a|s) = 0.5$)

Each decision state has 2 actions, so $\pi(a|s) = 0.5$.

**Bellman Equations:**

1.  $V(C_1) = 0.5 \underbrace{[-2 + V(C_2)]}_{Q(C_1, \text{Study})} + 0.5 \underbrace{[-1 + V(\text{FB})]}_{Q(C_1, \text{FB})}$
2.  $V(C_2) = 0.5 \underbrace{[-2 + V(C_3)]}_{Q(C_2, \text{Study})} + 0.5 \underbrace{[0 + 0]}_{Q(C_2, \text{Sleep})}$
3.  $V(C_3) = 0.5 \underbrace{[-2 + V(\text{Pass})]}_{Q(C_3, \text{Study})} + 0.5 \underbrace{[1 + 0.2 V(C_1) + 0.4 V(C_2) + 0.4 V(C_3)]}_{Q(C_3, \text{Pub})}$
4.  $V(\text{FB}) = 0.5 \underbrace{[-1 + V(\text{FB})]}_{Q(\text{FB}, \text{FB})} + 0.5 \underbrace{[0 + V(C_1)]}_{Q(\text{FB}, \text{Quit})}$
5.  $V(\text{Pass}) = +10 + 0 = 10 \quad$ (1 action: Sleep)

**Calculated Values ($\gamma=1$):**

| State | $V^\pi$ | Best Action ($Q^\pi$) |
|-------|---------|-------------|
| C1    | -2.08   | Study (-0.08) |
| C2    | 1.92    | Study (3.85) |
| C3    | 5.85    | Study (8.00) |
| Pass  | 10.00   | Sleep (10.0) |
| FB    | -3.08   | Quit (-2.08) |

---

## 4. Optimal Value Function $V^{\ast}$ and $Q^{\ast}$

$V^{\ast}(s) = \max_a Q^{\ast}(s,a) = \max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma\,V^{\ast}(s')\bigr]$.

**Optimal Values (γ=1):**

| State | $V^{\ast}$ | $\pi^{\ast}(s)$ | Reasoning |
|-------|-------|-------|-----------|
| C1    | 4.0   | Study | $Q^*(C1, \text{Study}) = -2 + 6 = 4$; $Q^*(C1, \text{FB}) = -1 + 4 = 3$. |
| C2    | 6.0   | Study | $Q^*(C2, \text{Study}) = -2 + 8 = 6$; $Q^*(C2, \text{Sleep}) = 0$. |
| C3    | 8.0   | Study | $Q^*(C3, \text{Study}) = -2 + 10 = 8$; $Q^*(\text{Pub}) \approx 7.4$. |
| Pass  | 10.0  | Sleep | Only one action. |
| FB    | 4.0   | Quit  | $Q^*(FB, \text{Quit}) = 0 + 4 = 4$; $Q^*(FB, FB) = -1 + 4 = 3$. |

---

## 5. Relationship Summary

$$V^{\pi}(s) = \sum_{a} \pi(a|s) Q^{\pi}(s,a) \quad [\text{average over policy}]$$
$$Q^{\pi}(s,a) = R + \gamma \sum_{s'} P(s'|s,a) V^{\pi}(s') \quad [\text{Bellman expectation}]$$

$$V^{\ast}(s) = \max_{a} Q^{\ast}(s,a) \quad [\text{optimal: take best action}]$$
$$Q^{\ast}(s,a) = R + \gamma \sum_{s'} P(s'|s,a) V^{\ast}(s') \quad [\text{Bellman optimality}]$$
$$\pi^{\ast}(s) = \arg\max_{a} Q^{\ast}(s,a) \quad [\text{greedy wrt } Q^{\ast}]$$
