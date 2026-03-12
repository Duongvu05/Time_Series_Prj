# Q2: Iterative Policy Evaluation – 4×4 Grid World

**Source:** Silver Lecture 3, pages 10–11  
**Setup:** Undiscounted (γ=1), uniform random policy (prob = 1/4 each direction), R = −1 per step.

---

## Grid Layout

```
 T   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14   T
```

States 0 (top-left) and 15 (bottom-right) are **terminal** (T), V = 0 always.

---

## Algorithm: Iterative Policy Evaluation

Starting from V₀(s) = 0 for all s, apply repeatedly:

$$V_{k+1}(s) = \sum_{a \in \{N,S,W,E\}} \frac{1}{4} \bigl[R + \gamma\,V_k(s'(s,a))\bigr]$$

where movements that would leave the grid **reflect** (agent stays in same cell).

---

## Case k = 1

After the **first sweep** from V₀ = 0:

For every non-terminal state s:
- Each of 4 actions gives reward R = −1 and lands on some s' with V₀(s') = 0.
- So: V₁(s) = (1/4)(−1 + 0) × 4 = **−1.0** for all non-terminal states.

```
  0.0  -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0   0.0
```

---

## Case k = 2

Using V₁ (all non-terminal = −1, terminals = 0), we compute V₂.

**State 1** (row 0, col 1):
- N → state 1 (hits top wall, stays): R + V₁(1) = −1 + (−1) = −2
- S → state 5: R + V₁(5) = −1 + (−1) = −2
- W → state 0 (terminal): R + V₁(0) = −1 + 0 = −1
- E → state 2: R + V₁(2) = −1 + (−1) = −2

V₂(1) = (1/4)(−2 − 2 − 1 − 2) = **(1/4)(−7) = −1.75**

**State 0 (corner, terminal):** V₂(0) = 0

**State 5** (interior):
- N → 1: −1+(−1)=−2; S → 9: −2; W → 4: −2; E → 6: −2

V₂(5) = (1/4)(−8) = **−2.0**

**State 15 (terminal):** V₂(15) = 0

**State 14** (row 3, col 2):
- S → 14 (wall): −1+(−1)=−2; N → 10: −2; W → 13: −2; E → 15: −1+0=−1

V₂(14) = (1/4)(−2 − 2 − 2 − 1) = **−1.75**

Full V₂ grid (by symmetry and computation):
```
  0.00  -1.75  -2.00  -2.00
 -1.75  -2.00  -2.00  -2.00
 -2.00  -2.00  -2.00  -1.75
 -2.00  -2.00  -1.75   0.00
```

---

## Case k = 3

Applying the same formula to V₂:

**State 1**:
- N → 1: −1+V₂(1)=−1+(−1.75)=−2.75
- S → 5: −1+(−2.00)=−3.00
- W → 0: −1+0=−1.00
- E → 2: −1+(−2.00)=−3.00

V₃(1) = (1/4)(−2.75 − 3.00 − 1.00 − 3.00) = (1/4)(−9.75) ≈ **−2.4375**

For interior **state 5**:
- N → 1: −1+(−1.75); S → 9: −1+(−2.00); W → 4: −1+(−1.75); E → 6: −1+(−2.00)

V₃(5) = (1/4)(−2.75 − 3.00 − 2.75 − 3.00) = (1/4)(−11.5) = **−2.875**

---

## Convergence as k → ∞

```python
V_conv, k = run_to_convergence(theta=1e-6)
```

The sequence V_k converges to V^π, the **true value function under the random policy**. Key observations:

1. **Monotone decrease:** V_k(s) is non-increasing in k for all s (values become more negative as the agent "learns" how bad random walk is).
2. **Convergence rate:** Since γ = 1, convergence is not guaranteed in general, but the grid world is **episodic** (absorbing terminals), so the operator T^π is still a contraction in practice.
3. **Limit V^π**: Represents the expected total reward for a random-walk agent. Cells near terminals have V closer to 0; cells far away have the most negative values (≈ −14).

**The greedy policy from V^π always points towards the nearest terminal,** which is the intuitive optimal policy.

---

## Running the Code

```python
from src.q2_grid_world.grid_world import iterative_policy_eval, run_to_convergence

for k in [1, 2, 3]:
    Vk = iterative_policy_eval(k)
    # inspect Vk.reshape(4, 4)

V_inf, steps = run_to_convergence()
print(f"Converged in {steps} sweeps")
```
