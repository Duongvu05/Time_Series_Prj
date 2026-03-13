# Q2: Iterative Policy Evaluation – 4×4 Grid World

**Source:** Silver Lecture 3, pages 10–11  
**Setup:** Undiscounted ($\gamma=1$), uniform random policy (prob = $1/4$ each direction), $R = -1$ per step.

---

## Grid Layout

```
 T   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14   T
```

States 0 (top-left) and 15 (bottom-right) are **terminal** (T), $V = 0$ always.

---

---

## Algorithm: Iterative Policy Evaluation

Starting from $V_{0}(s) = 0$ for all $s$, apply repeatedly:

### 1. Synchronous (Two-sweep) Updates
Uses two arrays to store old and new values:
$$V_{k+1}(s) = \sum_{a \in \{N,S,W,E\}} \frac{1}{4} \bigl[R + \gamma\,V_{k}(s'(s,a))\bigr]$$

### 2. In-place (Single-sweep) Updates
Uses a single array and updates values immediately:
$$V(s) \leftarrow \sum_{a \in \{N,S,W,E\}} \frac{1}{4} \bigl[R + \gamma\,V(s'(s,a))\bigr]$$
In-place updates usually **converge faster** because they use the most recent information available within the same sweep.

---

## Case $k = 1$
...
(Calculations omitted for brevity as they remain the same for $V_1$)

---

## Case $k = 2$ (Synchronous)
...
(Calculations remain the same)

---

## Convergence as $k \to \infty$

```python
# Synchronous convergence
V_conv, k = run_to_convergence(theta=1e-6)

# In-place convergence (faster)
V_conv_inp, k_inp = run_to_convergence_inplace(theta=1e-6)
```

The sequence $V_{k}$ converges to $V^{\pi}$, the **true value function under the random policy**. Key observations:

1. **Monotone decrease:** $V_{k}(s)$ is non-increasing in $k$ for all $s$.
2. **Convergence rate:** In-place updates typically require fewer sweeps to reach the same $\theta$.
3. **Limit $V^{\pi}$**: Represents the expected total reward for a random-walk agent. Cells near terminals have $V$ closer to $0$; cells far away have the most negative values ($\approx -14$).

**The greedy policy from $V^{\pi}$ always points towards the nearest terminal,** which is the intuitive optimal policy.

---

## Running the Code

```python
from src.q2_grid_world.grid_world import (
    iterative_policy_eval, 
    run_to_convergence,
    run_to_convergence_inplace
)

# Run standard iterative evaluation
V_inf, steps = run_to_convergence()
print(f"Synchronous converged in {steps} sweeps")

# Run in-place evaluation
V_inf_inp, steps_inp = run_to_convergence_inplace()
print(f"In-place converged in {steps_inp} sweeps")
```
