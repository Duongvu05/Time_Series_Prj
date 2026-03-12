# Q3: Jack's Car Rental – Policy Iteration

**Source:** Silver Lecture 3, pages 14–15

---

## Problem Setup

Jack manages two car rental locations. Each morning, customers arrive and rent cars (if available), and Jack earns **$10/car rented**. Overnight, Jack can move cars between locations (up to **5/night**, at **$2/car**).

| Parameter | Description | Value |
|-----------|-------------|-------|
| `MAX_CARS` | Max cars per location | 20 |
| `MAX_MOVE` | Max cars moved overnight | 5 |
| Rental reward | Per rental | +$10 |
| Move cost | Per car moved | -$2 |
| $\lambda_{\text{rent}_{1}}$ | Poisson rate, rentals at loc 1 | 3 |
| $\lambda_{\text{rent}_{2}}$ | Poisson rate, rentals at loc 2 | 4 |
| $\lambda_{\text{ret}_{1}}$  | Poisson rate, returns at loc 1 | 3 |
| $\lambda_{\text{ret}_{2}}$  | Poisson rate, returns at loc 2 | 2 |
| $\gamma$ | Discount factor | 0.9 |

---

## MDP Formulation

**State:** $s = (n_{1}, n_{2})$ — cars available at start of day at each location.  
**Action:** $a \in \{-5, \dots, +5\}$ — cars moved from loc2 → loc1 (negative = opposite).  
**Transition:** Stochastic (Poisson demands and returns).

### Bellman Expectation Equation

$$V^{\pi}(n_{1}, n_{2}) = \sum_{a} \pi(a|s) \Bigl[ -2|a| + \mathbb{E}_{\text{req}_{1}, \text{req}_{2}, \text{ret}_{1}, \text{ret}_{2}}\bigl[10(\text{rent}_{1} + \text{rent}_{2}) + \gamma\,V^{\pi}(n_{1}', n_{2}')\bigr] \Bigr]$$

where:
- $\text{rent}_{i} = \min(\text{req}_{i}, n_{i}')$ (can't rent more than available)
- $n_{i}' = \min(n_{i} \pm a, 20)$ after movement
- $n_{i}^{\text{end}} = \min(n_{i}' - \text{rent}_{i} + \text{ret}_{i}, 20)$

---

## Precomputed Transition Matrix (Key Optimisation)

Instead of summing over Poisson samples at every iteration, we precompute for each location:

$$\text{ExpRew}_{i}[n] = \sum_{req=0}^{\infty} P(req;\lambda_{i}) \cdot \min(req, n) \cdot 10$$

$$\text{Trans}_{i}[n, n_{\text{next}}] = \sum_{req} \sum_{ret} P(req;\lambda_{i}) \cdot P(ret;\lambda_{i}^{\text{ret}}) \cdot \mathbf{1}[\min(n-req^{+}, 0)+ret = n_{\text{next}}]$$

Then the expected future value for state $(n_{1}, n_{2})$ reduces to:

$$\mathbb{E}[V(n_{1}', n_{2}')] = (\text{Trans}_{1}[n_{1}, :]) \cdot V \cdot (\text{Trans}_{2}[n_{2}, :])^{\top}$$

This is a **single matrix-vector product per state transition**, making evaluation fast.

---

## Policy Iteration

### Step 1: Policy Evaluation

Iterate Bellman expectation until $||\Delta V|| < \theta$:

$$V_{k+1}(s) = -2|a| + \text{ExpRew}_{1}[n_{1}'] + \text{ExpRew}_{2}[n_{2}'] + \gamma \cdot \text{Trans}_{1}[n_{1}',:] \cdot V_{k} \cdot \text{Trans}_{2}[n_{2}',:]^{\top}$$

### Step 2: Policy Improvement

$$\pi'(s) = \arg\max_{a \in [-5,5]} Q(s,a)$$

Repeat until policy is stable.

---

## Expected Results (matching Silver slides page 15)

After policy iteration converges (~4 iterations):
- The optimal policy forms a characteristic pattern where Jack moves cars from the richer (more-returning) location to the one with higher demand.
- Policy roughly: move $\approx 1$–$2$ cars from loc1→loc2 when loc1 is crowded, or from loc2→loc1 otherwise.
- $V^{\ast}$ surface is smooth and increases with total cars available.

---

## Running

```bash
uv run python -m src.q3_jacks_car.jacks_car
```

This will:
1. Precompute transition matrices (one-time, ~5 seconds)
2. Run policy iteration (~3–5 iterations to convergence)
3. Plot policy heatmap and value surface (saved to `notebooks/jacks_car_policy.png`)
