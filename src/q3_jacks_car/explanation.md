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
| Move cost | Per car moved | −$2 |
| λ_rent₁ | Poisson rate, rentals at loc 1 | 3 |
| λ_rent₂ | Poisson rate, rentals at loc 2 | 4 |
| λ_ret₁  | Poisson rate, returns at loc 1 | 3 |
| λ_ret₂  | Poisson rate, returns at loc 2 | 2 |
| γ | Discount factor | 0.9 |

---

## MDP Formulation

**State:** s = (n₁, n₂) — cars available at start of day at each location.  
**Action:** a ∈ {−5, …, +5} — cars moved from loc2 → loc1 (negative = opposite).  
**Transition:** Stochastic (Poisson demands and returns).

### Bellman Expectation Equation

$$V^\pi(n_1, n_2) = \sum_{a} \pi(a|s) \Bigl[ -2|a| + \mathbb{E}_{req_1, req_2, ret_1, ret_2}\bigl[10(rent_1 + rent_2) + \gamma\,V^\pi(n_1', n_2')\bigr] \Bigr]$$

where:
- $rent_i = \min(req_i, n_i')$ (can't rent more than available)
- $n_i' = \min(n_i \pm a, 20)$ after movement
- $n_i^{end} = \min(n_i' - rent_i + ret_i, 20)$

---

## Precomputed Transition Matrix (Key Optimisation)

Instead of summing over Poisson samples at every iteration, we precompute for each location:

$$\text{EXP\_REW}_i[n] = \sum_{req=0}^{\infty} P(req;\lambda_i) \cdot \min(req, n) \cdot 10$$

$$\text{TRANS}_i[n, n_{next}] = \sum_{req} \sum_{ret} P(req;\lambda_i) \cdot P(ret;\lambda_i^{ret}) \cdot \mathbf{1}[\min(n-req^+, 0)+ret = n_{next}]$$

Then the expected future value for state (n₁, n₂) reduces to:

$$\mathbb{E}[V(n_1', n_2')] = (\text{TRANS}_1[n_1, :]) \cdot V \cdot (\text{TRANS}_2[n_2, :])^\top$$

This is a **single matrix-vector product per state transition**, making evaluation fast.

---

## Policy Iteration

### Step 1: Policy Evaluation

Iterate Bellman expectation until $\|\Delta V\| < \theta$:

$$V_{k+1}(s) = -2|a| + \text{EXP\_REW}_1[n_1'] + \text{EXP\_REW}_2[n_2'] + \gamma \cdot \text{TRANS}_1[n_1',:] \cdot V_k \cdot \text{TRANS}_2[n_2',:]^\top$$

### Step 2: Policy Improvement

$$\pi'(s) = \arg\max_{a \in [-5,5]} Q(s,a)$$

Repeat until policy is stable.

---

## Expected Results (matching Silver slides page 15)

After policy iteration converges (~4 iterations):
- The optimal policy forms a characteristic pattern where Jack moves cars from the richer (more-returning) location to the one with higher demand.
- Policy roughly: move ~1–2 cars from loc1→loc2 when loc1 is crowded, or from loc2→loc1 otherwise.
- V* surface is smooth and increases with total cars available.

---

## Running

```bash
uv run python -m src.q3_jacks_car.jacks_car
```

This will:
1. Precompute transition matrices (one-time, ~5 seconds)
2. Run policy iteration (~3–5 iterations to convergence)
3. Plot policy heatmap and value surface (saved to `notebooks/jacks_car_policy.png`)
