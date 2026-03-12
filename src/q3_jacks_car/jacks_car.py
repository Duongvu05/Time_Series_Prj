"""
Q3: Jack's Car Rental – Policy Iteration.

Silver Lecture 3, pages 14–15.

Problem Parameters (from slides):
- Max cars at each location: MAX_CARS = 20
- Max cars moved per night: MAX_MOVE = 5
- Rental reward: +$10 per car rented
- Move cost: -$2 per car moved
- Rental requests: Poisson(λ1=3) at loc1, Poisson(λ2=4) at loc2
- Returns:         Poisson(λ1=3) at loc1, Poisson(λ2=2) at loc2
- discount γ = 0.9

Key optimisation: precompute the transition / expected-reward arrays
so policy evaluation is pure matrix operations (fast NumPy).
"""

from __future__ import annotations

import itertools
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------

MAX_CARS = 20        # max cars at each location
MAX_MOVE = 5         # max cars transferred overnight
RENTAL_REWARD = 10.0
MOVE_COST = 2.0
GAMMA = 0.9

# Poisson parameters
LAMBDA_RENT_1 = 3
LAMBDA_RENT_2 = 4
LAMBDA_RETURN_1 = 3
LAMBDA_RETURN_2 = 2

# Truncate Poisson at this value (prob of seeing > POISSON_UPPER is negligible)
POISSON_UPPER = 11


# ---------------------------------------------------------------------------
# Poisson helper (cached)
# ---------------------------------------------------------------------------

@cache
def poisson_pmf(lam: int, n: int) -> float:
    """P(X = n) for X ~ Poisson(lam), truncated representation."""
    return float(poisson.pmf(n, lam))


def _poisson_probs(lam: int) -> NDArray[np.float64]:
    """Array of P(X=k) for k = 0, 1, ..., POISSON_UPPER."""
    return np.array([poisson_pmf(lam, k) for k in range(POISSON_UPPER + 1)])


# ---------------------------------------------------------------------------
# Precompute expected reward + transition probability for one location
# ---------------------------------------------------------------------------

def _location_dynamics(
    lam_rent: int,
    lam_return: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    For a single location, precompute:
      - expected_reward[n]          : E[reward | starting cars = n]
      - trans[n, n_next]            : P(n_cars_end_of_day = n_next | n_start = n)

    where n, n_next ∈ {0, ..., MAX_CARS}.
    """
    rent_probs = _poisson_probs(lam_rent)
    ret_probs = _poisson_probs(lam_return)

    expected_reward = np.zeros(MAX_CARS + 1, dtype=np.float64)
    trans = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float64)

    for n in range(MAX_CARS + 1):
        for req in range(POISSON_UPPER + 1):
            p_rent = rent_probs[req]
            actual_rent = min(req, n)
            cars_after_rent = n - actual_rent

            expected_reward[n] += p_rent * actual_rent * RENTAL_REWARD

            for ret in range(POISSON_UPPER + 1):
                p_ret = ret_probs[ret]
                n_next = min(cars_after_rent + ret, MAX_CARS)
                trans[n, n_next] += p_rent * p_ret

    return expected_reward, trans


logger.info("Precomputing location dynamics (one-time cost)...")
_EXP_REW_1, _TRANS_1 = _location_dynamics(LAMBDA_RENT_1, LAMBDA_RETURN_1)
_EXP_REW_2, _TRANS_2 = _location_dynamics(LAMBDA_RENT_2, LAMBDA_RETURN_2)
logger.info("Location dynamics precomputed.")


# ---------------------------------------------------------------------------
# Bellman value for a given action (vectorised over states)
# ---------------------------------------------------------------------------

def _bellman_q(
    V: NDArray[np.float64],
    action: int,
) -> NDArray[np.float64]:
    """
    Compute Q(s, action) for all states s = (n1, n2).

    action ∈ {-MAX_MOVE, ..., MAX_MOVE}
    positive = move from loc2 → loc1, negative = move from loc1 → loc2.

    Returns array of shape (MAX_CARS+1, MAX_CARS+1).
    """
    Q = np.full((MAX_CARS + 1, MAX_CARS + 1), -np.inf, dtype=np.float64)

    for n1, n2 in itertools.product(range(MAX_CARS + 1), repeat=2):
        moved = action
        # Check feasibility
        if moved > 0:
            moved = min(moved, n2)  # can't move more than available at loc2
        else:
            moved = -min(-moved, n1)  # can't move more than available at loc1

        n1_start = max(0, min(n1 + moved, MAX_CARS))
        n2_start = max(0, min(n2 - moved, MAX_CARS))

        move_cost = abs(moved) * MOVE_COST

        # Expected immediate reward
        r = _EXP_REW_1[n1_start] + _EXP_REW_2[n2_start] - move_cost

        # Expected future value (transition is independent across locations)
        # V_next = sum_{n1', n2'} P1(n1'|n1_start) P2(n2'|n2_start) V(n1', n2')
        # = (P1[n1_start, :] ⊗ P2[n2_start, :]) · V
        v_next = float(_TRANS_1[n1_start, :] @ V @ _TRANS_2[n2_start, :])

        Q[n1, n2] = r + GAMMA * v_next

    return Q


# ---------------------------------------------------------------------------
# Policy Evaluation
# ---------------------------------------------------------------------------

def policy_evaluation(
    policy: NDArray[np.int64],
    V: NDArray[np.float64] | None = None,
    theta: float = 1e-2,
    max_iter: int = 1000,
) -> NDArray[np.float64]:
    """
    Iterative policy evaluation.

    policy: (MAX_CARS+1, MAX_CARS+1) int array, values in [-MAX_MOVE, MAX_MOVE]
    """
    if V is None:
        V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float64)

    for it in range(max_iter):
        V_new = np.zeros_like(V)

        for n1, n2 in itertools.product(range(MAX_CARS + 1), repeat=2):
            a = int(policy[n1, n2])
            moved = a
            if moved > 0:
                moved = min(moved, n2)
            else:
                moved = -min(-moved, n1)

            n1_start = max(0, min(n1 + moved, MAX_CARS))
            n2_start = max(0, min(n2 - moved, MAX_CARS))
            move_cost = abs(moved) * MOVE_COST

            r = _EXP_REW_1[n1_start] + _EXP_REW_2[n2_start] - move_cost
            v_next = float(_TRANS_1[n1_start, :] @ V @ _TRANS_2[n2_start, :])
            V_new[n1, n2] = r + GAMMA * v_next

        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        logger.debug("[eval iter {:3d}] delta = {:.4f}", it + 1, delta)
        if delta < theta:
            break

    return V


# ---------------------------------------------------------------------------
# Policy Improvement
# ---------------------------------------------------------------------------

def policy_improvement(
    V: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Greedy policy: pi'(s) = argmax_a Q(s, a)."""
    Q_best = np.full((MAX_CARS + 1, MAX_CARS + 1), -np.inf, dtype=np.float64)
    policy_new = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.int64)

    for action in range(-MAX_MOVE, MAX_MOVE + 1):
        Q_a = _bellman_q(V, action)
        mask = Q_a > Q_best
        Q_best[mask] = Q_a[mask]
        policy_new[mask] = action

    return policy_new


# ---------------------------------------------------------------------------
# Policy Iteration
# ---------------------------------------------------------------------------

def policy_iteration(
    theta: float = 1e-2,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Full policy iteration loop."""
    policy: NDArray[np.int64] = np.zeros(
        (MAX_CARS + 1, MAX_CARS + 1), dtype=np.int64
    )
    V: NDArray[np.float64] = np.zeros(
        (MAX_CARS + 1, MAX_CARS + 1), dtype=np.float64
    )

    for pi_iter in range(50):
        logger.info("=== Policy Iteration {} ===", pi_iter)
        V = policy_evaluation(policy, V=V, theta=theta)
        policy_new = policy_improvement(V)

        if np.array_equal(policy_new, policy):
            logger.info("Policy converged after {} iterations!", pi_iter + 1)
            break
        policy = policy_new

    return V, policy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    V: NDArray[np.float64],
    policy: NDArray[np.int64],
    save_path: str | None = None,
) -> None:
    """Reproduce Silver Lecture 3 page 15 figures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Policy heatmap
    ax = axes[0]
    im = ax.contourf(
        np.arange(MAX_CARS + 1),
        np.arange(MAX_CARS + 1),
        policy,
        levels=np.arange(-MAX_MOVE - 0.5, MAX_MOVE + 1.5),
        cmap="RdYlGn",
    )
    ax.contour(
        np.arange(MAX_CARS + 1),
        np.arange(MAX_CARS + 1),
        policy,
        levels=np.arange(-MAX_MOVE, MAX_MOVE + 1),
        colors="black",
        linewidths=0.5,
    )
    plt.colorbar(im, ax=ax, label="Cars moved (loc2→loc1)")
    ax.set_xlabel("Cars at Location 2")
    ax.set_ylabel("Cars at Location 1")
    ax.set_title("Optimal Policy π*")

    # Value function surface
    ax2 = axes[1]
    n1_vals = np.arange(MAX_CARS + 1)
    n2_vals = np.arange(MAX_CARS + 1)
    N1, N2 = np.meshgrid(n1_vals, n2_vals, indexing="ij")
    c = ax2.contourf(N2, N1, V, levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax2, label="V*(s)")
    ax2.set_xlabel("Cars at Location 2")
    ax2.set_ylabel("Cars at Location 1")
    ax2.set_title("Optimal Value Function V*")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to {}", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    V, policy = policy_iteration(theta=1e-2)
    logger.info("Final Policy (rows=loc1, cols=loc2):\n{}", policy)
    plot_results(V, policy, save_path="notebooks/jacks_car_policy.png")


if __name__ == "__main__":
    main()
