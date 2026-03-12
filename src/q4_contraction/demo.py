"""
Q4: Contraction Mapping – numerical demonstration.

We verify numerically that:
  1. The Bellman Expectation operator T^pi is a gamma-contraction.
  2. The Bellman Optimality operator T* is a gamma-contraction.
  3. Value iteration convergence error satisfies
       ||V_k - V*||_inf ≤ gamma^k * ||V_0 - V*||_inf

We use the Student MDP (from Q1) as the test environment.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from loguru import logger
from numpy.typing import NDArray

# Import the Student MDP
from src.q1_student_mdp.mdp import (
    ACTIONS,
    N_STATES,
    TERMINAL,
    TM,
    _uniform_policy,
    value_iteration,
)

# ---------------------------------------------------------------------------
# Bellman operators as explicit functions
# ---------------------------------------------------------------------------

def bellman_expectation(
    V: NDArray[np.float64],
    policy: dict[int, dict[str, float]],
    gamma: float,
) -> NDArray[np.float64]:
    """
    T^pi V  (one application of the Bellman Expectation operator).

    (T^pi V)(s) = sum_a pi(a|s) sum_s' P(s'|s,a)[R + gamma * V(s')]
    """
    TV = np.zeros(N_STATES, dtype=np.float64)
    for s in range(N_STATES):
        if s == TERMINAL:
            continue
        v = 0.0
        for action, pi_a in policy[s].items():
            for s_next, prob, rew in TM[s].get(action, []):
                v += pi_a * prob * (rew + gamma * V[s_next])
        TV[s] = v
    return TV


def bellman_optimality(
    V: NDArray[np.float64],
    gamma: float,
) -> NDArray[np.float64]:
    """
    T* V  (one application of the Bellman Optimality operator).

    (T* V)(s) = max_a sum_s' P(s'|s,a)[R + gamma * V(s')]
    """
    TV = np.zeros(N_STATES, dtype=np.float64)
    for s in range(N_STATES):
        if s == TERMINAL:
            continue
        best = max(
            sum(prob * (rew + gamma * V[s_next])
                for s_next, prob, rew in TM[s].get(action, []))
            for action in ACTIONS[s]
        )
        TV[s] = best
    return TV


# ---------------------------------------------------------------------------
# Verify contraction property
# ---------------------------------------------------------------------------

def verify_contraction(gamma: float = 0.9, n_trials: int = 200) -> None:
    """
    For random pairs (V1, V2), verify:
       ||T V1 - T V2||_inf ≤ gamma * ||V1 - V2||_inf

    for both T^pi and T*.
    """
    rng = np.random.default_rng(42)
    policy = _uniform_policy()

    ratios_exp, ratios_opt = [], []

    for _ in range(n_trials):
        V1 = rng.uniform(-20, 20, N_STATES)
        V2 = rng.uniform(-20, 20, N_STATES)
        V1[TERMINAL] = V2[TERMINAL] = 0.0

        diff = float(np.max(np.abs(V1 - V2)))
        if diff < 1e-8:
            continue

        TV1_exp = bellman_expectation(V1, policy, gamma)
        TV2_exp = bellman_expectation(V2, policy, gamma)
        ratio_exp = float(np.max(np.abs(TV1_exp - TV2_exp))) / diff
        ratios_exp.append(ratio_exp)

        TV1_opt = bellman_optimality(V1, gamma)
        TV2_opt = bellman_optimality(V2, gamma)
        ratio_opt = float(np.max(np.abs(TV1_opt - TV2_opt))) / diff
        ratios_opt.append(ratio_opt)

    logger.info("γ = {} | T^pi max ratio = {:.6f} (≤ {}) | T* max ratio = {:.6f} (≤ {})",
                gamma, max(ratios_exp), gamma, max(ratios_opt), gamma)
    assert max(ratios_exp) <= gamma + 1e-9, "T^pi is NOT a contraction!"
    assert max(ratios_opt) <= gamma + 1e-9, "T* is NOT a contraction!"
    logger.info("✔ Both operators satisfy the contraction property for γ={}", gamma)


# ---------------------------------------------------------------------------
# Convergence curve plot
# ---------------------------------------------------------------------------

def plot_convergence(gamma: float = 0.9) -> None:
    """
    Show that ||V_k - V*||_inf ≤ gamma^k * ||V_0 - V*||_inf.
    """
    V_star = value_iteration(gamma=gamma)

    V = np.zeros(N_STATES, dtype=np.float64)
    errors, bounds = [], []
    err0 = float(np.max(np.abs(V - V_star)))

    for k in range(1, 60):
        # One Bellman optimality sweep
        V = bellman_optimality(V, gamma)
        err = float(np.max(np.abs(V - V_star)))
        bound = (gamma ** k) * err0
        errors.append(err)
        bounds.append(bound)
        if err < 1e-8:
            break

    K = len(errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, K + 1), errors, label=r"$\|V_k - V^*\|_\infty$", lw=2)
    ax.semilogy(
        range(1, K + 1),
        bounds,
        label=r"$\gamma^k \|V_0 - V^*\|_\infty$ (bound)",
        lw=2,
        linestyle="--",
    )
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("Error")
    ax.set_title(f"Value Iteration Convergence (gamma={gamma})")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("notebooks/contraction_convergence.png", dpi=150, bbox_inches="tight")
    logger.info("Convergence plot saved to notebooks/contraction_convergence.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 55)
    logger.info("Q4 – Contraction Mapping: Numerical Verification")
    logger.info("=" * 55)

    for g in [0.5, 0.9, 0.99]:
        verify_contraction(gamma=g)

    plot_convergence(gamma=0.9)


if __name__ == "__main__":
    main()
