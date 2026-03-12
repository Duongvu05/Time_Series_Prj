"""
Q2: Iterative Policy Evaluation on the 4×4 Grid World.

Silver Lecture 3, pages 10–11.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE          # 16
TERMINAL_STATES = {0, 15}

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # N, S, W, E
N_ACTIONS = len(ACTIONS)

REWARD = -1.0
GAMMA = 1.0


def _next_state(s: int, dr: int, dc: int) -> int:
    """Return next state after moving (dr, dc) from state s. Reflects at edges."""
    r, c = divmod(s, GRID_SIZE)
    nr = max(0, min(GRID_SIZE - 1, r + dr))
    nc = max(0, min(GRID_SIZE - 1, c + dc))
    return nr * GRID_SIZE + nc


# Precompute P[s][a] = (s_next, reward)
_P: list[list[tuple[int, float]]] = []
for s in range(N_STATES):
    row = []
    for dr, dc in ACTIONS:
        if s in TERMINAL_STATES:
            row.append((s, 0.0))
        else:
            row.append((_next_state(s, dr, dc), REWARD))
    _P.append(row)


# ---------------------------------------------------------------------------
# Iterative Policy Evaluation
# ---------------------------------------------------------------------------

def iterative_policy_eval_step(
    V: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Perform ONE sweep of policy evaluation (uniform random policy)."""
    V_new = np.zeros(N_STATES, dtype=np.float64)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            V_new[s] = 0.0
            continue
        v = 0.0
        for a in range(N_ACTIONS):
            s_next, rew = _P[s][a]
            v += (1.0 / N_ACTIONS) * (rew + GAMMA * V[s_next])
        V_new[s] = v
    return V_new


def iterative_policy_eval(k: int) -> NDArray[np.float64]:
    """Run k sweeps from V_0 = 0. Return V_k."""
    V = np.zeros(N_STATES, dtype=np.float64)
    for _ in range(k):
        V = iterative_policy_eval_step(V)
    return V


def run_to_convergence(theta: float = 1e-6) -> tuple[NDArray[np.float64], int]:
    """Run until max |V_{k+1}(s) - V_k(s)| < theta. Returns (V_converged, k)."""
    V = np.zeros(N_STATES, dtype=np.float64)
    k = 0
    while True:
        V_new = iterative_policy_eval_step(V)
        k += 1
        delta = float(np.max(np.abs(V_new - V)))
        if delta < theta:
            logger.info("Grid world converged after {} sweeps (delta={:.2e})", k, delta)
            return V_new, k
        V = V_new


# ---------------------------------------------------------------------------
# Greedy policy from V
# ---------------------------------------------------------------------------

def greedy_policy(V: NDArray[np.float64]) -> NDArray[np.int64]:
    """Returns the greedy deterministic policy from V."""
    pi = np.zeros(N_STATES, dtype=np.int64)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            pi[s] = 0
            continue
        best_a = int(
            np.argmax([_P[s][a][1] + GAMMA * V[_P[s][a][0]] for a in range(N_ACTIONS)])
        )
        pi[s] = best_a
    return pi


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_grid(V: NDArray[np.float64], title: str = "") -> None:
    """Log V values in a 4×4 grid."""
    lines = []
    if title:
        lines.append(title)
    lines.append("┌" + "────────┬" * (GRID_SIZE - 1) + "────────┐")
    for r in range(GRID_SIZE):
        row_vals = [f"{V[r * GRID_SIZE + c]:6.2f}" for c in range(GRID_SIZE)]
        lines.append("│ " + " │ ".join(row_vals) + " │")
        if r < GRID_SIZE - 1:
            lines.append("├" + "────────┼" * (GRID_SIZE - 1) + "────────┤")
    lines.append("└" + "────────┴" * (GRID_SIZE - 1) + "────────┘")
    logger.info("\n" + "\n".join(lines))


ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def display_policy(pi: NDArray[np.int64], title: str = "") -> None:
    """Log policy arrows in a 4×4 grid."""
    lines = []
    if title:
        lines.append(title)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = r * GRID_SIZE + c
            row.append("T" if s in TERMINAL_STATES else ARROW[int(pi[s])])
        lines.append("  " + "  ".join(row))
    logger.info("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Q2: Iterative Policy Evaluation – 4×4 Grid World ===")
    logger.info("Silver Lecture 3, pp. 10–11")

    for k in [1, 2, 3, 10, 100]:
        V_k = iterative_policy_eval(k)
        display_grid(V_k, title=f"V_{k}")

    V_conv, k_conv = run_to_convergence(theta=1e-6)
    display_grid(V_conv, title=f"V_∞ (converged at k={k_conv})")

    pi_star = greedy_policy(V_conv)
    display_policy(pi_star, title="Greedy Policy from V_∞")


if __name__ == "__main__":
    main()
