"""
Q1: Student MDP from Silver Lecture 2.

States: C1, C2, C3, Pass, Pub, FB, Sleep
(Sleep is terminal with V=0)
Transitions and rewards follow slides pp. 29-47.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# MDP Definition (Silver Lecture 2, Student Example)
# ---------------------------------------------------------------------------

# States (indices)
STATES: list[str] = ["C1", "C2", "C3", "FB", "Sleep"]
S_IDX: dict[str, int] = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
TERMINAL = S_IDX["Sleep"]

# Actions per state
ACTIONS: dict[int, list[str]] = {
    S_IDX["C1"]:   ["Study", "Facebook"],
    S_IDX["C2"]:   ["Study", "Sleep"],
    S_IDX["C3"]:   ["Study", "Pub"],
    S_IDX["FB"]:   ["Facebook", "Quit"],
    S_IDX["Sleep"]: ["Stay"],
}

# Transition & reward tables
_TRANSITIONS: list[tuple[str, str, str, float, float]] = [
    # action,       from,   to,     prob, reward
    ("Study",    "C1",   "C2",   1.0,  -2.0),
    ("Facebook", "C1",   "FB",   1.0,  -1.0),
    ("Study",    "C2",   "C3",   1.0,  -2.0),
    ("Sleep",    "C2",   "Sleep",1.0,   0.0),
    ("Study",    "C3",   "Sleep",1.0,  10.0),
    ("Pub",      "C3",   "C1",   0.2,   1.0),
    ("Pub",      "C3",   "C2",   0.4,   1.0),
    ("Pub",      "C3",   "C3",   0.4,   1.0),
    ("Facebook", "FB",   "FB",   1.0,  -1.0),
    ("Quit",     "FB",   "C1",   1.0,   0.0),
    ("Stay",     "Sleep","Sleep",1.0,   0.0),
]

# Build lookup: transitions[from_state_idx][action_name] = [(to_idx, prob, reward)]
TransitionMap = dict[int, dict[str, list[tuple[int, float, float]]]]


def build_transition_map() -> TransitionMap:
    """Build nested dict for efficient lookup."""
    tm: TransitionMap = {i: {} for i in range(N_STATES)}
    for action, frm, to, prob, rew in _TRANSITIONS:
        fi = S_IDX[frm]
        ti = S_IDX[to]
        tm[fi].setdefault(action, []).append((ti, prob, rew))
    return tm


TM = build_transition_map()

# ---------------------------------------------------------------------------
# Random (uniform) policy used in Lecture 2 slides pp. 29-40
# ---------------------------------------------------------------------------

def _uniform_policy() -> dict[int, dict[str, float]]:
    """Returns policy dict pi[s] = {action: prob} for uniform random policy."""
    policy: dict[int, dict[str, float]] = {}
    for si, actions in ACTIONS.items():
        n = len(actions)
        policy[si] = {a: 1.0 / n for a in actions}
    return policy


# ---------------------------------------------------------------------------
# Policy Evaluation  (Bellman Expectation, iterative)
# ---------------------------------------------------------------------------

def policy_evaluation(
    policy: dict[int, dict[str, float]],
    gamma: float = 1.0,
    theta: float = 1e-9,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """
    Iterative policy evaluation.

    V^pi(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) [R(s,a,s') + gamma * V^pi(s')]
    """
    V: NDArray[np.float64] = np.zeros(N_STATES, dtype=np.float64)

    for iteration in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in range(N_STATES):
            if s == TERMINAL:
                continue
            v = 0.0
            for action, pi_a in policy[s].items():
                for (s_next, prob, rew) in TM[s].get(action, []):
                    v += pi_a * prob * (rew + gamma * V[s_next])
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:
            logger.debug("Policy evaluation converged at iteration {}", iteration + 1)
            break

    return V


# ---------------------------------------------------------------------------
# Q-value from V
# ---------------------------------------------------------------------------

def compute_q_from_v(
    V: NDArray[np.float64],
    gamma: float = 1.0,
) -> dict[int, dict[str, float]]:
    """Q^pi(s, a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma * V(s')]"""
    Q: dict[int, dict[str, float]] = {}
    for s in range(N_STATES):
        Q[s] = {}
        for action in ACTIONS[s]:
            q = 0.0
            for (s_next, prob, rew) in TM[s].get(action, []):
                q += prob * (rew + gamma * V[s_next])
            Q[s][action] = q
    return Q


# ---------------------------------------------------------------------------
# Value Iteration → optimal V*
# ---------------------------------------------------------------------------

def value_iteration(
    gamma: float = 1.0,
    theta: float = 1e-9,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """Value iteration: V*(s) = max_a sum_s' P(s'|s,a) [R + gamma * V*(s')]"""
    V: NDArray[np.float64] = np.zeros(N_STATES, dtype=np.float64)

    for iteration in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in range(N_STATES):
            if s == TERMINAL:
                continue
            action_vals = []
            for action in ACTIONS[s]:
                q = sum(
                    prob * (rew + gamma * V[s_next])
                    for (s_next, prob, rew) in TM[s].get(action, [])
                )
                action_vals.append(q)
            V_new[s] = max(action_vals)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:
            logger.debug("Value iteration converged at iteration {}", iteration + 1)
            break

    return V


def optimal_policy(
    V_star: NDArray[np.float64], gamma: float = 1.0
) -> dict[int, str]:
    """Extract greedy (optimal) policy from V*."""
    pi: dict[int, str] = {}
    for s in range(N_STATES):
        if s == TERMINAL:
            pi[s] = "Stay"
            continue
        best_action = max(
            ACTIONS[s],
            key=lambda a: sum(
                prob * (rew + gamma * V_star[s_next])
                for (s_next, prob, rew) in TM[s].get(a, [])
            ),
        )
        pi[s] = best_action
    return pi


def optimal_q(
    V_star: NDArray[np.float64], gamma: float = 1.0
) -> dict[int, dict[str, float]]:
    """Q*(s,a) computed from V*."""
    return compute_q_from_v(V_star, gamma)


# ---------------------------------------------------------------------------
# Pretty logging helpers
# ---------------------------------------------------------------------------

def print_values(V: NDArray[np.float64], label: str = "V") -> None:
    sep = "─" * 40
    lines = [f"\n{sep}", f"  {label}", sep]
    for s, name in enumerate(STATES):
        lines.append(f"  {name:8s}: {V[s]:8.3f}")
    logger.info("\n".join(lines))


def print_q(Q: dict[int, dict[str, float]], label: str = "Q") -> None:
    sep = "─" * 40
    lines = [f"\n{sep}", f"  {label}", sep]
    for s, name in enumerate(STATES):
        for action, val in Q[s].items():
            lines.append(f"  ({name:6s}, {action:10s}): {val:8.3f}")
    logger.info("\n".join(lines))


def print_policy(pi: dict[int, str]) -> None:
    sep = "─" * 40
    lines = [f"\n{sep}", "  Optimal Policy", sep]
    for s, name in enumerate(STATES):
        lines.append(f"  {name:8s}: {pi[s]}")
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    gamma = 1.0
    logger.info("=== Q1: Student MDP Demo (γ={}) ===", gamma)

    rand_policy = _uniform_policy()
    V_pi = policy_evaluation(rand_policy, gamma=gamma)
    print_values(V_pi, label=f"V^π (uniform policy, γ={gamma})")

    Q_pi = compute_q_from_v(V_pi, gamma=gamma)
    print_q(Q_pi, label="Q^π (uniform policy)")

    V_star = value_iteration(gamma=gamma)
    print_values(V_star, label="V* (optimal)")

    Q_star = optimal_q(V_star, gamma=gamma)
    print_q(Q_star, label="Q* (optimal)")

    pi_star = optimal_policy(V_star, gamma=gamma)
    print_policy(pi_star)

    gamma2 = 0.9
    logger.info("=== Discounted case (γ={}) ===", gamma2)
    V_pi_09 = policy_evaluation(rand_policy, gamma=gamma2)
    V_star_09 = value_iteration(gamma=gamma2)
    print_values(V_pi_09, label=f"V^π (uniform, γ={gamma2})")
    print_values(V_star_09, label=f"V* (γ={gamma2})")


if __name__ == "__main__":
    main()
