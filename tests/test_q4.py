"""Tests for Q4: Contraction Mapping property."""

from __future__ import annotations

import numpy as np
import pytest

from src.q1_student_mdp.mdp import (
    N_STATES,
    TERMINAL,
    _uniform_policy,
    value_iteration,
)
from src.q4_contraction.demo import (
    bellman_expectation,
    bellman_optimality,
    verify_contraction,
)


class TestBellmanOperators:
    """Verify basic properties of T^pi and T*."""

    def test_bellman_expectation_shape(self) -> None:
        policy = _uniform_policy()
        V = np.zeros(N_STATES)
        TV = bellman_expectation(V, policy, gamma=0.9)
        assert TV.shape == (N_STATES,)

    def test_bellman_optimality_shape(self) -> None:
        V = np.zeros(N_STATES)
        TV = bellman_optimality(V, gamma=0.9)
        assert TV.shape == (N_STATES,)

    def test_terminal_stays_zero(self) -> None:
        policy = _uniform_policy()
        V = np.random.default_rng(0).uniform(-10, 10, N_STATES)
        V[TERMINAL] = 0.0
        TV_exp = bellman_expectation(V, policy, gamma=0.9)
        TV_opt = bellman_optimality(V, gamma=0.9)
        assert TV_exp[TERMINAL] == pytest.approx(0.0, abs=1e-9)
        assert TV_opt[TERMINAL] == pytest.approx(0.0, abs=1e-9)


class TestContractionProperty:
    @pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
    def test_contraction_ratio(self, gamma: float) -> None:
        """
        ||T^pi V1 - T^pi V2||_inf ≤ gamma * ||V1 - V2||_inf
        ||T*   V1 - T*   V2||_inf ≤ gamma * ||V1 - V2||_inf
        """
        rng = np.random.default_rng(7)
        policy = _uniform_policy()
        max_ratio_exp = 0.0
        max_ratio_opt = 0.0

        for _ in range(100):
            V1 = rng.uniform(-20, 20, N_STATES)
            V2 = rng.uniform(-20, 20, N_STATES)
            V1[TERMINAL] = V2[TERMINAL] = 0.0
            diff = float(np.max(np.abs(V1 - V2)))
            if diff < 1e-8:
                continue

            TV1 = bellman_expectation(V1, policy, gamma)
            TV2 = bellman_expectation(V2, policy, gamma)
            max_ratio_exp = max(max_ratio_exp,
                                float(np.max(np.abs(TV1 - TV2))) / diff)

            TsV1 = bellman_optimality(V1, gamma)
            TsV2 = bellman_optimality(V2, gamma)
            max_ratio_opt = max(max_ratio_opt,
                                float(np.max(np.abs(TsV1 - TsV2))) / diff)

        assert max_ratio_exp <= gamma + 1e-8
        assert max_ratio_opt <= gamma + 1e-8

    def test_verify_contraction_passes(self) -> None:
        """verify_contraction should not raise AssertionError."""
        verify_contraction(gamma=0.9, n_trials=50)


class TestConvergenceBound:
    def test_value_iter_stays_within_bound(self) -> None:
        """
        After k Bellman optimality sweeps starting from V_0=0:
          ||V_k - V*||_inf ≤ gamma^k * ||V_0 - V*||_inf
        """
        gamma = 0.9
        V_star = value_iteration(gamma=gamma)
        V = np.zeros(N_STATES, dtype=np.float64)
        err0 = float(np.max(np.abs(V - V_star)))

        for k in range(1, 30):
            V = bellman_optimality(V, gamma)
            err = float(np.max(np.abs(V - V_star)))
            bound = (gamma ** k) * err0
            assert err <= bound + 1e-6, (
                f"k={k}: error={err:.6f} > bound={bound:.6f}"
            )
            if err < 1e-8:
                break
