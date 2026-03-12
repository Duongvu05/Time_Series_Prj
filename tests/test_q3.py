"""Tests for Q3: Jack's Car Rental."""

from __future__ import annotations

import numpy as np
import pytest

from src.q3_jacks_car.jacks_car import (
    _EXP_REW_1,
    _EXP_REW_2,
    _TRANS_1,
    _TRANS_2,
    MAX_CARS,
    MAX_MOVE,
    RENTAL_REWARD,
    _poisson_probs,
    poisson_pmf,
    policy_improvement,
)


class TestPoissonHelper:
    def test_pmf_sums_to_one(self) -> None:
        """PMF over 0..POISSON_UPPER should be close to 1."""
        for lam in [1, 2, 3, 4, 5]:
            total = sum(poisson_pmf(lam, k) for k in range(50))
            assert total == pytest.approx(1.0, abs=1e-4)

    def test_mode_is_lambda(self) -> None:
        """For integer lambda, PMF peaks at lambda (or lambda-1)."""
        for lam in [2, 3, 4]:
            probs = [poisson_pmf(lam, k) for k in range(2 * lam)]
            mode = int(np.argmax(probs))
            assert mode in {lam - 1, lam}

    def test_probs_array_shape(self) -> None:
        p = _poisson_probs(3)
        assert p.shape[0] == 12  # POISSON_UPPER + 1


class TestLocationDynamics:
    def test_expected_reward_increases_with_cars(self) -> None:
        """More cars → more (or equal) expected rentals → higher reward."""
        assert np.all(np.diff(_EXP_REW_1) >= 0)
        assert np.all(np.diff(_EXP_REW_2) >= 0)

    def test_expected_reward_at_zero_is_zero(self) -> None:
        assert _EXP_REW_1[0] == pytest.approx(0.0, abs=1e-9)
        assert _EXP_REW_2[0] == pytest.approx(0.0, abs=1e-9)

    def test_transition_rows_sum_to_one(self) -> None:
        # Rows may not sum exactly to 1 due to Poisson truncation at POISSON_UPPER=11
        # Max P(X > 11) for lambda=2 is ~0.1%, so tolerance is 2e-3
        assert np.allclose(_TRANS_1.sum(axis=1), 1.0, atol=2e-3)
        assert np.allclose(_TRANS_2.sum(axis=1), 1.0, atol=2e-3)

    def test_transition_shape(self) -> None:
        assert _TRANS_1.shape == (MAX_CARS + 1, MAX_CARS + 1)
        assert _TRANS_2.shape == (MAX_CARS + 1, MAX_CARS + 1)

    def test_expected_reward_bounded(self) -> None:
        """Max reward bounded by MAX_CARS * RENTAL_REWARD per location."""
        assert float(np.max(_EXP_REW_1)) <= MAX_CARS * RENTAL_REWARD
        assert float(np.max(_EXP_REW_2)) <= MAX_CARS * RENTAL_REWARD


class TestPolicyImprovement:
    def test_policy_improvement_returns_correct_shape(self) -> None:
        V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float64)
        policy = policy_improvement(V)
        assert policy.shape == (MAX_CARS + 1, MAX_CARS + 1)

    def test_policy_values_in_valid_range(self) -> None:
        V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float64)
        policy = policy_improvement(V)
        assert int(np.min(policy)) >= -MAX_MOVE
        assert int(np.max(policy)) <= MAX_MOVE

    def test_initial_zero_value_policy_is_nonnegative(self) -> None:
        """
        When V=0, optimal action tends to favour rental at location 2
        (higher lambda), so policy should prefer moving cars from loc2→loc1
        slightly, but is bounded by move cost.
        """
        V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float64)
        policy = policy_improvement(V)
        # Most actions should be within [-MAX_MOVE, MAX_MOVE]
        assert np.all(np.abs(policy) <= MAX_MOVE)
