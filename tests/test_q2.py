"""Tests for Q2: Iterative Policy Evaluation on 4×4 Grid World."""

from __future__ import annotations

import numpy as np
import pytest

from src.q2_grid_world.grid_world import (
    N_STATES,
    TERMINAL_STATES,
    greedy_policy,
    iterative_policy_eval,
    iterative_policy_eval_step,
    run_to_convergence,
    run_to_convergence_inplace,
)


class TestGridWorld:
    def test_v0_is_all_zeros(self) -> None:
        V = iterative_policy_eval(k=0)
        assert np.allclose(V, 0.0)

    def test_terminal_states_always_zero(self) -> None:
        for k in [1, 2, 3, 10, 50]:
            V = iterative_policy_eval(k=k)
            for t in TERMINAL_STATES:
                assert V[t] == pytest.approx(0.0, abs=1e-9)

    def test_v1_boundary_cell(self) -> None:
        """
        After k=1, cells adjacent to terminal should have V = -1.
        State 1 (top-row, adjacent to terminal 0) with 4 equal actions:
          N->0 (rew=-1), S->5 (rew=-1), W->0 (rew=-1), E->2 (rew=-1)
          V_1(1) = (1/4)*(-1+0)*4 = -1.0
        """
        V1 = iterative_policy_eval(k=1)
        assert V1[1] == pytest.approx(-1.0, abs=1e-9)

    def test_v1_corner_far(self) -> None:
        """
        State 10 is in the interior far from terminals.
        After k=1 all non-terminal states should have V_1 = -1
        (since V_0=0 everywhere, each action gives R=-1).
        """
        V1 = iterative_policy_eval(k=1)
        for s in range(N_STATES):
            if s not in TERMINAL_STATES:
                assert V1[s] == pytest.approx(-1.0, abs=1e-9)

    def test_v2_matches_slides(self) -> None:
        """
        Silver Lecture 3 p.10, k=2 spot checks.
        State 1 (top row, col 1) – N action hits top wall and REFLECTS to state 1:
          N→1 (wall): -1+V1(1)=-1+(-1)=-2
          S→5:        -1+V1(5)=-1+(-1)=-2
          W→0 (term): -1+V1(0)=-1+0=-1
          E→2:        -1+V1(2)=-1+(-1)=-2
          V_2(1) = 0.25 * (-2-2-1-2) = -7/4 = -1.75
        State 5 (interior):
          N→1: -2; S→9: -2; W→4: -2; E→6: -2
          V_2(5) = 0.25*(-8) = -2.0
        """
        V2 = iterative_policy_eval(k=2)
        assert V2[1] == pytest.approx(-1.75, abs=1e-9)
        assert V2[5] == pytest.approx(-2.0, abs=1e-9)


    def test_monotone_decrease(self) -> None:
        """V_k(s) should be non-increasing (getting more negative) as k grows."""
        Vs = [iterative_policy_eval(k=k) for k in range(1, 10)]
        for i in range(len(Vs) - 1):
            assert np.all(Vs[i + 1] <= Vs[i] + 1e-6)

    def test_convergence(self) -> None:
        V_conv, k = run_to_convergence(theta=1e-6)
        assert k > 0
        # Converged values should be symmetric (grid is symmetric)
        # V(1)==V(4), V(2)==V(8), etc.  (by symmetry of random walk on square grid)
        assert V_conv[1] == pytest.approx(V_conv[4], abs=0.5)

    def test_convergence_stops_improving(self) -> None:
        """After convergence, one more step should change nothing."""
        V_conv, _ = run_to_convergence(theta=1e-6)
        V_next = iterative_policy_eval_step(V_conv)
        assert np.allclose(V_conv, V_next, atol=1e-5)

    def test_convergence_inplace(self) -> None:
        """In-place and regular convergence should reach the same V_infinity."""
        V_reg, _ = run_to_convergence(theta=1e-8)
        V_inp, _ = run_to_convergence_inplace(theta=1e-8)
        assert np.allclose(V_reg, V_inp, atol=1e-4)

    def test_greedy_policy_shape(self) -> None:
        V_conv, _ = run_to_convergence()
        pi = greedy_policy(V_conv)
        assert pi.shape == (N_STATES,)

    def test_greedy_terminal_stays(self) -> None:
        V_conv, _ = run_to_convergence()
        pi = greedy_policy(V_conv)
        for t in TERMINAL_STATES:
            # terminal states should have action 0
            assert pi[t] == 0
