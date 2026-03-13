"""Tests for Q1: Student MDP."""

from __future__ import annotations

import numpy as np
import pytest

from src.q1_student_mdp.mdp import (
    ACTIONS,
    N_STATES,
    S_IDX,
    TM,
    _uniform_policy,
    compute_q_from_v,
    optimal_policy,
    optimal_q,
    policy_evaluation,
    value_iteration,
)


class TestTransitionMap:
    def test_all_states_in_map(self) -> None:
        assert len(TM) == N_STATES

    def test_c1_study_leads_to_c2(self) -> None:
        c1 = S_IDX["C1"]
        transitions = TM[c1]["Study"]
        assert len(transitions) == 1
        s_next, prob, rew = transitions[0]
        assert s_next == S_IDX["C2"]
        assert prob == pytest.approx(1.0)
        assert rew == pytest.approx(-2.0)

    def test_c3_pub_has_three_transitions(self) -> None:
        c3 = S_IDX["C3"]
        transitions = TM[c3]["Pub"]
        assert len(transitions) == 3
        total_prob = sum(t[1] for t in transitions)
        assert total_prob == pytest.approx(1.0)

    def test_c3_study_leads_to_sleep_with_10(self) -> None:
        c3 = S_IDX["C3"]
        transitions = TM[c3]["Study"]
        assert len(transitions) == 1
        s_next, _, rew = transitions[0]
        assert s_next == S_IDX["Sleep"]
        assert rew == pytest.approx(10.0)


class TestPolicyEvaluation:
    """
    Expected values from Câu 1 - ML2.pdf.
    With gamma=1 and the uniform random policy:
    V^pi(Facebook)=-2.3, C1=-1.3, C2=2.7, C3=7.4
    """

    def test_terminal_state_is_zero(self) -> None:
        policy = _uniform_policy()
        V = policy_evaluation(policy, gamma=1.0)
        assert V[S_IDX["Sleep"]] == pytest.approx(0.0, abs=1e-6)

    def test_values_match_pdf(self) -> None:
        policy = _uniform_policy()
        V = policy_evaluation(policy, gamma=1.0)
        assert V[S_IDX["FB"]] == pytest.approx(-2.3, abs=0.1)
        assert V[S_IDX["C1"]] == pytest.approx(-1.3, abs=0.1)
        assert V[S_IDX["C2"]] == pytest.approx(2.7, abs=0.1)
        assert V[S_IDX["C3"]] == pytest.approx(7.4, abs=0.1)

    def test_discount_reduces_values(self) -> None:
        """With gamma<1, values should be different."""
        policy = _uniform_policy()
        V1 = policy_evaluation(policy, gamma=1.0)
        V09 = policy_evaluation(policy, gamma=0.9)
        # Check that values changed
        assert not np.allclose(V1, V09)


class TestQValue:
    def test_q_shape(self) -> None:
        policy = _uniform_policy()
        V = policy_evaluation(policy, gamma=1.0)
        Q = compute_q_from_v(V, gamma=1.0)
        assert set(Q.keys()) == set(range(N_STATES))

    def test_q_and_v_consistency(self) -> None:
        """V^pi(s) = sum_a pi(a|s) Q^pi(s,a)."""
        policy = _uniform_policy()
        V = policy_evaluation(policy, gamma=1.0)
        Q = compute_q_from_v(V, gamma=1.0)
        for s in range(N_STATES):
            v_from_q = sum(policy[s][a] * Q[s][a] for a in ACTIONS[s])
            assert v_from_q == pytest.approx(V[s], abs=1e-4)


class TestValueIteration:
    def test_v_star_geq_v_pi(self) -> None:
        """V* should be ≥ V^pi component-wise."""
        policy = _uniform_policy()
        V_pi = policy_evaluation(policy, gamma=1.0)
        V_star = value_iteration(gamma=1.0)
        assert np.all(V_star >= V_pi - 1e-4)

    def test_v_star_matches_pdf(self) -> None:
        """PDF optimal values: F=6, C1=6, C2=8, C3=10."""
        V_star = value_iteration(gamma=1.0)
        assert V_star[S_IDX["FB"]] == pytest.approx(6.0, abs=0.1)
        assert V_star[S_IDX["C1"]] == pytest.approx(6.0, abs=0.1)
        assert V_star[S_IDX["C2"]] == pytest.approx(8.0, abs=0.1)
        assert V_star[S_IDX["C3"]] == pytest.approx(10.0, abs=0.1)

    def test_optimal_policy_c3_is_study(self) -> None:
        V_star = value_iteration(gamma=1.0)
        pi = optimal_policy(V_star, gamma=1.0)
        assert pi[S_IDX["C3"]] == "Study"

    def test_optimal_q_geq_v_star(self) -> None:
        """max_a Q*(s,a) = V*(s)."""
        V_star = value_iteration(gamma=1.0)
        Q_star = optimal_q(V_star, gamma=1.0)
        for s in range(N_STATES):
            best_q = max(Q_star[s].values())
            assert best_q == pytest.approx(V_star[s], abs=1e-4)
