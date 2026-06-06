import numpy as np

from robotx_safe_docking_control.minco_usv_optimizer import (
    MincoUsvOptimizer,
    MincoUsvOptimizerConfig,
)
from robotx_safe_docking_control.minco_usv_penalties import MincoUsvPenaltyConfig
from robotx_safe_docking_control.usv_flatness import UsvModelParams


def optimizer_without_active_penalties(piece_count=3, max_iterations=20):
    return MincoUsvOptimizer(
        MincoUsvPenaltyConfig(
            params=UsvModelParams(min_thrust=-500.0, max_thrust=500.0),
            tau_v_bar=1e5,
            velocity_bounds=(10.0, 10.0, 10.0),
            acceleration_bounds=(10.0, 10.0, 10.0),
            lambda_tau_v=0.0,
            lambda_velocity=0.0,
            lambda_acceleration=0.0,
            lambda_actuator=0.0,
            quadrature_order=4,
        ),
        MincoUsvOptimizerConfig(
            piece_count=piece_count,
            fixed_total_time=5.0,
            max_iterations=max_iterations,
        ),
    )


def boundary_conditions():
    head_pva = np.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.02],
        [0.0, 0.0, 0.0],
    ])
    tail_pva = np.array([
        [3.0, 1.0, 0.35],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    return head_pva, tail_pva


def test_fixed_total_time_softmax_times_are_positive_and_sum_to_total():
    optimizer = optimizer_without_active_penalties(piece_count=4)
    theta = np.array([-1.0, 0.2, 1.4, -0.3])
    ts = optimizer.theta_to_times(theta)
    assert np.all(ts > 0.0)
    assert np.isclose(np.sum(ts), 5.0)


def test_optimizer_reduces_objective_and_preserves_fixed_total_time():
    head_pva, tail_pva = boundary_conditions()
    optimizer = optimizer_without_active_penalties(piece_count=3, max_iterations=25)
    result = optimizer.optimize(head_pva, tail_pva)
    assert result.success
    assert result.objective_after < result.objective_before
    assert np.isclose(np.sum(result.ts), 5.0)
    assert result.objective_after <= 0.25 * result.objective_before


def test_outer_objective_gradient_matches_finite_difference():
    head_pva, tail_pva = boundary_conditions()
    optimizer = optimizer_without_active_penalties(piece_count=2, max_iterations=1)
    inPs, ts = optimizer.initial_guess(head_pva, tail_pva)
    variables = optimizer.pack_variables(inPs, optimizer.times_to_theta(ts))
    _objective, analytical, _trajectory, _penalty = optimizer.objective_and_gradient(
        variables,
        head_pva,
        tail_pva,
    )
    eps = 1e-6
    finite_difference = np.zeros_like(analytical)
    for index in range(variables.size):
        perturbed = variables.copy()
        perturbed[index] += eps
        plus, _grad, _traj, _penalty = optimizer.objective_and_gradient(
            perturbed,
            head_pva,
            tail_pva,
        )
        perturbed[index] -= 2.0 * eps
        minus, _grad, _traj, _penalty = optimizer.objective_and_gradient(
            perturbed,
            head_pva,
            tail_pva,
        )
        finite_difference[index] = (plus - minus) / (2.0 * eps)

    relative_error = np.linalg.norm(analytical - finite_difference) / max(
        1.0, np.linalg.norm(finite_difference)
    )
    assert relative_error < 1e-5
