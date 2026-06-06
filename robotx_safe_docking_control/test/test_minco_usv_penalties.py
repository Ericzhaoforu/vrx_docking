import numpy as np

from robotx_safe_docking_control.minco_s3nu import MINCO_S3NU
from robotx_safe_docking_control.minco_usv_penalties import (
    MincoUsvDensePenalty,
    MincoUsvPenaltyConfig,
)
from robotx_safe_docking_control.usv_flatness import UsvModelParams


def make_curved_minco():
    head_pva = np.array([
        [0.0, 0.0, 0.0],
        [0.4, 0.0, 0.08],
        [0.0, 0.0, 0.0],
    ])
    tail_pva = np.array([
        [2.5, 1.1, 0.45],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    inPs = np.array([
        [0.7, 0.1, 0.08],
        [1.5, 0.55, 0.22],
    ])
    ts = np.array([1.0, 1.1, 1.2])
    trajectory = MINCO_S3NU(dim=3)
    trajectory.set_conditions(head_pva, tail_pva)
    trajectory.set_parameters(inPs, ts)
    return trajectory


def test_loose_bounds_have_zero_reported_violations():
    trajectory = make_curved_minco()
    params = UsvModelParams(min_thrust=-1e6, max_thrust=1e6)
    penalty = MincoUsvDensePenalty(MincoUsvPenaltyConfig(
        params=params,
        tau_v_bar=1e6,
        velocity_bounds=(1e6, 1e6, 1e6),
        acceleration_bounds=(1e6, 1e6, 1e6),
    ))
    evaluation = penalty.evaluate(trajectory)
    assert evaluation.diagnostics["velocity_violation"] == 0.0
    assert evaluation.diagnostics["acceleration_violation"] == 0.0
    assert evaluation.diagnostics["actuator_bound_violation"] == 0.0
    assert evaluation.diagnostics["max_abs_tau_v"] <= 1e6


def test_tau_v_bound_violation_is_reported():
    trajectory = make_curved_minco()
    penalty = MincoUsvDensePenalty(MincoUsvPenaltyConfig(
        params=UsvModelParams(min_thrust=-1e6, max_thrust=1e6),
        tau_v_bar=0.01,
        velocity_bounds=(1e6, 1e6, 1e6),
        acceleration_bounds=(1e6, 1e6, 1e6),
    ))
    evaluation = penalty.evaluate(trajectory)
    assert evaluation.diagnostics["max_abs_tau_v"] > 0.01
    assert evaluation.tau_v > 0.0


def test_velocity_and_acceleration_violations_are_reported():
    trajectory = make_curved_minco()
    penalty = MincoUsvDensePenalty(MincoUsvPenaltyConfig(
        params=UsvModelParams(min_thrust=-1e6, max_thrust=1e6),
        tau_v_bar=1e6,
        velocity_bounds=(0.01, 0.01, 0.01),
        acceleration_bounds=(0.01, 0.01, 0.01),
    ))
    evaluation = penalty.evaluate(trajectory)
    assert evaluation.diagnostics["velocity_violation"] > 0.0
    assert evaluation.diagnostics["acceleration_violation"] > 0.0
    assert evaluation.velocity > 0.0
    assert evaluation.acceleration > 0.0


def test_actuator_bound_violation_uses_differential_thrust_mapping():
    trajectory = make_curved_minco()
    penalty = MincoUsvDensePenalty(MincoUsvPenaltyConfig(
        params=UsvModelParams(min_thrust=-1.0, max_thrust=1.0),
        tau_v_bar=1e6,
        velocity_bounds=(1e6, 1e6, 1e6),
        acceleration_bounds=(1e6, 1e6, 1e6),
    ))
    evaluation = penalty.evaluate(trajectory)
    assert evaluation.diagnostics["actuator_bound_violation"] > 0.0
    assert evaluation.actuator > 0.0
    assert (
        evaluation.diagnostics["max_left_thrust"] > 1.0 or
        evaluation.diagnostics["max_right_thrust"] > 1.0 or
        evaluation.diagnostics["min_left_thrust"] < -1.0 or
        evaluation.diagnostics["min_right_thrust"] < -1.0
    )


def test_penalty_finite_difference_gradients_are_finite_and_shaped():
    trajectory = make_curved_minco()
    penalty = MincoUsvDensePenalty(MincoUsvPenaltyConfig(
        params=UsvModelParams(),
        tau_v_bar=20.0,
        velocity_bounds=(0.8, 0.4, 0.4),
        acceleration_bounds=(0.8, 0.4, 0.4),
        quadrature_order=4,
    ))
    coeff_grad, time_grad = penalty.finite_difference_gradients(trajectory)
    assert coeff_grad.shape == trajectory.coefficients().shape
    assert time_grad.shape == trajectory.ts.shape
    assert np.all(np.isfinite(coeff_grad))
    assert np.all(np.isfinite(time_grad))


def test_penalty_analytic_gradient_matches_directional_finite_difference():
    trajectory = make_curved_minco()
    penalty = MincoUsvDensePenalty(MincoUsvPenaltyConfig(
        params=UsvModelParams(min_thrust=-100.0, max_thrust=100.0),
        tau_v_bar=20.0,
        velocity_bounds=(0.8, 0.4, 0.4),
        acceleration_bounds=(0.8, 0.4, 0.4),
        quadrature_order=4,
    ))
    coeff_grad, time_grad = penalty.analytic_gradients(trajectory)
    coeffs = trajectory.coefficients()
    ts = trajectory.ts.copy()

    rng = np.random.default_rng(42)
    coeff_direction = rng.normal(size=coeffs.shape)
    coeff_direction /= np.linalg.norm(coeff_direction)
    time_direction = rng.normal(size=ts.shape)
    time_direction /= np.linalg.norm(time_direction)

    eps = 1e-6
    plus = penalty._evaluate_coefficients(
        coeffs + eps * coeff_direction,
        ts + eps * time_direction,
    )
    minus = penalty._evaluate_coefficients(
        coeffs - eps * coeff_direction,
        ts - eps * time_direction,
    )
    numerical = (plus - minus) / (2.0 * eps)
    analytical = (
        float(np.sum(coeff_grad * coeff_direction)) +
        float(np.sum(time_grad * time_direction))
    )
    relative_error = abs(numerical - analytical) / max(1.0, abs(numerical))
    assert relative_error < 1e-6
