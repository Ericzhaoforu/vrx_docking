import math

import numpy as np

from robotx_safe_docking_control.usv_flatness import (
    UsvModelParams,
    allocation_feasible,
    body_to_flat_velocity,
    flat_to_body_velocity,
    generalized_force_to_thrust,
    theta_tau_nominal,
    thrust_to_generalized_force,
    wrap_angle,
)


def test_angle_wrap():
    assert math.isclose(wrap_angle(math.pi + 0.2), -math.pi + 0.2)
    assert math.isclose(wrap_angle(-math.pi - 0.2), math.pi - 0.2)


def test_flat_body_velocity_inverse_consistency():
    z = np.array([1.0, -2.0, 0.73])
    z_dot = np.array([0.8, -0.2, 0.13])
    nu = flat_to_body_velocity(z, z_dot)
    recovered = body_to_flat_velocity(z[2], nu)
    assert np.allclose(recovered, z_dot)


def test_yaw_rate_is_body_r():
    z = np.array([0.0, 0.0, -1.2])
    z_dot = np.array([0.1, 0.2, -0.33])
    nu = flat_to_body_velocity(z, z_dot)
    assert math.isclose(nu[2], z_dot[2])


def test_allocation_equal_thrust_gives_surge_only():
    params = UsvModelParams()
    tau_u, tau_r = thrust_to_generalized_force(100.0, 100.0, params)
    assert math.isclose(tau_u, 200.0)
    assert math.isclose(tau_r, 0.0)


def test_allocation_right_greater_than_left_gives_positive_yaw():
    params = UsvModelParams(thruster_half_spacing=2.0)
    tau_u, tau_r = thrust_to_generalized_force(50.0, 100.0, params)
    assert math.isclose(tau_u, 150.0)
    assert tau_r > 0.0


def test_generalized_force_allocation_roundtrip():
    params = UsvModelParams(thruster_half_spacing=1.5)
    left, right = generalized_force_to_thrust(120.0, 30.0, params)
    tau_u, tau_r = thrust_to_generalized_force(left, right, params)
    assert math.isclose(tau_u, 120.0)
    assert math.isclose(tau_r, 30.0)
    assert allocation_feasible(tau_u, tau_r, params)


def test_theta_tau_nominal_finite_representative_state():
    params = UsvModelParams()
    z = np.array([2.0, 3.0, 0.4])
    z_dot = np.array([0.5, 0.1, 0.05])
    z_ddot = np.array([0.02, -0.01, 0.005])
    tau = theta_tau_nominal(z, z_dot, z_ddot, params)
    assert tau.shape == (3,)
    assert np.all(np.isfinite(tau))
