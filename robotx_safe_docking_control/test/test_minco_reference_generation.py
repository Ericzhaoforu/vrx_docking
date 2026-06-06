import numpy as np

from robotx_safe_docking_control.minco_reference_generation import (
    BOUNDARY_ACCEL_SOURCE_IDS,
    BoundaryAccelerationSelection,
    MincoLocalReferenceConfig,
    create_minco_local_reference,
    nominal_flat_acceleration_from_tau,
    select_boundary_acceleration,
)
from robotx_safe_docking_control.usv_flatness import UsvModelParams


def test_boundary_acceleration_policy_priority_order():
    params = UsvModelParams()
    z = np.array([1.0, -0.5, 0.25])
    z_dot = np.array([0.2, 0.1, 0.03])
    previous_minco = np.array([1.0, 2.0, 3.0])
    previous_mpc = np.array([4.0, 5.0, 6.0])
    last_tau = np.array([20.0, 0.0, 5.0])

    selected = select_boundary_acceleration(
        z,
        z_dot,
        params,
        previous_minco_z_ddot=previous_minco,
        previous_mpc_z_ddot=previous_mpc,
        last_applied_tau=last_tau,
    )
    assert selected.source == "previous_minco"
    assert selected.source_id == BOUNDARY_ACCEL_SOURCE_IDS["previous_minco"]
    assert np.allclose(selected.z_ddot, previous_minco)

    selected = select_boundary_acceleration(
        z,
        z_dot,
        params,
        previous_mpc_z_ddot=previous_mpc,
        last_applied_tau=last_tau,
    )
    assert selected.source == "previous_mpc"
    assert np.allclose(selected.z_ddot, previous_mpc)

    selected = select_boundary_acceleration(
        z,
        z_dot,
        params,
        last_applied_tau=last_tau,
    )
    assert selected.source == "last_command"
    assert np.allclose(
        selected.z_ddot,
        nominal_flat_acceleration_from_tau(z, z_dot, last_tau, params),
    )

    selected = select_boundary_acceleration(z, z_dot, params)
    assert selected.source == "zero"
    assert np.allclose(selected.z_ddot, np.zeros(3))


def test_minco_local_reference_generation_exports_feasible_reference():
    params = UsvModelParams(min_thrust=-500.0, max_thrust=500.0)
    z = np.array([0.0, 0.0, 0.1])
    z_dot = np.array([0.15, 0.0, 0.02])
    boundary = BoundaryAccelerationSelection(
        z_ddot=np.array([0.01, -0.02, 0.003]),
        source="previous_mpc",
    )
    config = MincoLocalReferenceConfig(
        terminal_offset_body=(2.5, 0.5, 0.25),
        dt=0.2,
        piece_count=3,
        fixed_total_time=5.0,
        tau_v_bar=1e5,
        velocity_bounds=(10.0, 10.0, 10.0),
        acceleration_bounds=(10.0, 10.0, 10.0),
        lambda_tau_v=0.0,
        lambda_velocity=0.0,
        lambda_acceleration=0.0,
        lambda_actuator=0.0,
        quadrature_order=4,
        max_iterations=30,
    )

    result = create_minco_local_reference(
        "minco_phase7_test",
        z,
        z_dot,
        boundary,
        params,
        config,
    )
    reference = result.reference
    assert reference.name == "minco_phase7_test"
    assert reference.diagnostics["feasible_reference_exported"] == 1.0
    assert reference.diagnostics["reference_source_minco"] == 1.0
    assert reference.diagnostics["minco_optimizer_success"] == 1.0
    assert reference.diagnostics["minco_boundary_acceleration_source_id"] == (
        BOUNDARY_ACCEL_SOURCE_IDS["previous_mpc"]
    )
    assert np.allclose(reference.z[0], z)
    assert np.allclose(reference.z_dot[0], z_dot)
    assert np.allclose(reference.z_ddot[0], boundary.z_ddot)
    assert reference.horizon_at_offsets(0.0, [0.0, 0.2])["z"].shape == (2, 3)
