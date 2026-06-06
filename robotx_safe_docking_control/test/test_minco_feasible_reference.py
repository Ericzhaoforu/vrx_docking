import numpy as np

from robotx_safe_docking_control.minco_feasible_reference import (
    minco_to_feasible_reference,
)
from robotx_safe_docking_control.minco_usv_optimizer import (
    MincoUsvOptimizer,
    MincoUsvOptimizerConfig,
)
from robotx_safe_docking_control.minco_usv_penalties import MincoUsvPenaltyConfig
from robotx_safe_docking_control.usv_flatness import (
    UsvModelParams,
    generalized_force_to_thrust,
    theta_tau_nominal,
)


def make_optimizer_result():
    params = UsvModelParams(min_thrust=-500.0, max_thrust=500.0)
    penalty_config = MincoUsvPenaltyConfig(
        params=params,
        tau_v_bar=1e5,
        velocity_bounds=(10.0, 10.0, 10.0),
        acceleration_bounds=(10.0, 10.0, 10.0),
        lambda_tau_v=0.0,
        lambda_velocity=0.0,
        lambda_acceleration=0.0,
        lambda_actuator=0.0,
        quadrature_order=4,
    )
    optimizer = MincoUsvOptimizer(
        penalty_config,
        MincoUsvOptimizerConfig(
            piece_count=3,
            fixed_total_time=5.0,
            max_iterations=20,
        ),
    )
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
    return optimizer.optimize(head_pva, tail_pva), params, penalty_config


def test_minco_feasible_reference_export_contract():
    result, params, penalty_config = make_optimizer_result()
    reference = minco_to_feasible_reference(
        "minco_test",
        result.trajectory,
        params,
        dt=0.2,
        penalty_config=penalty_config,
        optimizer_result=result,
    )
    assert reference.name == "minco_test"
    assert reference.duration == result.trajectory.duration()
    assert reference.z.shape == reference.z_dot.shape == reference.z_ddot.shape
    assert reference.nu.shape == reference.nu_dot.shape == reference.tau.shape
    assert reference.thrust.shape == (len(reference.t), 2)
    assert reference.diagnostics["feasible_reference_exported"] == 1.0
    assert reference.diagnostics["minco_optimizer_success"] == 1.0
    assert reference.diagnostics["minco_objective_after"] < (
        reference.diagnostics["minco_objective_before"]
    )


def test_minco_feasible_reference_tau_and_thrust_consistency():
    result, params, penalty_config = make_optimizer_result()
    reference = minco_to_feasible_reference(
        "minco_test",
        result.trajectory,
        params,
        dt=0.25,
        penalty_config=penalty_config,
        optimizer_result=result,
    )
    tau_check = np.asarray([
        theta_tau_nominal(z, z_dot, z_ddot, params)
        for z, z_dot, z_ddot in zip(reference.z, reference.z_dot, reference.z_ddot)
    ])
    thrust_check = np.asarray([
        generalized_force_to_thrust(tau[0], tau[2], params)
        for tau in reference.tau
    ])
    assert np.allclose(reference.tau, tau_check)
    assert np.allclose(reference.thrust, thrust_check)


def test_minco_feasible_reference_sampling_and_horizon_keys():
    result, params, penalty_config = make_optimizer_result()
    reference = minco_to_feasible_reference(
        "minco_test",
        result.trajectory,
        params,
        dt=0.2,
        penalty_config=penalty_config,
        optimizer_result=result,
    )
    sample = reference.sample(0.35)
    horizon = reference.horizon_at_offsets(0.0, [0.0, 0.2, 0.4])
    expected_keys = {"z", "z_dot", "z_ddot", "nu", "nu_dot", "tau", "thrust"}
    assert set(sample.keys()) == expected_keys
    assert set(horizon.keys()) == expected_keys
    assert horizon["z"].shape == (3, 3)
    assert horizon["thrust"].shape == (3, 2)
