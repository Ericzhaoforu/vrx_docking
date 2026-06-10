import numpy as np

from robotx_safe_docking_control.minco_reference_generation import (
    BoundaryAccelerationSelection,
    MincoLocalReferenceConfig,
    create_minco_local_reference,
)
from robotx_safe_docking_control.feasible_references import FeasibleReference
from robotx_safe_docking_control.minco_reference_manager import (
    ActiveMincoReferenceManager,
    REF_REJECT_TAU_V,
)
from robotx_safe_docking_control.usv_flatness import UsvModelParams


def make_candidate():
    params = UsvModelParams(min_thrust=-500.0, max_thrust=500.0)
    state = {
        "z": np.array([0.0, 0.0, 0.1], dtype=float),
        "z_dot": np.array([0.1, 0.0, 0.0], dtype=float),
    }
    boundary = BoundaryAccelerationSelection(
        z_ddot=np.zeros(3, dtype=float),
        source="zero",
    )
    config = MincoLocalReferenceConfig(
        terminal_offset_body=(1.5, 0.2, 0.1),
        dt=0.2,
        piece_count=2,
        fixed_total_time=5.0,
        tau_v_bar=1e5,
        velocity_bounds=(10.0, 10.0, 10.0),
        acceleration_bounds=(10.0, 10.0, 10.0),
        lambda_tau_v=0.0,
        lambda_velocity=0.0,
        lambda_acceleration=0.0,
        lambda_actuator=0.0,
        quadrature_order=4,
        max_iterations=20,
    )
    result = create_minco_local_reference(
        "manager_test", state["z"], state["z_dot"], boundary, params, config)
    return result, state, boundary


def manager(tau_v_bound=1e5):
    return ActiveMincoReferenceManager.from_values(
        enabled=True,
        check_rate_hz=1.0,
        progress_distance=0.5,
        progress_yaw=0.12,
        no_replan_goal_distance=0.5,
        no_replan_goal_yaw=0.08,
        modes=("straight", "gentle_arc"),
        pva_residual_tolerance=1e-5,
        start_z_tolerance=1e-6,
        start_z_dot_tolerance=1e-6,
        tau_v_diagnostic_bound=tau_v_bound,
        velocity_violation_tolerance=1e-6,
        acceleration_violation_tolerance=1e-6,
        actuator_violation_tolerance=1e-6,
        penalty_total_tolerance=1e9,
    )


def make_reference():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    z = np.column_stack((
        t,
        np.zeros_like(t),
        np.zeros_like(t),
    ))
    z_dot = np.column_stack((
        np.ones_like(t),
        np.zeros_like(t),
        np.zeros_like(t),
    ))
    zeros3 = np.zeros((len(t), 3), dtype=float)
    zeros2 = np.zeros((len(t), 2), dtype=float)
    return FeasibleReference(
        name="manager_reference",
        t=t,
        z=z,
        z_dot=z_dot,
        z_ddot=zeros3.copy(),
        nu=z_dot.copy(),
        nu_dot=zeros3.copy(),
        tau=zeros3.copy(),
        thrust=zeros2.copy(),
        diagnostics={},
    )


def make_yaw_reference():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    z = np.column_stack((
        np.zeros_like(t),
        np.zeros_like(t),
        0.2 * t,
    ))
    z_dot = np.column_stack((
        np.zeros_like(t),
        np.zeros_like(t),
        np.full_like(t, 0.2),
    ))
    zeros3 = np.zeros((len(t), 3), dtype=float)
    zeros2 = np.zeros((len(t), 2), dtype=float)
    return FeasibleReference(
        name="manager_yaw_reference",
        t=t,
        z=z,
        z_dot=z_dot,
        z_ddot=zeros3.copy(),
        nu=z_dot.copy(),
        nu_dot=zeros3.copy(),
        tau=zeros3.copy(),
        thrust=zeros2.copy(),
        diagnostics={},
    )


def test_manager_accepts_continuous_feasible_candidate():
    result, state, boundary = make_candidate()
    ref_manager = manager()
    validation = ref_manager.validate(result, state, boundary)
    assert validation.accepted
    assert validation.metrics["reference_start_z_error"] < 1e-9
    assert validation.metrics["reference_start_z_dot_error"] < 1e-9
    ref_manager.accept(
        now=1.0,
        mode="gentle_arc",
        left_before=0.0,
        right_before=0.0,
        is_initial=False,
    )
    assert ref_manager.swap_count == 1
    ref_manager.record_swap_thrust(5.0, -2.0)
    assert ref_manager.max_swap_thrust_jump == 5.0


def test_manager_rejects_candidate_above_tau_v_bound():
    result, state, boundary = make_candidate()
    ref_manager = manager(tau_v_bound=0.0)
    validation = ref_manager.validate(result, state, boundary)
    assert not validation.accepted
    assert validation.reason == REF_REJECT_TAU_V
    assert ref_manager.reject_count == 1


def test_rejected_candidate_does_not_replace_active_metrics():
    result, state, boundary = make_candidate()
    ref_manager = manager()
    validation = ref_manager.validate(result, state, boundary)
    assert validation.accepted
    active_tau_v = ref_manager.active_metrics["reference_max_abs_tau_v"]

    ref_manager.config.tau_v_diagnostic_bound = 0.0
    validation = ref_manager.validate(result, state, boundary)
    assert not validation.accepted
    assert ref_manager.active_metrics["reference_max_abs_tau_v"] == active_tau_v


def test_replan_check_rate_gates_attempts():
    ref_manager = manager()
    ref_manager.mark_replan_attempt(1.0)
    reference = make_reference()

    assert not ref_manager.should_replan(1.5, 0.0, reference)
    assert ref_manager.should_replan(2.1, 0.0, reference)


def test_replan_suppressed_near_reference_start():
    ref_manager = manager()
    ref_manager.config.progress_distance = 1.5
    ref_manager.mark_replan_attempt(0.0)
    reference = make_reference()

    assert not ref_manager.should_replan(1.0, 0.0, reference)


def test_replan_suppressed_near_goal():
    ref_manager = manager()
    ref_manager.mark_replan_attempt(0.0)
    reference = make_reference()
    goal = np.array([1.0, 0.0, 0.0], dtype=float)

    assert not ref_manager.should_replan(
        1.1, 0.0, reference, goal_z=goal)


def test_replan_triggered_by_yaw_progress():
    ref_manager = manager()
    ref_manager.config.progress_distance = 10.0
    ref_manager.mark_replan_attempt(0.0)
    reference = make_yaw_reference()

    assert ref_manager.should_replan(1.0, 0.0, reference)
