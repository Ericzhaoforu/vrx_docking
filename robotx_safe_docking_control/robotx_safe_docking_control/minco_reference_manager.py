from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .minco_reference_generation import BoundaryAccelerationSelection
from .minco_reference_generation import MincoReferenceBuildResult
from .usv_flatness import wrap_angle


REF_REJECT_NONE = 0.0
REF_REJECT_OPTIMIZER = 1.0
REF_REJECT_PVA_RESIDUAL = 2.0
REF_REJECT_START_Z = 3.0
REF_REJECT_START_Z_DOT = 4.0
REF_REJECT_TAU_V = 5.0
REF_REJECT_VELOCITY = 6.0
REF_REJECT_ACCELERATION = 7.0
REF_REJECT_ACTUATOR = 8.0
REF_REJECT_PENALTY = 9.0

REPLAN_REASON_NONE = 0.0
REPLAN_REASON_DISABLED_OR_NO_REFERENCE = 1.0
REPLAN_REASON_REFERENCE_START_MISSING = 2.0
REPLAN_REASON_FIRST_REPLAN = 3.0
REPLAN_REASON_CHECK_INTERVAL = 4.0
REPLAN_REASON_NEAR_GOAL = 5.0
REPLAN_REASON_PROGRESS_GATE = 6.0
REPLAN_REASON_PROGRESS_NOT_MET = 7.0


@dataclass
class MincoReferenceManagerConfig:
    enabled: bool = False
    check_interval: float = 1.0
    progress_distance: float = 0.5
    progress_yaw: float = 0.12
    no_replan_goal_distance: float = 0.5
    no_replan_goal_yaw: float = 0.08
    modes: tuple[str, ...] = ("straight", "gentle_arc")
    pva_residual_tolerance: float = 1e-5
    start_z_tolerance: float = 5e-2
    start_z_dot_tolerance: float = 5e-2
    tau_v_diagnostic_bound: float = 250.0
    velocity_violation_tolerance: float = 5e-2
    acceleration_violation_tolerance: float = 5e-2
    actuator_violation_tolerance: float = 2.0
    penalty_total_tolerance: float = 1e7


@dataclass
class MincoReferenceValidation:
    accepted: bool
    reason: float
    metrics: dict[str, float]


class ActiveMincoReferenceManager:
    """Tracks online MINCO candidate validation and active-reference swaps."""

    def __init__(self, config: MincoReferenceManagerConfig):
        self.config = config
        self.candidate_count = 0
        self.swap_count = 0
        self.reject_count = 0
        self.solver_failure_after_swap_count = 0
        self.last_swap_flag = 0.0
        self.last_reject_flag = 0.0
        self.last_reject_reason = REF_REJECT_NONE
        self.last_mode = ""
        self.last_mode_id = 0.0
        self.last_boundary_source_id = 0.0
        self.last_replan_time = None
        self.last_swap_time = None
        self.pending_swap_solver_check = False
        self.pending_swap_thrust_before = None
        self.last_swap_thrust_jump = 0.0
        self.max_swap_thrust_jump = 0.0
        self.last_should_replan = 0.0
        self.last_should_replan_reason = REPLAN_REASON_NONE
        self.last_should_replan_time_since_last = 0.0
        self.last_should_replan_progress = 0.0
        self.last_should_replan_yaw_progress = 0.0
        self.last_should_replan_goal_distance = -1.0
        self.last_should_replan_goal_yaw_error = -1.0
        self.active_metrics = self._zero_metrics()

    @classmethod
    def from_values(
        cls,
        *,
        enabled: bool,
        check_rate_hz: float,
        progress_distance: float,
        progress_yaw: float,
        no_replan_goal_distance: float,
        no_replan_goal_yaw: float,
        modes: Iterable[str],
        pva_residual_tolerance: float,
        start_z_tolerance: float,
        start_z_dot_tolerance: float,
        tau_v_diagnostic_bound: float,
        velocity_violation_tolerance: float,
        acceleration_violation_tolerance: float,
        actuator_violation_tolerance: float,
        penalty_total_tolerance: float,
    ) -> "ActiveMincoReferenceManager":
        cleaned_modes = tuple(
            item.strip().lower()
            for item in modes
            if item and item.strip()
        )
        if not cleaned_modes:
            cleaned_modes = ("straight", "gentle_arc")
        check_rate_hz = max(float(check_rate_hz), 1e-3)
        return cls(MincoReferenceManagerConfig(
            enabled=bool(enabled),
            check_interval=1.0 / check_rate_hz,
            progress_distance=max(float(progress_distance), 0.0),
            progress_yaw=max(float(progress_yaw), 0.0),
            no_replan_goal_distance=max(float(no_replan_goal_distance), 0.0),
            no_replan_goal_yaw=max(float(no_replan_goal_yaw), 0.0),
            modes=cleaned_modes,
            pva_residual_tolerance=max(float(pva_residual_tolerance), 0.0),
            start_z_tolerance=max(float(start_z_tolerance), 0.0),
            start_z_dot_tolerance=max(float(start_z_dot_tolerance), 0.0),
            tau_v_diagnostic_bound=max(float(tau_v_diagnostic_bound), 0.0),
            velocity_violation_tolerance=max(
                float(velocity_violation_tolerance), 0.0),
            acceleration_violation_tolerance=max(
                float(acceleration_violation_tolerance), 0.0),
            actuator_violation_tolerance=max(
                float(actuator_violation_tolerance), 0.0),
            penalty_total_tolerance=max(float(penalty_total_tolerance), 0.0),
        ))

    def begin_cycle(self):
        self.last_swap_flag = 0.0
        self.last_reject_flag = 0.0

    def should_replan(
        self,
        now: float,
        reference_start_time,
        reference,
        goal_z=None,
    ) -> bool:
        if not self.config.enabled or reference is None:
            self._record_should_replan(
                False, REPLAN_REASON_DISABLED_OR_NO_REFERENCE)
            return False
        if reference_start_time is None:
            self._record_should_replan(
                True, REPLAN_REASON_REFERENCE_START_MISSING)
            return True
        elapsed = max(float(now) - float(reference_start_time), 0.0)
        if self.last_replan_time is None:
            self._record_should_replan(True, REPLAN_REASON_FIRST_REPLAN)
            return True
        time_since_last = float(now) - float(self.last_replan_time)
        if time_since_last < self.config.check_interval:
            self._record_should_replan(
                False,
                REPLAN_REASON_CHECK_INTERVAL,
                time_since_last=time_since_last)
            return False

        goal_distance = -1.0
        goal_yaw_error = -1.0
        if goal_z is not None:
            goal = np.asarray(goal_z, dtype=float).reshape(3)
            ref_now = np.asarray(reference.sample(elapsed)["z"], dtype=float)
            goal_distance = float(np.linalg.norm(goal[:2] - ref_now[:2]))
            goal_yaw_error = abs(wrap_angle(float(goal[2] - ref_now[2])))
            if (
                    goal_distance <= self.config.no_replan_goal_distance and
                    goal_yaw_error <= self.config.no_replan_goal_yaw):
                self._record_should_replan(
                    False,
                    REPLAN_REASON_NEAR_GOAL,
                    time_since_last=time_since_last,
                    goal_distance=goal_distance,
                    goal_yaw_error=goal_yaw_error)
                return False

        ref_start = np.asarray(reference.sample(0.0)["z"], dtype=float)
        ref_now = np.asarray(reference.sample(elapsed)["z"], dtype=float)
        progress = float(np.linalg.norm(ref_now[:2] - ref_start[:2]))
        yaw_progress = abs(wrap_angle(float(ref_now[2] - ref_start[2])))
        decision = bool(
            progress >= self.config.progress_distance or
            yaw_progress >= self.config.progress_yaw)
        self._record_should_replan(
            decision,
            (
                REPLAN_REASON_PROGRESS_GATE
                if decision else REPLAN_REASON_PROGRESS_NOT_MET),
            time_since_last=time_since_last,
            progress=progress,
            yaw_progress=yaw_progress,
            goal_distance=goal_distance,
            goal_yaw_error=goal_yaw_error)
        return decision

    def _record_should_replan(
        self,
        decision: bool,
        reason: float,
        *,
        time_since_last: float = 0.0,
        progress: float = 0.0,
        yaw_progress: float = 0.0,
        goal_distance: float = -1.0,
        goal_yaw_error: float = -1.0,
    ):
        self.last_should_replan = 1.0 if decision else 0.0
        self.last_should_replan_reason = float(reason)
        self.last_should_replan_time_since_last = float(time_since_last)
        self.last_should_replan_progress = float(progress)
        self.last_should_replan_yaw_progress = float(yaw_progress)
        self.last_should_replan_goal_distance = float(goal_distance)
        self.last_should_replan_goal_yaw_error = float(goal_yaw_error)

    def mark_replan_attempt(self, now: float):
        self.last_replan_time = float(now)

    def next_mode(self, default_name: str) -> str:
        name = default_name.strip().lower()
        if name in (
                "straight", "gentle_arc", "arc", "local_offset", "stop",
                "goal_lattice", "lattice", "goal"):
            return name
        index = self.candidate_count % len(self.config.modes)
        return self.config.modes[index]

    def validate(
        self,
        result: MincoReferenceBuildResult,
        state,
        boundary_acceleration: BoundaryAccelerationSelection,
    ) -> MincoReferenceValidation:
        self.candidate_count += 1
        diagnostics = dict(result.reference.diagnostics)
        metrics = self._candidate_metrics(result, state, boundary_acceleration)

        checks = [
            (
                metrics["reference_pva_residual"] <=
                self.config.pva_residual_tolerance,
                REF_REJECT_PVA_RESIDUAL,
            ),
            (
                metrics["reference_start_z_error"] <=
                self.config.start_z_tolerance,
                REF_REJECT_START_Z,
            ),
            (
                metrics["reference_start_z_dot_error"] <=
                self.config.start_z_dot_tolerance,
                REF_REJECT_START_Z_DOT,
            ),
            (
                metrics["reference_max_abs_tau_v"] <=
                self.config.tau_v_diagnostic_bound,
                REF_REJECT_TAU_V,
            ),
            (
                metrics["reference_velocity_violation"] <=
                self.config.velocity_violation_tolerance,
                REF_REJECT_VELOCITY,
            ),
            (
                metrics["reference_acceleration_violation"] <=
                self.config.acceleration_violation_tolerance,
                REF_REJECT_ACCELERATION,
            ),
            (
                metrics["reference_actuator_bound_violation"] <=
                self.config.actuator_violation_tolerance,
                REF_REJECT_ACTUATOR,
            ),
            (
                metrics["reference_penalty_total"] <=
                self.config.penalty_total_tolerance,
                REF_REJECT_PENALTY,
            ),
        ]
        for passed, reason in checks:
            if not passed:
                self.reject(reason, metrics)
                return MincoReferenceValidation(False, reason, metrics)

        self.active_metrics = metrics
        self.last_boundary_source_id = float(
            boundary_acceleration.source_id)
        self.last_reject_reason = REF_REJECT_NONE
        return MincoReferenceValidation(True, REF_REJECT_NONE, metrics)

    def record_optimizer_failure(self, reason: float = REF_REJECT_OPTIMIZER):
        self.candidate_count += 1
        self.reject(reason, self._zero_metrics())

    def accept(
        self,
        *,
        now: float,
        mode: str,
        left_before: float,
        right_before: float,
        is_initial: bool = False,
    ):
        self.last_replan_time = float(now)
        self.last_mode = mode
        self.last_mode_id = self.mode_id(mode)
        if is_initial:
            self.last_swap_flag = 0.0
            self.pending_swap_solver_check = False
            self.pending_swap_thrust_before = None
        else:
            self.swap_count += 1
            self.last_swap_flag = 1.0
            self.last_swap_time = float(now)
            self.pending_swap_solver_check = True
            self.pending_swap_thrust_before = (
                float(left_before), float(right_before))
        self.last_swap_thrust_jump = 0.0

    def reject(self, reason: float, metrics: dict[str, float]):
        self.reject_count += 1
        self.last_reject_flag = 1.0
        self.last_reject_reason = float(reason)

    def record_solver_after_swap(self, success: bool):
        if not self.pending_swap_solver_check:
            return
        if not success:
            self.solver_failure_after_swap_count += 1
        self.pending_swap_solver_check = False

    def record_swap_thrust(self, left_cmd: float, right_cmd: float):
        if self.pending_swap_thrust_before is None:
            return
        left_before, right_before = self.pending_swap_thrust_before
        jump = max(
            abs(float(left_cmd) - left_before),
            abs(float(right_cmd) - right_before),
        )
        self.last_swap_thrust_jump = float(jump)
        self.max_swap_thrust_jump = max(self.max_swap_thrust_jump, float(jump))
        self.pending_swap_thrust_before = None

    def debug_values(self, now: float) -> dict[str, float]:
        values = {
            "reference_candidate_count": float(self.candidate_count),
            "reference_swap_count": float(self.swap_count),
            "reference_reject_count": float(self.reject_count),
            "reference_last_swap": float(self.last_swap_flag),
            "reference_last_reject": float(self.last_reject_flag),
            "reference_last_reject_reason": float(self.last_reject_reason),
            "reference_mode_id": float(self.last_mode_id),
            "reference_boundary_accel_source": float(
                self.last_boundary_source_id),
            "reference_solver_failure_after_swap_count": float(
                self.solver_failure_after_swap_count),
            "reference_last_swap_thrust_jump": float(
                self.last_swap_thrust_jump),
            "reference_max_swap_thrust_jump": float(
                self.max_swap_thrust_jump),
            "reference_time_since_swap": (
                float(now) - float(self.last_swap_time)
                if self.last_swap_time is not None else 0.0),
            "reference_should_replan": float(self.last_should_replan),
            "reference_should_replan_reason": float(
                self.last_should_replan_reason),
            "reference_should_replan_time_since_last": float(
                self.last_should_replan_time_since_last),
            "reference_should_replan_progress": float(
                self.last_should_replan_progress),
            "reference_should_replan_yaw_progress": float(
                self.last_should_replan_yaw_progress),
            "reference_should_replan_goal_distance": float(
                self.last_should_replan_goal_distance),
            "reference_should_replan_goal_yaw_error": float(
                self.last_should_replan_goal_yaw_error),
        }
        values.update(self.active_metrics)
        return values

    @staticmethod
    def mode_id(mode: str) -> float:
        mapping = {
            "straight": 1.0,
            "stop": 1.0,
            "gentle_arc": 2.0,
            "arc": 2.0,
            "local_offset": 2.0,
            "hold": 3.0,
            "yaw": 4.0,
            "goal_lattice": 5.0,
            "lattice": 5.0,
            "goal": 5.0,
        }
        return mapping.get(mode.strip().lower(), 0.0)

    @staticmethod
    def _zero_metrics() -> dict[str, float]:
        return {
            "reference_start_z_error": 0.0,
            "reference_start_z_dot_error": 0.0,
            "reference_start_z_ddot_error": 0.0,
            "reference_pva_residual": 0.0,
            "reference_continuity_residual": 0.0,
            "reference_max_abs_tau_v": 0.0,
            "reference_rms_tau_v": 0.0,
            "reference_velocity_violation": 0.0,
            "reference_acceleration_violation": 0.0,
            "reference_actuator_bound_violation": 0.0,
            "reference_penalty_total": 0.0,
            "frontend_mode": 0.0,
            "frontend_goal_distance": 0.0,
            "frontend_goal_inside_horizon": 0.0,
            "frontend_goal_position_error": 0.0,
            "frontend_goal_yaw_error": 0.0,
            "frontend_selected_depth": 0.0,
            "frontend_collision_free": 0.0,
            "frontend_terminal_stop": 0.0,
            "frontend_candidate_count": 0.0,
            "frontend_collision_reject_count": 0.0,
            "frontend_infeasible_reject_count": 0.0,
            "frontend_selected_cost": 0.0,
            "frontend_expansion_count": 0.0,
            "frontend_analytic_attempted": 0.0,
            "frontend_analytic_attempt_count": 0.0,
            "frontend_analytic_success": 0.0,
            "frontend_reach_goal": 0.0,
            "frontend_reach_horizon": 0.0,
            "frontend_tau_v_bar": 0.0,
            "frontend_control_discretization": 0.0,
            "frontend_terminal_x": 0.0,
            "frontend_terminal_y": 0.0,
            "frontend_terminal_psi": 0.0,
            "frontend_global_goal_x": 0.0,
            "frontend_global_goal_y": 0.0,
            "frontend_global_goal_psi": 0.0,
            "frontend_local_goal_x": 0.0,
            "frontend_local_goal_y": 0.0,
            "frontend_local_goal_psi": 0.0,
        }

    @staticmethod
    def _candidate_metrics(
        result: MincoReferenceBuildResult,
        state,
        boundary_acceleration: BoundaryAccelerationSelection,
    ) -> dict[str, float]:
        reference = result.reference
        diagnostics = reference.diagnostics
        start_z_error = np.asarray(reference.z[0] - state["z"], dtype=float)
        start_z_error[2] = wrap_angle(float(start_z_error[2]))
        start_z_dot_error = np.asarray(
            reference.z_dot[0] - state["z_dot"], dtype=float)
        start_z_ddot_error = np.asarray(
            reference.z_ddot[0] - boundary_acceleration.z_ddot,
            dtype=float)
        return {
            "reference_start_z_error": float(np.linalg.norm(start_z_error)),
            "reference_start_z_dot_error": float(
                np.linalg.norm(start_z_dot_error)),
            "reference_start_z_ddot_error": float(
                np.linalg.norm(start_z_ddot_error)),
            "reference_pva_residual": float(
                diagnostics.get("pva_residual", 0.0)),
            "reference_continuity_residual": float(
                diagnostics.get("continuity_residual", 0.0)),
            "reference_max_abs_tau_v": float(
                diagnostics.get("max_abs_tau_v", 0.0)),
            "reference_rms_tau_v": float(
                diagnostics.get("rms_tau_v", 0.0)),
            "reference_velocity_violation": float(
                diagnostics.get("velocity_violation", 0.0)),
            "reference_acceleration_violation": float(
                diagnostics.get("acceleration_violation", 0.0)),
            "reference_actuator_bound_violation": float(
                diagnostics.get("actuator_bound_violation", 0.0)),
            "reference_penalty_total": float(
                diagnostics.get("minco_penalty_total", 0.0)),
            "frontend_mode": float(
                diagnostics.get("frontend_mode", 0.0)),
            "frontend_goal_distance": float(
                diagnostics.get("frontend_goal_distance", 0.0)),
            "frontend_goal_inside_horizon": float(
                diagnostics.get("frontend_goal_inside_horizon", 0.0)),
            "frontend_goal_position_error": float(
                diagnostics.get("frontend_goal_position_error", 0.0)),
            "frontend_goal_yaw_error": float(
                diagnostics.get("frontend_goal_yaw_error", 0.0)),
            "frontend_selected_depth": float(
                diagnostics.get("frontend_selected_depth", 0.0)),
            "frontend_collision_free": float(
                diagnostics.get("frontend_collision_free", 0.0)),
            "frontend_terminal_stop": float(
                diagnostics.get("frontend_terminal_stop", 0.0)),
            "frontend_candidate_count": float(
                diagnostics.get("frontend_candidate_count", 0.0)),
            "frontend_collision_reject_count": float(
                diagnostics.get("frontend_collision_reject_count", 0.0)),
            "frontend_infeasible_reject_count": float(
                diagnostics.get("frontend_infeasible_reject_count", 0.0)),
            "frontend_selected_cost": float(
                diagnostics.get("frontend_selected_cost", 0.0)),
            "frontend_expansion_count": float(
                diagnostics.get("frontend_expansion_count", 0.0)),
            "frontend_analytic_attempted": float(
                diagnostics.get("frontend_analytic_attempted", 0.0)),
            "frontend_analytic_attempt_count": float(
                diagnostics.get("frontend_analytic_attempt_count", 0.0)),
            "frontend_analytic_success": float(
                diagnostics.get("frontend_analytic_success", 0.0)),
            "frontend_reach_goal": float(
                diagnostics.get("frontend_reach_goal", 0.0)),
            "frontend_reach_horizon": float(
                diagnostics.get("frontend_reach_horizon", 0.0)),
            "frontend_tau_v_bar": float(
                diagnostics.get("frontend_tau_v_bar", 0.0)),
            "frontend_control_discretization": float(
                diagnostics.get("frontend_control_discretization", 0.0)),
            "frontend_terminal_x": float(
                diagnostics.get("frontend_terminal_x", 0.0)),
            "frontend_terminal_y": float(
                diagnostics.get("frontend_terminal_y", 0.0)),
            "frontend_terminal_psi": float(
                diagnostics.get("frontend_terminal_psi", 0.0)),
            "frontend_global_goal_x": float(
                diagnostics.get("frontend_global_goal_x", 0.0)),
            "frontend_global_goal_y": float(
                diagnostics.get("frontend_global_goal_y", 0.0)),
            "frontend_global_goal_psi": float(
                diagnostics.get("frontend_global_goal_psi", 0.0)),
            "frontend_local_goal_x": float(
                diagnostics.get("frontend_local_goal_x", 0.0)),
            "frontend_local_goal_y": float(
                diagnostics.get("frontend_local_goal_y", 0.0)),
            "frontend_local_goal_psi": float(
                diagnostics.get("frontend_local_goal_psi", 0.0)),
        }
