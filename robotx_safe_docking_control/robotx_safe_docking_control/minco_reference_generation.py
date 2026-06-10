from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

from .feasible_references import FeasibleReference
from .minco_feasible_reference import minco_to_feasible_reference
from .minco_offline_validation import minco_residuals
from .minco_usv_optimizer import MincoUsvOptimizer
from .minco_usv_optimizer import MincoUsvOptimizerConfig
from .minco_usv_penalties import MincoUsvPenaltyConfig
from .usv_flatness import UsvModelParams
from .usv_flatness import body_to_flat_acceleration
from .usv_flatness import flat_to_body_velocity
from .usv_flatness import nominal_coriolis_times_velocity
from .usv_flatness import nominal_damping

try:
    import robotx_minco_cpp as _minco_cpp
except Exception:  # pragma: no cover - optional compiled runtime
    try:
        from . import _minco_cpp
    except Exception:
        _minco_cpp = None


BOUNDARY_ACCEL_SOURCE_IDS = {
    "previous_minco": 1.0,
    "previous_mpc": 2.0,
    "last_command": 3.0,
    "zero": 4.0,
    "active_reference": 5.0,
    "committed_reference": 5.0,
}


@dataclass
class BoundaryAccelerationSelection:
    z_ddot: np.ndarray
    source: str

    @property
    def source_id(self) -> float:
        return BOUNDARY_ACCEL_SOURCE_IDS[self.source]


@dataclass
class MincoLocalReferenceConfig:
    terminal_offset_body: tuple[float, float, float] = (2.0, 0.4, 0.25)
    terminal_z: Optional[tuple[float, float, float]] = None
    terminal_z_dot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    terminal_z_ddot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dt: float = 0.1
    piece_count: int = 2
    fixed_total_time: Optional[float] = 8.0
    reference_speed: float = 0.6
    time_dilation: float = 1.5
    min_segment_time: float = 0.3
    smooth_weight: float = 1.0
    time_weight: float = 0.0
    max_iterations: int = 20
    tau_v_bar: float = 250.0
    velocity_bounds: tuple[float, float, float] = (2.0, 0.5, 0.6)
    acceleration_bounds: tuple[float, float, float] = (0.9, 0.5, 0.8)
    lambda_tau_v: float = 1.0
    lambda_velocity: float = 1.0
    lambda_acceleration: float = 1.0
    lambda_actuator: float = 10.0
    quadrature_order: int = 8
    penalty_mu: float = 20.0
    initial_inPs: Optional[np.ndarray] = None
    initial_ts: Optional[np.ndarray] = None


@dataclass
class MincoReferenceBuildResult:
    reference: object
    optimizer_result: object
    boundary_acceleration_source: str


@dataclass
class CppMincoReferenceOptimizerResult:
    success: bool
    message: str
    objective_before: float
    objective_after: float
    iterations: int
    inPs: np.ndarray
    ts: np.ndarray
    theta: np.ndarray
    trajectory: object
    diagnostics: dict


def nominal_flat_acceleration_from_tau(
    z,
    z_dot,
    tau,
    params: UsvModelParams,
) -> np.ndarray:
    """Compute flat acceleration from nominal USV dynamics and a command."""
    z = np.asarray(z, dtype=float).reshape(3)
    z_dot = np.asarray(z_dot, dtype=float).reshape(3)
    tau = np.asarray(tau, dtype=float).reshape(3)
    nu = flat_to_body_velocity(z, z_dot)
    rhs = (
        tau -
        nominal_coriolis_times_velocity(nu, params) -
        nominal_damping(nu, params)
    )
    nu_dot = np.linalg.solve(params.mass_matrix, rhs)
    return body_to_flat_acceleration(float(z[2]), nu, nu_dot)


def select_boundary_acceleration(
    z,
    z_dot,
    params: UsvModelParams,
    previous_minco_z_ddot=None,
    previous_mpc_z_ddot=None,
    last_applied_tau=None,
) -> BoundaryAccelerationSelection:
    """Apply the required MINCO boundary-acceleration fallback policy."""
    for source, candidate in (
        ("previous_minco", previous_minco_z_ddot),
        ("previous_mpc", previous_mpc_z_ddot),
    ):
        if _valid_vector(candidate):
            return BoundaryAccelerationSelection(
                z_ddot=np.asarray(candidate, dtype=float).reshape(3),
                source=source,
            )

    if _valid_vector(last_applied_tau):
        return BoundaryAccelerationSelection(
            z_ddot=nominal_flat_acceleration_from_tau(
                z, z_dot, last_applied_tau, params
            ),
            source="last_command",
        )

    return BoundaryAccelerationSelection(
        z_ddot=np.zeros(3, dtype=float),
        source="zero",
    )


def create_minco_local_reference(
    name: str,
    z,
    z_dot,
    boundary_acceleration: BoundaryAccelerationSelection,
    params: UsvModelParams,
    config: Optional[MincoLocalReferenceConfig] = None,
) -> MincoReferenceBuildResult:
    """Build a local MINCO reference compatible with the existing NMPC tracker."""
    config = config or MincoLocalReferenceConfig()
    z = np.asarray(z, dtype=float).reshape(3)
    z_dot = np.asarray(z_dot, dtype=float).reshape(3)
    z_ddot = np.asarray(boundary_acceleration.z_ddot, dtype=float).reshape(3)

    if config.terminal_z is None:
        terminal_offset = np.asarray(
            config.terminal_offset_body, dtype=float).reshape(3)
        c = float(np.cos(z[2]))
        s = float(np.sin(z[2]))
        terminal_xy = z[:2] + np.array([
            c * terminal_offset[0] - s * terminal_offset[1],
            s * terminal_offset[0] + c * terminal_offset[1],
        ])
        terminal_z = np.array([
            terminal_xy[0],
            terminal_xy[1],
            z[2] + terminal_offset[2],
        ], dtype=float)
    else:
        terminal_z = np.asarray(config.terminal_z, dtype=float).reshape(3)

    head_pva = np.vstack((z, z_dot, z_ddot))
    tail_pva = np.vstack((
        terminal_z,
        np.asarray(config.terminal_z_dot, dtype=float).reshape(3),
        np.asarray(config.terminal_z_ddot, dtype=float).reshape(3),
    ))

    penalty_config = MincoUsvPenaltyConfig(
        params=params,
        tau_v_bar=float(config.tau_v_bar),
        velocity_bounds=tuple(config.velocity_bounds),
        acceleration_bounds=tuple(config.acceleration_bounds),
        lambda_tau_v=float(config.lambda_tau_v),
        lambda_velocity=float(config.lambda_velocity),
        lambda_acceleration=float(config.lambda_acceleration),
        lambda_actuator=float(config.lambda_actuator),
        quadrature_order=int(config.quadrature_order),
        penalty_mu=float(config.penalty_mu),
    )
    optimizer = MincoUsvOptimizer(
        penalty_config,
        MincoUsvOptimizerConfig(
            piece_count=int(config.piece_count),
            fixed_total_time=config.fixed_total_time,
            reference_speed=float(config.reference_speed),
            time_dilation=float(config.time_dilation),
            min_segment_time=float(config.min_segment_time),
            smooth_weight=float(config.smooth_weight),
            time_weight=float(config.time_weight),
            max_iterations=int(config.max_iterations),
        ),
    )
    use_cpp_planner = (
        _minco_cpp is not None and
        hasattr(_minco_cpp, "plan_minco_reference") and
        os.environ.get("ROBOTX_MINCO_PLAN_CPP", "1") != "0" and
        os.environ.get("ROBOTX_MINCO_USE_CPP", "1") != "0")
    if use_cpp_planner:
        reference, optimizer_result = _build_cpp_minco_reference(
            name=name,
            optimizer=optimizer,
            head_pva=head_pva,
            tail_pva=tail_pva,
            params=params,
            config=config,
            boundary_acceleration=boundary_acceleration,
            z_ddot=z_ddot,
        )
        return MincoReferenceBuildResult(
            reference=reference,
            optimizer_result=optimizer_result,
            boundary_acceleration_source=boundary_acceleration.source,
        )

    optimizer_result = optimizer.optimize(
        head_pva,
        tail_pva,
        initial_inPs=config.initial_inPs,
        initial_ts=config.initial_ts,
    )
    if not optimizer_result.success:
        raise RuntimeError(
            "MINCO optimizer failed: " + str(optimizer_result.message)
        )

    reference = minco_to_feasible_reference(
        name,
        optimizer_result.trajectory,
        params,
        dt=float(config.dt),
        penalty_config=penalty_config,
        optimizer_result=optimizer_result,
    )
    reference.diagnostics.update(
        minco_residuals(optimizer_result.trajectory, head_pva, tail_pva)
    )
    reference.diagnostics.update({
        "reference_source_minco": 1.0,
        "minco_boundary_acceleration_source_id": (
            boundary_acceleration.source_id
        ),
        "minco_initial_z_ddot_norm": float(np.linalg.norm(z_ddot)),
    })
    return MincoReferenceBuildResult(
        reference=reference,
        optimizer_result=optimizer_result,
        boundary_acceleration_source=boundary_acceleration.source,
    )


def _build_cpp_minco_reference(
    name: str,
    optimizer: MincoUsvOptimizer,
    head_pva: np.ndarray,
    tail_pva: np.ndarray,
    params: UsvModelParams,
    config: MincoLocalReferenceConfig,
    boundary_acceleration: BoundaryAccelerationSelection,
    z_ddot: np.ndarray,
):
    context = optimizer._cpp_context()
    result = _minco_cpp.plan_minco_reference(
        np.asarray(head_pva, dtype=float),
        np.asarray(tail_pva, dtype=float),
        context["params"],
        context["optimizer"],
        context["penalty"],
        float(config.dt),
        None if config.initial_inPs is None else np.asarray(
            config.initial_inPs, dtype=float),
        None if config.initial_ts is None else np.asarray(
            config.initial_ts, dtype=float),
    )
    if not bool(result["success"]):
        raise RuntimeError(
            "C++ MINCO planner failed: " + str(result["message"])
        )
    diagnostics = {
        key: value
        for key, value in dict(result["diagnostics"]).items()
    }
    diagnostics.update({
        "reference_source_minco": 1.0,
        "minco_boundary_acceleration_source_id": (
            boundary_acceleration.source_id
        ),
        "minco_initial_z_ddot_norm": float(np.linalg.norm(z_ddot)),
    })
    reference = FeasibleReference(
        name=name,
        t=np.asarray(result["t"], dtype=float),
        z=np.asarray(result["z"], dtype=float),
        z_dot=np.asarray(result["z_dot"], dtype=float),
        z_ddot=np.asarray(result["z_ddot"], dtype=float),
        nu=np.asarray(result["nu"], dtype=float),
        nu_dot=np.asarray(result["nu_dot"], dtype=float),
        tau=np.asarray(result["tau"], dtype=float),
        thrust=np.asarray(result["thrust"], dtype=float),
        diagnostics=diagnostics,
    )
    optimizer_result = CppMincoReferenceOptimizerResult(
        success=bool(result["success"]),
        message=str(result["message"]),
        objective_before=float(result["objective_before"]),
        objective_after=float(result["objective_after"]),
        iterations=int(result["iterations"]),
        inPs=np.asarray(result["inPs"], dtype=float),
        ts=np.asarray(result["ts"], dtype=float),
        theta=np.asarray(result["theta"], dtype=float),
        trajectory=None,
        diagnostics=diagnostics,
    )
    return reference, optimizer_result


def _valid_vector(value) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    return array.shape == (3,) and bool(np.all(np.isfinite(array)))
