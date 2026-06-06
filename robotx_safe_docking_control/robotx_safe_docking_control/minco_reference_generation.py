from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .minco_feasible_reference import minco_to_feasible_reference
from .minco_usv_optimizer import MincoUsvOptimizer
from .minco_usv_optimizer import MincoUsvOptimizerConfig
from .minco_usv_penalties import MincoUsvPenaltyConfig
from .usv_flatness import UsvModelParams
from .usv_flatness import body_to_flat_acceleration
from .usv_flatness import flat_to_body_velocity
from .usv_flatness import nominal_coriolis_times_velocity
from .usv_flatness import nominal_damping


BOUNDARY_ACCEL_SOURCE_IDS = {
    "previous_minco": 1.0,
    "previous_mpc": 2.0,
    "last_command": 3.0,
    "zero": 4.0,
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


@dataclass
class MincoReferenceBuildResult:
    reference: object
    optimizer_result: object
    boundary_acceleration_source: str


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

    terminal_offset = np.asarray(config.terminal_offset_body, dtype=float).reshape(3)
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
    optimizer_result = optimizer.optimize(head_pva, tail_pva)
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


def _valid_vector(value) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    return array.shape == (3,) and bool(np.all(np.isfinite(array)))
