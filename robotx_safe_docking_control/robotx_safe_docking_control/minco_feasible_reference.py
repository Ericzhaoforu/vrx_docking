from __future__ import annotations

import numpy as np

from .feasible_references import FeasibleReference
from .feasible_references import reference_diagnostics
from .minco_s3nu import MINCO_S3NU
from .minco_usv_penalties import MincoUsvDensePenalty
from .minco_usv_penalties import MincoUsvPenaltyConfig
from .usv_flatness import UsvModelParams
from .usv_flatness import flat_to_body_acceleration
from .usv_flatness import flat_to_body_velocity
from .usv_flatness import generalized_force_to_thrust
from .usv_flatness import theta_tau_nominal


def minco_to_feasible_reference(
    name: str,
    trajectory: MINCO_S3NU,
    params: UsvModelParams,
    dt: float = 0.1,
    penalty_config: MincoUsvPenaltyConfig | None = None,
    optimizer_result=None,
) -> FeasibleReference:
    """Sample a MINCO trajectory into the existing NMPC reference interface."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    duration = trajectory.duration()
    t = _sample_times(duration, dt)
    z = np.zeros((len(t), 3), dtype=float)
    z_dot = np.zeros((len(t), 3), dtype=float)
    z_ddot = np.zeros((len(t), 3), dtype=float)
    nu = np.zeros((len(t), 3), dtype=float)
    nu_dot = np.zeros((len(t), 3), dtype=float)
    tau = np.zeros((len(t), 3), dtype=float)
    thrust = np.zeros((len(t), 2), dtype=float)

    for index, query_time in enumerate(t):
        sample = trajectory.sample(float(query_time))
        z[index] = sample.z
        z_dot[index] = sample.z_dot
        z_ddot[index] = sample.z_ddot
        nu[index] = flat_to_body_velocity(z[index], z_dot[index])
        nu_dot[index] = flat_to_body_acceleration(
            float(z[index, 2]), z_dot[index], z_ddot[index]
        )
        tau[index] = theta_tau_nominal(z[index], z_dot[index], z_ddot[index], params)
        thrust[index] = generalized_force_to_thrust(
            tau[index, 0], tau[index, 2], params
        )

    diagnostics = reference_diagnostics(tau, thrust, z_dot, nu)
    diagnostics.update({
        "feasible_reference_exported": 1.0,
        "minco_energy": trajectory.get_energy(),
        "minco_duration": duration,
        "minco_piece_count": float(trajectory.piece_count()),
        "minco_min_segment_time": float(np.min(trajectory.ts)),
        "minco_max_segment_time": float(np.max(trajectory.ts)),
    })

    if penalty_config is not None:
        penalty_eval = MincoUsvDensePenalty(penalty_config).evaluate(trajectory)
        diagnostics.update({
            "minco_penalty_total": penalty_eval.total,
            "velocity_violation": penalty_eval.diagnostics["velocity_violation"],
            "acceleration_violation": penalty_eval.diagnostics[
                "acceleration_violation"
            ],
            "actuator_bound_violation": penalty_eval.diagnostics[
                "actuator_bound_violation"
            ],
            "max_abs_tau_v": penalty_eval.diagnostics["max_abs_tau_v"],
            "rms_tau_v": penalty_eval.diagnostics["rms_tau_v"],
        })

    if optimizer_result is not None:
        diagnostics.update({
            "minco_objective_before": float(optimizer_result.objective_before),
            "minco_objective_after": float(optimizer_result.objective_after),
            "minco_optimizer_success": 1.0 if optimizer_result.success else 0.0,
            "minco_optimizer_iterations": float(optimizer_result.iterations),
        })

    return FeasibleReference(
        name=name,
        t=t,
        z=z,
        z_dot=z_dot,
        z_ddot=z_ddot,
        nu=nu,
        nu_dot=nu_dot,
        tau=tau,
        thrust=thrust,
        diagnostics=diagnostics,
    )


def _sample_times(duration: float, dt: float) -> np.ndarray:
    duration = float(duration)
    if duration < 0.0:
        raise ValueError("duration must be nonnegative.")
    if duration == 0.0:
        return np.array([0.0], dtype=float)

    steps = int(np.floor(duration / dt)) + 1
    times = np.linspace(0.0, dt * (steps - 1), steps, dtype=float)
    if times[-1] < duration - 1e-12:
        times = np.concatenate((times, np.array([duration], dtype=float)))
    else:
        times[-1] = duration
    return times
