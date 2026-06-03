import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle_array(values):
    return np.arctan2(np.sin(values), np.cos(values))


def r3(psi: float) -> np.ndarray:
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def flat_to_body_velocity(z, z_dot) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z_dot = np.asarray(z_dot, dtype=float)
    return r3(float(z[2])).T @ z_dot


def flat_to_body_velocity_at_yaw(psi: float, z_dot) -> np.ndarray:
    z_dot = np.asarray(z_dot, dtype=float)
    return r3(psi).T @ z_dot


def body_to_flat_velocity(psi: float, nu) -> np.ndarray:
    return r3(psi) @ np.asarray(nu, dtype=float)


def body_to_flat_acceleration(psi: float, nu, nu_dot) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    nu_dot = np.asarray(nu_dot, dtype=float)
    c = math.cos(psi)
    s = math.sin(psi)
    u, v, r = nu
    u_dot, v_dot, r_dot = nu_dot
    x_ddot = c * u_dot - s * v_dot - r * (s * u + c * v)
    y_ddot = s * u_dot + c * v_dot + r * (c * u - s * v)
    return np.array([x_ddot, y_ddot, r_dot], dtype=float)


def flat_to_body_acceleration(psi: float, z_dot, z_ddot) -> np.ndarray:
    z_dot = np.asarray(z_dot, dtype=float)
    z_ddot = np.asarray(z_ddot, dtype=float)
    nu = flat_to_body_velocity_at_yaw(psi, z_dot)
    acc_body = r3(psi).T @ z_ddot
    u, v, r = nu
    return np.array([
        acc_body[0] + r * v,
        acc_body[1] - r * u,
        acc_body[2],
    ], dtype=float)


@dataclass
class UsvModelParams:
    mass: float = 180.0
    iz: float = 446.0
    du: float = 100.0
    duu: float = 150.0
    dv: float = 100.0
    dvv: float = 100.0
    dr: float = 800.0
    drr: float = 800.0
    thruster_half_spacing: float = 1.027135
    min_thrust: float = -500.0
    max_thrust: float = 500.0
    thrust_rate_limit: float = 1000.0

    @property
    def mass_matrix(self) -> np.ndarray:
        return np.diag([self.mass, self.mass, self.iz])


def nominal_coriolis_times_velocity(nu, params: UsvModelParams) -> np.ndarray:
    u, v, r = np.asarray(nu, dtype=float)
    m = params.mass
    return np.array([
        -m * v * r,
        m * u * r,
        0.0,
    ], dtype=float)


def nominal_damping(nu, params: UsvModelParams) -> np.ndarray:
    u, v, r = np.asarray(nu, dtype=float)
    return np.array([
        params.du * u + params.duu * abs(u) * u,
        params.dv * v + params.dvv * abs(v) * v,
        params.dr * r + params.drr * abs(r) * r,
    ], dtype=float)


def theta_tau_nominal(z, z_dot, z_ddot,
                      params: UsvModelParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    psi = float(z[2])
    nu = flat_to_body_velocity_at_yaw(psi, z_dot)
    nu_dot = flat_to_body_acceleration(psi, z_dot, z_ddot)
    return (
        params.mass_matrix @ nu_dot +
        nominal_coriolis_times_velocity(nu, params) +
        nominal_damping(nu, params)
    )


def generalized_force_to_thrust(tau_u: float, tau_r: float,
                                params: UsvModelParams):
    l = params.thruster_half_spacing
    left = 0.5 * (float(tau_u) - float(tau_r) / l)
    right = 0.5 * (float(tau_u) + float(tau_r) / l)
    return left, right


def thrust_to_generalized_force(left: float, right: float,
                                params: UsvModelParams):
    tau_u = float(left) + float(right)
    tau_r = params.thruster_half_spacing * (float(right) - float(left))
    return tau_u, tau_r


def allocation_feasible(tau_u: float, tau_r: float,
                        params: UsvModelParams) -> bool:
    left, right = generalized_force_to_thrust(tau_u, tau_r, params)
    return (
        params.min_thrust <= left <= params.max_thrust and
        params.min_thrust <= right <= params.max_thrust
    )


def clip_allocated_thrust(tau_u: float, tau_r: float,
                          params: UsvModelParams):
    left, right = generalized_force_to_thrust(tau_u, tau_r, params)
    left = float(np.clip(left, params.min_thrust, params.max_thrust))
    right = float(np.clip(right, params.min_thrust, params.max_thrust))
    tau_u_c, tau_r_c = thrust_to_generalized_force(left, right, params)
    return left, right, tau_u_c, tau_r_c
