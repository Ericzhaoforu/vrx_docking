from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .usv_flatness import UsvModelParams
from .usv_flatness import flat_to_body_acceleration
from .usv_flatness import flat_to_body_velocity
from .usv_flatness import generalized_force_to_thrust
from .usv_flatness import theta_tau_nominal


@dataclass
class MincoCoefficientReference:
    """Exact MINCO coefficient reference published by the planner node."""

    name: str
    trajectory_id: int
    start_time: float
    segment_times: np.ndarray
    coefficients: np.ndarray
    params: UsvModelParams
    diagnostics: Dict[str, float]

    @classmethod
    def from_msg(cls, msg, params: UsvModelParams):
        dimension = int(msg.dimension)
        order = int(msg.polynomial_order)
        segment_count = int(msg.segment_count)
        if dimension != 3:
            raise ValueError("MINCO trajectory dimension must be 3.")
        if order < 2:
            raise ValueError("MINCO polynomial_order must be at least 2.")
        segment_times = np.asarray(msg.segment_times, dtype=float)
        if segment_times.shape != (segment_count,):
            raise ValueError("segment_times length does not match segment_count.")
        coeffs = np.asarray(msg.coefficients, dtype=float)
        expected = segment_count * dimension * (order + 1)
        if coeffs.size != expected:
            raise ValueError(
                "coefficients length does not match "
                "segment_count * dimension * (polynomial_order + 1).")
        coeffs = coeffs.reshape(segment_count, dimension, order + 1)
        start_time = (
            float(msg.start_time.sec) +
            float(msg.start_time.nanosec) * 1e-9)
        diagnostics = {
            "trajectory_id": float(msg.trajectory_id),
            "minco_duration": float(np.sum(segment_times)),
            "minco_piece_count": float(segment_count),
            "max_abs_tau_v": float(msg.max_abs_tau_v),
            "rms_tau_v": float(msg.rms_tau_v),
            "velocity_violation": float(msg.velocity_violation),
            "acceleration_violation": float(msg.acceleration_violation),
            "actuator_bound_violation": float(msg.actuator_bound_violation),
            "minco_cpp_planner_solve_time_ms": float(msg.solve_time_ms),
            "minco_coefficients_reference": 1.0,
        }
        return cls(
            name=f"minco_topic_{int(msg.trajectory_id)}",
            trajectory_id=int(msg.trajectory_id),
            start_time=start_time,
            segment_times=segment_times,
            coefficients=coeffs,
            params=params,
            diagnostics=diagnostics,
        )

    @property
    def duration(self) -> float:
        return float(np.sum(self.segment_times))

    @property
    def t(self) -> np.ndarray:
        return np.concatenate((
            np.asarray([0.0], dtype=float),
            np.cumsum(self.segment_times)))

    def sample(self, query_t: float):
        z, z_dot, z_ddot = self.sample_flat(query_t)
        nu = flat_to_body_velocity(z, z_dot)
        nu_dot = flat_to_body_acceleration(float(z[2]), z_dot, z_ddot)
        tau = theta_tau_nominal(z, z_dot, z_ddot, self.params)
        thrust = np.asarray(
            generalized_force_to_thrust(tau[0], tau[2], self.params),
            dtype=float)
        return {
            "z": z,
            "z_dot": z_dot,
            "z_ddot": z_ddot,
            "nu": nu,
            "nu_dot": nu_dot,
            "tau": tau,
            "thrust": thrust,
        }

    def sample_flat(self, query_t: float):
        piece_index, local_time = self.local_time(query_t)
        coeff = self.coefficients[piece_index]
        z = self.evaluate_piece(coeff, local_time, 0)
        z_dot = self.evaluate_piece(coeff, local_time, 1)
        z_ddot = self.evaluate_piece(coeff, local_time, 2)
        return z, z_dot, z_ddot

    def horizon_at_offsets(self, start_t: float, offsets):
        samples = [self.sample(float(start_t) + float(offset))
                   for offset in offsets]
        return {
            key: np.asarray([sample[key] for sample in samples], dtype=float)
            for key in samples[0].keys()
        }

    def horizon(self, start_t: float, dt: float, count: int):
        offsets = [float(dt) * float(index) for index in range(count)]
        return self.horizon_at_offsets(start_t, offsets)

    def local_time(self, query_t: float):
        query_t = min(max(float(query_t), 0.0), self.duration)
        cumulative = np.cumsum(self.segment_times)
        piece_index = int(np.searchsorted(cumulative, query_t, side="right"))
        piece_index = min(piece_index, len(self.segment_times) - 1)
        previous = 0.0 if piece_index == 0 else float(cumulative[piece_index - 1])
        local_time = min(
            max(query_t - previous, 0.0),
            float(self.segment_times[piece_index]))
        return piece_index, local_time

    @staticmethod
    def evaluate_piece(coeff_dim_power, local_time: float, derivative: int):
        order = coeff_dim_power.shape[1] - 1
        value = np.zeros(3, dtype=float)
        for power in range(derivative, order + 1):
            scale = 1.0
            for factor in range(derivative):
                scale *= float(power - factor)
            value += (
                scale *
                (float(local_time) ** float(power - derivative)) *
                coeff_dim_power[:, power])
        return value
