from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Tuple

import numpy as np

from .minco_s3nu import MINCO_S3NU
from .usv_flatness import UsvModelParams
from .usv_flatness import flat_to_body_acceleration
from .usv_flatness import flat_to_body_velocity
from .usv_flatness import generalized_force_to_thrust
from .usv_flatness import theta_tau_nominal


@dataclass
class MincoUsvPenaltyConfig:
    params: UsvModelParams
    tau_v_bar: float = 250.0
    velocity_bounds: Tuple[float, float, float] = (2.0, 0.5, 0.6)
    acceleration_bounds: Tuple[float, float, float] = (0.9, 0.5, 0.8)
    lambda_tau_v: float = 1.0
    lambda_velocity: float = 1.0
    lambda_acceleration: float = 1.0
    lambda_actuator: float = 1.0
    quadrature_order: int = 8
    penalty_mu: float = 20.0


@dataclass
class MincoUsvPenaltyEvaluation:
    total: float
    tau_v: float
    velocity: float
    acceleration: float
    actuator: float
    diagnostics: Dict[str, float]


def smooth_positive_penalty(g, mu: float) -> np.ndarray:
    """Squared smooth positive-part penalty phi_mu(g)."""
    g = np.clip(np.asarray(g, dtype=float), -1e6, 1e6)
    mu = max(float(mu), 1e-6)
    softplus = _softplus(mu * g)
    return (softplus / mu) ** 2


def smooth_positive_penalty_derivative(g, mu: float) -> np.ndarray:
    """Derivative of squared smooth positive-part penalty with respect to g."""
    g_array = np.asarray(g, dtype=float)
    clipped = np.clip(g_array, -1e6, 1e6)
    mu = max(float(mu), 1e-6)
    x = mu * clipped
    derivative = 2.0 * (_softplus(x) / mu) * _sigmoid(x)
    derivative = np.asarray(derivative, dtype=float)
    derivative[(g_array < -1e6) | (g_array > 1e6)] = 0.0
    return derivative


def _softplus(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    softplus = np.empty_like(x, dtype=float)
    large = x > 40.0
    softplus[large] = x[large]
    x_safe = x[~large]
    softplus[~large] = (
        np.log1p(np.exp(-np.abs(x_safe))) + np.maximum(x_safe, 0.0)
    )
    return softplus


def _sigmoid(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x, dtype=float)
    positive = x >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


class MincoUsvDensePenalty:
    """
    Dense quadrature penalties for the USV MINCO constrained problem.

    This class converts the original continuous-time constraints into smooth
    penalty terms. It deliberately does not add effort, guide, obstacle, or
    corridor costs.
    """

    def __init__(self, config: MincoUsvPenaltyConfig):
        self.config = config
        if config.quadrature_order <= 0:
            raise ValueError("quadrature_order must be positive.")
        nodes, weights = np.polynomial.legendre.leggauss(config.quadrature_order)
        self.alphas = 0.5 * (nodes + 1.0)
        self.weights = 0.5 * weights

    def evaluate(self, trajectory: MINCO_S3NU) -> MincoUsvPenaltyEvaluation:
        components = {
            "tau_v": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "actuator": 0.0,
        }
        metric_rows = []

        for piece_index, T in enumerate(trajectory.ts):
            T = float(T)
            for alpha, weight in zip(self.alphas, self.weights):
                sample = trajectory.sample_piece(piece_index, alpha * T)
                node = self._evaluate_node(sample.z, sample.z_dot, sample.z_ddot)
                scale = T * float(weight)
                for key in components:
                    components[key] += scale * node[key]
                metric_rows.append(node["metrics"])

        diagnostics = self._diagnostics(metric_rows)
        total = (
            self.config.lambda_tau_v * components["tau_v"] +
            self.config.lambda_velocity * components["velocity"] +
            self.config.lambda_acceleration * components["acceleration"] +
            self.config.lambda_actuator * components["actuator"]
        )
        return MincoUsvPenaltyEvaluation(
            total=float(total),
            tau_v=float(components["tau_v"]),
            velocity=float(components["velocity"]),
            acceleration=float(components["acceleration"]),
            actuator=float(components["actuator"]),
            diagnostics=diagnostics,
        )

    def finite_difference_gradients(
        self,
        trajectory: MINCO_S3NU,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct gradients of weighted penalty total with respect to coefficients
        and times, computed at fixed coefficients for the time partials.
        """
        base_coeffs = trajectory.coefficients()
        base_ts = trajectory.ts.copy()
        base_total = self.evaluate(trajectory).total
        coeff_grad = np.zeros_like(base_coeffs)
        time_grad = np.zeros_like(base_ts)

        for index in np.ndindex(base_coeffs.shape):
            perturbed = base_coeffs.copy()
            perturbed[index] += eps
            value = self._evaluate_coefficients(perturbed, base_ts)
            coeff_grad[index] = (value - base_total) / eps

        for piece_index in range(len(base_ts)):
            perturbed_ts = base_ts.copy()
            perturbed_ts[piece_index] += eps
            value = self._evaluate_coefficients(base_coeffs, perturbed_ts)
            time_grad[piece_index] = (value - base_total) / eps

        return coeff_grad, time_grad

    def analytic_gradients(
        self,
        trajectory: MINCO_S3NU,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct gradients of weighted penalty total with respect to coefficients
        and segment times at fixed coefficients.

        This follows the GCOPTER-style dense quadrature chain rule. For a node
        s = alpha * T, the coefficient gradient uses dz/dC, dz_dot/dC, and
        dz_ddot/dC from the polynomial bases. The direct time partial includes
        both d(T*w*f)/dT = w*f and the local-time shift term
        T*w*alpha*df/ds.
        """
        coeffs = trajectory.coefficients()
        time_grad = np.zeros_like(trajectory.ts)
        coeff_grad = np.zeros_like(coeffs)

        for piece_index, T in enumerate(np.asarray(trajectory.ts, dtype=float)):
            T = float(T)
            coeff = coeffs[piece_index]
            for alpha, weight in zip(self.alphas, self.weights):
                local_time = float(alpha) * T
                b0 = MINCO_S3NU.basis(local_time, 0)
                b1 = MINCO_S3NU.basis(local_time, 1)
                b2 = MINCO_S3NU.basis(local_time, 2)
                b3 = MINCO_S3NU.basis(local_time, 3)
                z = b0.reshape(1, -1) @ coeff
                z_dot = b1.reshape(1, -1) @ coeff
                z_ddot = b2.reshape(1, -1) @ coeff
                z_jerk = b3.reshape(1, -1) @ coeff
                node = self._evaluate_node(
                    z.reshape(-1),
                    z_dot.reshape(-1),
                    z_ddot.reshape(-1),
                )
                scale = T * float(weight)
                coeff_grad[piece_index] += scale * (
                    np.outer(b0, node["grad_z"]) +
                    np.outer(b1, node["grad_z_dot"]) +
                    np.outer(b2, node["grad_z_ddot"])
                )
                df_ds = (
                    float(node["grad_z"] @ z_dot.reshape(-1)) +
                    float(node["grad_z_dot"] @ z_ddot.reshape(-1)) +
                    float(node["grad_z_ddot"] @ z_jerk.reshape(-1))
                )
                time_grad[piece_index] += float(weight) * (
                    node["weighted_total"] + T * float(alpha) * df_ds
                )

        return coeff_grad, time_grad

    def _evaluate_coefficients(self, coeffs, ts) -> float:
        value = 0.0
        for piece_index, T in enumerate(np.asarray(ts, dtype=float)):
            T = float(T)
            coeff = coeffs[piece_index]
            for alpha, weight in zip(self.alphas, self.weights):
                s = alpha * T
                z = MINCO_S3NU.basis(s, 0).reshape(1, -1) @ coeff
                z_dot = MINCO_S3NU.basis(s, 1).reshape(1, -1) @ coeff
                z_ddot = MINCO_S3NU.basis(s, 2).reshape(1, -1) @ coeff
                node = self._evaluate_node(
                    z.reshape(-1),
                    z_dot.reshape(-1),
                    z_ddot.reshape(-1),
                )
                value += T * float(weight) * (
                    self.config.lambda_tau_v * node["tau_v"] +
                    self.config.lambda_velocity * node["velocity"] +
                    self.config.lambda_acceleration * node["acceleration"] +
                    self.config.lambda_actuator * node["actuator"]
                )
        return float(value)

    def _evaluate_node(self, z, z_dot, z_ddot):
        config = self.config
        params = config.params
        nu = flat_to_body_velocity(z, z_dot)
        nu_dot = flat_to_body_acceleration(float(z[2]), z_dot, z_ddot)
        tau = theta_tau_nominal(z, z_dot, z_ddot, params)
        left, right = generalized_force_to_thrust(tau[0], tau[2], params)

        tau_v_g = np.array([
            tau[1] - config.tau_v_bar,
            -tau[1] - config.tau_v_bar,
        ])
        velocity_g = np.abs(nu) - np.asarray(config.velocity_bounds, dtype=float)
        acceleration_g = (
            np.abs(nu_dot) - np.asarray(config.acceleration_bounds, dtype=float)
        )
        actuator_g = np.array([
            params.min_thrust - left,
            left - params.max_thrust,
            params.min_thrust - right,
            right - params.max_thrust,
        ], dtype=float)

        tau_v_penalty = float(np.sum(
            smooth_positive_penalty(tau_v_g, config.penalty_mu)
        ))
        velocity_penalty = float(np.sum(
            smooth_positive_penalty(velocity_g, config.penalty_mu)
        ))
        acceleration_penalty = float(np.sum(
            smooth_positive_penalty(acceleration_g, config.penalty_mu)
        ))
        actuator_penalty = float(np.sum(
            smooth_positive_penalty(actuator_g, config.penalty_mu)
        ))
        grad_z, grad_z_dot, grad_z_ddot = self._node_gradients(
            z,
            z_dot,
            z_ddot,
            nu,
            nu_dot,
            tau,
            left,
            right,
            tau_v_g,
            velocity_g,
            acceleration_g,
            actuator_g,
        )
        weighted_total = (
            config.lambda_tau_v * tau_v_penalty +
            config.lambda_velocity * velocity_penalty +
            config.lambda_acceleration * acceleration_penalty +
            config.lambda_actuator * actuator_penalty
        )

        return {
            "tau_v": tau_v_penalty,
            "velocity": velocity_penalty,
            "acceleration": acceleration_penalty,
            "actuator": actuator_penalty,
            "weighted_total": float(weighted_total),
            "grad_z": grad_z,
            "grad_z_dot": grad_z_dot,
            "grad_z_ddot": grad_z_ddot,
            "metrics": {
                "tau_v": float(tau[1]),
                "nu": np.asarray(nu, dtype=float),
                "nu_dot": np.asarray(nu_dot, dtype=float),
                "left": float(left),
                "right": float(right),
                "tau": np.asarray(tau, dtype=float),
                "tau_v_violation": float(max(np.max(tau_v_g), 0.0)),
                "velocity_violation": float(max(np.max(velocity_g), 0.0)),
                "acceleration_violation": float(max(np.max(acceleration_g), 0.0)),
                "actuator_violation": float(max(np.max(actuator_g), 0.0)),
            },
        }

    def _node_gradients(
        self,
        z,
        z_dot,
        z_ddot,
        nu,
        nu_dot,
        tau,
        left,
        right,
        tau_v_g,
        velocity_g,
        acceleration_g,
        actuator_g,
    ):
        config = self.config
        mu = config.penalty_mu
        (
            jac_nu_z,
            jac_nu_z_dot,
            jac_nu_dot_z,
            jac_nu_dot_z_dot,
            jac_nu_dot_z_ddot,
            jac_tau_z,
            jac_tau_z_dot,
            jac_tau_z_ddot,
        ) = self._node_jacobians(z, z_dot, z_ddot, nu, nu_dot, tau)

        grad_z = np.zeros(3, dtype=float)
        grad_z_dot = np.zeros(3, dtype=float)
        grad_z_ddot = np.zeros(3, dtype=float)

        tau_v_dphi = smooth_positive_penalty_derivative(tau_v_g, mu)
        grad_tau = np.zeros(3, dtype=float)
        grad_tau[1] = config.lambda_tau_v * (
            tau_v_dphi[0] - tau_v_dphi[1]
        )
        grad_z += jac_tau_z.T @ grad_tau
        grad_z_dot += jac_tau_z_dot.T @ grad_tau
        grad_z_ddot += jac_tau_z_ddot.T @ grad_tau

        velocity_dphi = smooth_positive_penalty_derivative(velocity_g, mu)
        grad_nu = (
            config.lambda_velocity *
            velocity_dphi *
            self._abs_gradient(nu)
        )
        grad_z += jac_nu_z.T @ grad_nu
        grad_z_dot += jac_nu_z_dot.T @ grad_nu

        acceleration_dphi = smooth_positive_penalty_derivative(
            acceleration_g, mu
        )
        grad_nu_dot = (
            config.lambda_acceleration *
            acceleration_dphi *
            self._abs_gradient(nu_dot)
        )
        grad_z += jac_nu_dot_z.T @ grad_nu_dot
        grad_z_dot += jac_nu_dot_z_dot.T @ grad_nu_dot
        grad_z_ddot += jac_nu_dot_z_ddot.T @ grad_nu_dot

        actuator_dphi = smooth_positive_penalty_derivative(actuator_g, mu)
        grad_left = config.lambda_actuator * (
            -actuator_dphi[0] + actuator_dphi[1]
        )
        grad_right = config.lambda_actuator * (
            -actuator_dphi[2] + actuator_dphi[3]
        )
        l = config.params.thruster_half_spacing
        grad_tau = np.array([
            0.5 * (grad_left + grad_right),
            0.0,
            0.5 * (grad_right - grad_left) / l,
        ], dtype=float)
        grad_z += jac_tau_z.T @ grad_tau
        grad_z_dot += jac_tau_z_dot.T @ grad_tau
        grad_z_ddot += jac_tau_z_ddot.T @ grad_tau

        return grad_z, grad_z_dot, grad_z_ddot

    def _node_jacobians(self, z, z_dot, z_ddot, nu, nu_dot, tau):
        del tau
        del nu_dot
        params = self.config.params
        z = np.asarray(z, dtype=float).reshape(3)
        z_dot = np.asarray(z_dot, dtype=float).reshape(3)
        z_ddot = np.asarray(z_ddot, dtype=float).reshape(3)
        nu = np.asarray(nu, dtype=float).reshape(3)
        psi = float(z[2])
        c = float(np.cos(psi))
        s = float(np.sin(psi))
        x_ddot = float(z_ddot[0])
        y_ddot = float(z_ddot[1])
        u, v, r = [float(value) for value in nu]
        a_u = c * x_ddot + s * y_ddot
        a_v = -s * x_ddot + c * y_ddot

        jac_nu_z = np.zeros((3, 3), dtype=float)
        jac_nu_z[0, 2] = v
        jac_nu_z[1, 2] = -u
        jac_nu_z_dot = np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        jac_nu_dot_z = np.zeros((3, 3), dtype=float)
        jac_nu_dot_z[0, 2] = a_v - r * u
        jac_nu_dot_z[1, 2] = -a_u - r * v
        jac_nu_dot_z_dot = np.zeros((3, 3), dtype=float)
        jac_nu_dot_z_dot[0] = r * np.array([-s, c, 0.0]) + np.array([0.0, 0.0, v])
        jac_nu_dot_z_dot[1] = -r * np.array([c, s, 0.0]) - np.array([0.0, 0.0, u])
        jac_nu_dot_z_ddot = np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        du_gain = params.du + 2.0 * params.duu * abs(u)
        dv_gain = params.dv + 2.0 * params.dvv * abs(v)
        dr_gain = params.dr + 2.0 * params.drr * abs(r)
        jac_tau_z = np.zeros((3, 3), dtype=float)
        jac_tau_z[0, 2] = params.mass * a_v + du_gain * v
        jac_tau_z[1, 2] = -params.mass * a_u - dv_gain * u
        jac_tau_z_dot = np.zeros((3, 3), dtype=float)
        jac_tau_z_dot[0] = du_gain * np.array([c, s, 0.0])
        jac_tau_z_dot[1] = dv_gain * np.array([-s, c, 0.0])
        jac_tau_z_dot[2] = dr_gain * np.array([0.0, 0.0, 1.0])
        jac_tau_z_ddot = np.array([
            [params.mass * c, params.mass * s, 0.0],
            [-params.mass * s, params.mass * c, 0.0],
            [0.0, 0.0, params.iz],
        ], dtype=float)

        return (
            jac_nu_z,
            jac_nu_z_dot,
            jac_nu_dot_z,
            jac_nu_dot_z_dot,
            jac_nu_dot_z_ddot,
            jac_tau_z,
            jac_tau_z_dot,
            jac_tau_z_ddot,
        )

    @staticmethod
    def _abs_gradient(value):
        value = np.asarray(value, dtype=float)
        grad = np.sign(value)
        grad[np.abs(value) < 1e-12] = 0.0
        return grad

    @staticmethod
    def _diagnostics(metric_rows) -> Dict[str, float]:
        if not metric_rows:
            return {
                "max_abs_tau_v": 0.0,
                "rms_tau_v": 0.0,
                "velocity_violation": 0.0,
                "acceleration_violation": 0.0,
                "actuator_bound_violation": 0.0,
            }

        tau_v = np.asarray([row["tau_v"] for row in metric_rows], dtype=float)
        left = np.asarray([row["left"] for row in metric_rows], dtype=float)
        right = np.asarray([row["right"] for row in metric_rows], dtype=float)
        tau = np.asarray([row["tau"] for row in metric_rows], dtype=float)
        return {
            "max_abs_tau_v": float(np.max(np.abs(tau_v))),
            "rms_tau_v": float(np.sqrt(np.mean(tau_v**2))),
            "velocity_violation": float(max(
                row["velocity_violation"] for row in metric_rows
            )),
            "acceleration_violation": float(max(
                row["acceleration_violation"] for row in metric_rows
            )),
            "actuator_bound_violation": float(max(
                row["actuator_violation"] for row in metric_rows
            )),
            "max_abs_tau_u": float(np.max(np.abs(tau[:, 0]))),
            "max_abs_tau_r": float(np.max(np.abs(tau[:, 2]))),
            "max_left_thrust": float(np.max(left)),
            "min_left_thrust": float(np.min(left)),
            "max_right_thrust": float(np.max(right)),
            "min_right_thrust": float(np.min(right)),
        }
