from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:  # pragma: no cover - runtime dependency guard
    scipy_minimize = None

try:
    import robotx_minco_cpp as _minco_cpp
except Exception:  # pragma: no cover - optional compiled acceleration
    try:
        from . import _minco_cpp
    except Exception:
        _minco_cpp = None

from .minco_s3nu import MINCO_S3NU
from .minco_usv_penalties import MincoUsvDensePenalty
from .minco_usv_penalties import MincoUsvPenaltyConfig


@dataclass
class MincoUsvOptimizerConfig:
    piece_count: int = 3
    fixed_total_time: Optional[float] = None
    reference_speed: float = 0.8
    time_dilation: float = 1.3
    min_segment_time: float = 0.2
    smooth_weight: float = 1.0
    time_weight: float = 0.0
    max_iterations: int = 80
    gradient_tolerance: float = 1e-6
    function_tolerance: float = 1e-9
    penalty_fd_eps: float = 1e-6
    spatial_margin: float = 2.0
    yaw_margin: float = 1.0
    theta_bound: float = 6.0


@dataclass
class MincoUsvOptimizerResult:
    success: bool
    message: str
    objective_before: float
    objective_after: float
    iterations: int
    inPs: np.ndarray
    ts: np.ndarray
    theta: np.ndarray
    trajectory: MINCO_S3NU
    diagnostics: dict


class MincoUsvOptimizer:
    """
    Outer MINCO optimizer over intermediate points and time variables.

    Polynomial coefficients are generated only by MINCO_S3NU.set_parameters().
    They are never part of the sparse outer optimization vector.
    """

    def __init__(
        self,
        penalty_config: MincoUsvPenaltyConfig,
        config: Optional[MincoUsvOptimizerConfig] = None,
    ):
        self.penalty = MincoUsvDensePenalty(penalty_config)
        self.config = config or MincoUsvOptimizerConfig()
        if self.config.piece_count <= 0:
            raise ValueError("piece_count must be positive.")
        if self.config.reference_speed <= 0.0:
            raise ValueError("reference_speed must be positive.")
        if self.config.min_segment_time <= 0.0:
            raise ValueError("min_segment_time must be positive.")

    def optimize(
        self,
        head_pva,
        tail_pva,
        initial_inPs=None,
        initial_ts=None,
    ) -> MincoUsvOptimizerResult:
        head_pva, tail_pva = self._validate_pva(head_pva, tail_pva)
        if initial_inPs is None or initial_ts is None:
            inPs0, ts0 = self.initial_guess(head_pva, tail_pva)
        else:
            inPs0 = np.asarray(initial_inPs, dtype=float)
            ts0 = np.asarray(initial_ts, dtype=float)
            expected_inPs_shape = (int(self.config.piece_count) - 1, head_pva.shape[1])
            if inPs0.shape != expected_inPs_shape:
                raise ValueError(
                    "initial_inPs must have shape "
                    f"{expected_inPs_shape}, got {inPs0.shape}."
                )
            if ts0.shape != (int(self.config.piece_count),):
                raise ValueError(
                    "initial_ts must have shape "
                    f"({int(self.config.piece_count)},), got {ts0.shape}."
                )
            if np.any(ts0 <= 0.0):
                raise ValueError("initial_ts entries must be positive.")
        theta0 = self.times_to_theta(ts0)
        x0 = self.pack_variables(inPs0, theta0)

        objective_before, _grad_before, _traj_before, _eval_before = (
            self.objective_and_gradient(x0, head_pva, tail_pva)
        )

        cpp_context = self._cpp_context()
        use_cpp_optimizer = (
            _minco_cpp is not None and
            hasattr(_minco_cpp, "minco_optimize_lbfgs") and
            os.environ.get("ROBOTX_MINCO_OPTIMIZER_CPP", "1") != "0" and
            os.environ.get("ROBOTX_MINCO_USE_CPP", "1") != "0")
        if use_cpp_optimizer:
            result = _minco_cpp.minco_optimize_lbfgs(
                x0,
                head_pva,
                tail_pva,
                cpp_context["params"],
                cpp_context["optimizer"],
                cpp_context["penalty"],
            )
            x_opt = np.asarray(result["variables"], dtype=float)
            objective_after, grad_after, trajectory, penalty_eval = (
                self.objective_and_gradient(x_opt, head_pva, tail_pva)
            )
            inPs, theta = self.unpack_variables(x_opt)
            ts = self.theta_to_times(theta)
            diagnostics = dict(penalty_eval.diagnostics)
            result_diagnostics = dict(result.get("diagnostics", {}))
            diagnostics.update(result_diagnostics)
            diagnostics.update({
                "optimizer_fun": float(result.get(
                    "optimizer_fun", objective_after)),
                "optimizer_grad_inf_norm": float(np.max(np.abs(grad_after))),
                "optimizer_eval_count": float(result.get("eval_count", 0)),
                "optimizer_nfev": float(result.get("eval_count", 0)),
                "optimizer_njev": float(result.get("eval_count", 0)),
                "optimizer_iterations": float(result.get("iterations", 0)),
                "optimizer_status_code": float(result.get("status_code", 0)),
            })
            return MincoUsvOptimizerResult(
                success=bool(result["success"]),
                message=str(result["message"]),
                objective_before=float(result.get(
                    "objective_before", objective_before)),
                objective_after=float(objective_after),
                iterations=int(result.get("iterations", 0)),
                inPs=inPs,
                ts=ts,
                theta=theta,
                trajectory=trajectory,
                diagnostics=diagnostics,
            )

        if scipy_minimize is None:
            raise RuntimeError(
                "scipy.optimize is required when the C++ LBFGS optimizer is unavailable."
            )

        use_cpp = (
            _minco_cpp is not None and
            os.environ.get("ROBOTX_MINCO_USE_CPP", "1") != "0")
        evaluation_count = 0

        def objective_with_gradient(x):
            nonlocal evaluation_count
            evaluation_count += 1
            if use_cpp:
                result = _minco_cpp.minco_objective_gradient(
                    np.asarray(x, dtype=float),
                    head_pva,
                    tail_pva,
                    cpp_context["params"],
                    cpp_context["optimizer"],
                    cpp_context["penalty"],
                )
                return float(result["objective"]), np.asarray(
                    result["gradient"], dtype=float)
            objective, gradient, _trajectory, _penalty_eval = (
                self.objective_and_gradient(x, head_pva, tail_pva)
            )
            return objective, gradient

        result = scipy_minimize(
            objective_with_gradient,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=self.variable_bounds(head_pva, tail_pva),
            options={
                "maxiter": int(self.config.max_iterations),
                "gtol": float(self.config.gradient_tolerance),
                "ftol": float(self.config.function_tolerance),
                "maxls": 30,
            },
        )

        objective_after, _grad_after, trajectory, penalty_eval = (
            self.objective_and_gradient(result.x, head_pva, tail_pva)
        )
        inPs, theta = self.unpack_variables(result.x)
        ts = self.theta_to_times(theta)
        diagnostics = dict(penalty_eval.diagnostics)
        diagnostics.update({
            "optimizer_fun": float(result.fun),
            "optimizer_grad_inf_norm": float(np.max(np.abs(result.jac))),
            "optimizer_eval_count": float(evaluation_count),
            "optimizer_nfev": float(getattr(result, "nfev", 0)),
            "optimizer_njev": float(getattr(result, "njev", 0)),
        })
        return MincoUsvOptimizerResult(
            success=bool(result.success),
            message=str(result.message),
            objective_before=float(objective_before),
            objective_after=float(objective_after),
            iterations=int(result.nit),
            inPs=inPs,
            ts=ts,
            theta=theta,
            trajectory=trajectory,
            diagnostics=diagnostics,
        )

    def objective_and_gradient(self, variables, head_pva, tail_pva):
        head_pva, tail_pva = self._validate_pva(head_pva, tail_pva)
        inPs, theta = self.unpack_variables(variables)
        ts = self.theta_to_times(theta)

        trajectory = MINCO_S3NU(dim=head_pva.shape[1])
        trajectory.set_conditions(head_pva, tail_pva)
        trajectory.set_parameters(inPs, ts)

        energy = trajectory.get_energy()
        energy_coeff_grad, energy_time_grad = trajectory.energy_gradients()
        penalty_eval = self.penalty.evaluate(trajectory)
        penalty_coeff_grad, penalty_time_grad = self.penalty.analytic_gradients(
            trajectory,
        )

        coeff_grad = (
            self.config.smooth_weight * energy_coeff_grad +
            penalty_coeff_grad
        )
        time_grad = (
            self.config.smooth_weight * energy_time_grad +
            penalty_time_grad
        )
        if self.config.fixed_total_time is None:
            time_grad = time_grad + self.config.time_weight

        propagated = trajectory.propagate_gradients(coeff_grad, time_grad)
        grad_inPs = propagated["inPs"]
        grad_ts = propagated["ts"]
        grad_theta = self.times_gradient_to_theta_gradient(grad_ts, ts)

        objective = self.config.smooth_weight * energy + penalty_eval.total
        if self.config.fixed_total_time is None:
            objective += self.config.time_weight * float(np.sum(ts))

        gradient = self.pack_variables(grad_inPs, grad_theta)
        return float(objective), gradient, trajectory, penalty_eval

    def objective_and_gradient_cpp(self, variables, head_pva, tail_pva):
        """C++ equivalent of objective_and_gradient for tests and profiling."""
        if _minco_cpp is None:
            raise RuntimeError("The compiled _minco_cpp extension is unavailable.")
        context = self._cpp_context()
        result = _minco_cpp.minco_objective_gradient(
            np.asarray(variables, dtype=float),
            np.asarray(head_pva, dtype=float),
            np.asarray(tail_pva, dtype=float),
            context["params"],
            context["optimizer"],
            context["penalty"],
        )
        return float(result["objective"]), np.asarray(result["gradient"], dtype=float)

    def _cpp_context(self):
        params = self.penalty.config.params
        return {
            "params": {
                "mass": float(params.mass),
                "iz": float(params.iz),
                "du": float(params.du),
                "duu": float(params.duu),
                "dv": float(params.dv),
                "dvv": float(params.dvv),
                "dr": float(params.dr),
                "drr": float(params.drr),
                "thruster_half_spacing": float(params.thruster_half_spacing),
                "min_thrust": float(params.min_thrust),
                "max_thrust": float(params.max_thrust),
            },
            "optimizer": {
                "piece_count": int(self.config.piece_count),
                "fixed_total_time_enabled": self.config.fixed_total_time is not None,
                "fixed_total_time": (
                    0.0 if self.config.fixed_total_time is None
                    else float(self.config.fixed_total_time)),
                "reference_speed": float(self.config.reference_speed),
                "time_dilation": float(self.config.time_dilation),
                "min_segment_time": float(self.config.min_segment_time),
                "smooth_weight": float(self.config.smooth_weight),
                "time_weight": float(self.config.time_weight),
                "max_iterations": int(self.config.max_iterations),
                "gradient_tolerance": float(self.config.gradient_tolerance),
                "function_tolerance": float(self.config.function_tolerance),
                "spatial_margin": float(self.config.spatial_margin),
                "yaw_margin": float(self.config.yaw_margin),
                "theta_bound": float(self.config.theta_bound),
            },
            "penalty": {
                "tau_v_bar": float(self.penalty.config.tau_v_bar),
                "velocity_bounds": list(self.penalty.config.velocity_bounds),
                "acceleration_bounds": list(
                    self.penalty.config.acceleration_bounds),
                "lambda_tau_v": float(self.penalty.config.lambda_tau_v),
                "lambda_velocity": float(self.penalty.config.lambda_velocity),
                "lambda_acceleration": float(
                    self.penalty.config.lambda_acceleration),
                "lambda_actuator": float(self.penalty.config.lambda_actuator),
                "quadrature_order": int(self.penalty.config.quadrature_order),
                "penalty_mu": float(self.penalty.config.penalty_mu),
            },
        }

    def initial_guess(self, head_pva, tail_pva):
        head_pva, tail_pva = self._validate_pva(head_pva, tail_pva)
        piece_count = int(self.config.piece_count)
        dim = head_pva.shape[1]
        if piece_count == 1:
            inPs = np.empty((0, dim), dtype=float)
        else:
            start = head_pva[0].copy()
            goal = tail_pva[0].copy()
            goal[2] = start[2] + self._shortest_yaw_delta(start[2], goal[2])
            points = []
            for index in range(1, piece_count):
                alpha = index / piece_count
                points.append((1.0 - alpha) * start + alpha * goal)
            inPs = np.asarray(points, dtype=float)

        total_time = self.config.fixed_total_time
        distance = float(np.linalg.norm(tail_pva[0, :2] - head_pva[0, :2]))
        if total_time is None:
            total_time = max(
                self.config.time_dilation * distance / self.config.reference_speed,
                piece_count * self.config.min_segment_time,
            )
        else:
            total_time = max(
                float(total_time),
                piece_count * self.config.min_segment_time,
            )

        if piece_count == 1:
            return inPs, np.asarray([total_time], dtype=float)

        waypoints_xy = np.vstack((head_pva[0, :2], inPs[:, :2], tail_pva[0, :2]))
        lengths = np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1)
        weights = lengths + 1e-6
        ts = total_time * weights / float(np.sum(weights))
        ts = np.maximum(ts, self.config.min_segment_time)
        ts = total_time * ts / float(np.sum(ts))
        return inPs, ts

    def pack_variables(self, inPs, theta) -> np.ndarray:
        return np.concatenate([
            np.asarray(inPs, dtype=float).reshape(-1),
            np.asarray(theta, dtype=float).reshape(-1),
        ])

    def unpack_variables(self, variables):
        variables = np.asarray(variables, dtype=float).reshape(-1)
        dim = 3
        piece_count = int(self.config.piece_count)
        point_size = (piece_count - 1) * dim
        inPs = variables[:point_size].reshape(piece_count - 1, dim)
        theta = variables[point_size:point_size + piece_count]
        if theta.size != piece_count:
            raise ValueError("Optimization vector has wrong size.")
        return inPs, theta

    def times_to_theta(self, ts) -> np.ndarray:
        ts = np.asarray(ts, dtype=float).reshape(-1)
        if np.any(ts <= 0.0):
            raise ValueError("Segment times must be positive.")
        if self.config.fixed_total_time is None:
            return np.log(np.maximum(ts - self.config.min_segment_time, 1e-9))
        return np.log(ts)

    def theta_to_times(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.config.piece_count:
            raise ValueError("theta must have one value per piece.")
        shifted = theta - float(np.max(theta))
        exp_theta = np.exp(shifted)
        if self.config.fixed_total_time is None:
            return self.config.min_segment_time + np.exp(
                np.clip(theta, -50.0, 50.0)
            )
        total_time = max(
            float(self.config.fixed_total_time),
            self.config.piece_count * self.config.min_segment_time,
        )
        remaining = total_time - self.config.piece_count * self.config.min_segment_time
        if remaining <= 1e-12:
            return np.full(
                self.config.piece_count,
                self.config.min_segment_time,
                dtype=float,
            )
        return (
            self.config.min_segment_time +
            remaining * exp_theta / float(np.sum(exp_theta))
        )

    def times_gradient_to_theta_gradient(self, grad_ts, ts) -> np.ndarray:
        grad_ts = np.asarray(grad_ts, dtype=float).reshape(-1)
        ts = np.asarray(ts, dtype=float).reshape(-1)
        if self.config.fixed_total_time is None:
            return grad_ts * (ts - self.config.min_segment_time)
        shifted_ts = ts - self.config.min_segment_time
        remaining = float(np.sum(shifted_ts))
        if remaining <= 1e-12:
            return np.zeros_like(ts)
        pi = shifted_ts / remaining
        weighted_mean = float(grad_ts @ pi)
        return shifted_ts * (grad_ts - weighted_mean)

    def variable_bounds(self, head_pva, tail_pva):
        dim = 3
        piece_count = int(self.config.piece_count)
        margin = max(float(self.config.spatial_margin), 0.0)
        yaw_margin = max(float(self.config.yaw_margin), 0.0)
        theta_bound = max(float(self.config.theta_bound), 1.0)
        start = np.asarray(head_pva[0], dtype=float)
        goal = np.asarray(tail_pva[0], dtype=float).copy()
        goal[2] = start[2] + self._shortest_yaw_delta(start[2], goal[2])

        lower = np.minimum(start, goal)
        upper = np.maximum(start, goal)
        lower[:2] -= margin
        upper[:2] += margin
        lower[2] -= yaw_margin
        upper[2] += yaw_margin

        bounds = []
        for _ in range(max(piece_count - 1, 0)):
            for axis in range(dim):
                bounds.append((float(lower[axis]), float(upper[axis])))
        bounds.extend([(-theta_bound, theta_bound)] * piece_count)
        return bounds

    @staticmethod
    def _shortest_yaw_delta(start: float, goal: float) -> float:
        return float(np.arctan2(np.sin(goal - start), np.cos(goal - start)))

    @staticmethod
    def _validate_pva(head_pva, tail_pva):
        head_pva = np.asarray(head_pva, dtype=float)
        tail_pva = np.asarray(tail_pva, dtype=float)
        if head_pva.shape != (3, 3):
            raise ValueError("head_pva must have shape (3,3).")
        if tail_pva.shape != (3, 3):
            raise ValueError("tail_pva must have shape (3,3).")
        return head_pva, tail_pva
