from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np

from .minco_feasible_reference import minco_to_feasible_reference
from .minco_s3nu import MINCO_S3NU
from .minco_usv_optimizer import MincoUsvOptimizer
from .minco_usv_optimizer import MincoUsvOptimizerConfig
from .minco_usv_penalties import MincoUsvPenaltyConfig
from .usv_flatness import UsvModelParams
from .usv_flatness import wrap_angle


@dataclass
class MincoOfflineCase:
    name: str
    head_pva: np.ndarray
    tail_pva: np.ndarray
    total_time: float
    piece_count: int = 3


def default_offline_cases() -> List[MincoOfflineCase]:
    head = np.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    return [
        MincoOfflineCase(
            name="straight",
            head_pva=head.copy(),
            tail_pva=np.array([
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=float),
            total_time=5.0,
            piece_count=3,
        ),
        MincoOfflineCase(
            name="gentle_arc",
            head_pva=head.copy(),
            tail_pva=np.array([
                [3.0, 1.0, 0.35],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=float),
            total_time=6.0,
            piece_count=3,
        ),
    ]


def run_offline_validation(
    cases: Iterable[MincoOfflineCase] | None = None,
    output_json: str | Path | None = None,
) -> Dict[str, Dict[str, float]]:
    params = UsvModelParams(min_thrust=-500.0, max_thrust=500.0)
    penalty_config = MincoUsvPenaltyConfig(
        params=params,
        tau_v_bar=250.0,
        velocity_bounds=(2.0, 0.8, 0.8),
        acceleration_bounds=(1.5, 1.0, 1.0),
        lambda_tau_v=1.0,
        lambda_velocity=1.0,
        lambda_acceleration=1.0,
        lambda_actuator=1.0,
        quadrature_order=6,
    )
    results = {}
    for case in cases or default_offline_cases():
        optimizer = MincoUsvOptimizer(
            penalty_config,
            MincoUsvOptimizerConfig(
                piece_count=case.piece_count,
                fixed_total_time=case.total_time,
                max_iterations=50,
            ),
        )
        result = optimizer.optimize(case.head_pva, case.tail_pva)
        reference = minco_to_feasible_reference(
            f"minco_{case.name}",
            result.trajectory,
            params,
            dt=0.1,
            penalty_config=penalty_config,
            optimizer_result=result,
        )
        residuals = minco_residuals(result.trajectory, case.head_pva, case.tail_pva)
        gradient_error = outer_gradient_check(
            optimizer,
            case.head_pva,
            case.tail_pva,
        )
        terminal_pose_error = np.asarray(reference.z[-1] - case.tail_pva[0], dtype=float)
        terminal_pose_error[2] = wrap_angle(float(terminal_pose_error[2]))
        metrics = {
            "optimizer_success": 1.0 if result.success else 0.0,
            "objective_before": result.objective_before,
            "objective_after": result.objective_after,
            "pva_residual": residuals["pva_residual"],
            "intermediate_residual": residuals["intermediate_residual"],
            "continuity_residual": residuals["continuity_residual"],
            "z_dot_continuity_residual": residuals["z_dot_continuity_residual"],
            "z_ddot_continuity_residual": residuals["z_ddot_continuity_residual"],
            "z_jerk_continuity_residual": residuals["z_jerk_continuity_residual"],
            "z_snap_continuity_residual": residuals["z_snap_continuity_residual"],
            "gradient_check_error": gradient_error,
            "max_abs_tau_v": reference.diagnostics["max_abs_tau_v"],
            "rms_tau_v": reference.diagnostics["rms_tau_v"],
            "velocity_violation": reference.diagnostics["velocity_violation"],
            "acceleration_violation": reference.diagnostics["acceleration_violation"],
            "actuator_bound_violation": reference.diagnostics[
                "actuator_bound_violation"
            ],
            "terminal_pose_error": float(np.linalg.norm(terminal_pose_error)),
            "terminal_velocity_error": float(np.linalg.norm(reference.z_dot[-1])),
            "duration": reference.duration,
            "sample_count": float(len(reference.t)),
            "feasible_reference_exported": reference.diagnostics[
                "feasible_reference_exported"
            ],
        }
        results[case.name] = metrics

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            name: {key: float(value) for key, value in metrics.items()}
            for name, metrics in results.items()
        }
        output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True))

    return results


def minco_residuals(trajectory: MINCO_S3NU, head_pva, tail_pva) -> Dict[str, float]:
    head_pva = np.asarray(head_pva, dtype=float)
    tail_pva = np.asarray(tail_pva, dtype=float)
    start = trajectory.sample(0.0)
    end = trajectory.sample(trajectory.duration())
    pva_residual = max(
        float(np.linalg.norm(start.z - head_pva[0])),
        float(np.linalg.norm(start.z_dot - head_pva[1])),
        float(np.linalg.norm(start.z_ddot - head_pva[2])),
        float(np.linalg.norm(end.z - tail_pva[0])),
        float(np.linalg.norm(end.z_dot - tail_pva[1])),
        float(np.linalg.norm(end.z_ddot - tail_pva[2])),
    )

    intermediate_residual = 0.0
    continuity = {
        "z_dot_continuity_residual": 0.0,
        "z_ddot_continuity_residual": 0.0,
        "z_jerk_continuity_residual": 0.0,
        "z_snap_continuity_residual": 0.0,
    }
    for joint_index in range(trajectory.piece_count() - 1):
        left = trajectory.sample_piece(joint_index, trajectory.ts[joint_index])
        right = trajectory.sample_piece(joint_index + 1, 0.0)
        point = trajectory.inPs[joint_index]
        intermediate_residual = max(
            intermediate_residual,
            float(np.linalg.norm(left.z - point)),
            float(np.linalg.norm(right.z - point)),
        )
        for field, key in [
            ("z_dot", "z_dot_continuity_residual"),
            ("z_ddot", "z_ddot_continuity_residual"),
            ("z_jerk", "z_jerk_continuity_residual"),
            ("z_snap", "z_snap_continuity_residual"),
        ]:
            continuity[key] = max(
                continuity[key],
                float(np.linalg.norm(getattr(left, field) - getattr(right, field))),
            )

    continuity_residual = max(continuity.values()) if continuity else 0.0
    return {
        "pva_residual": pva_residual,
        "intermediate_residual": intermediate_residual,
        "continuity_residual": continuity_residual,
        **continuity,
    }


def outer_gradient_check(
    optimizer: MincoUsvOptimizer,
    head_pva,
    tail_pva,
    eps: float = 1e-6,
) -> float:
    inPs, ts = optimizer.initial_guess(head_pva, tail_pva)
    variables = optimizer.pack_variables(inPs, optimizer.times_to_theta(ts))
    _objective, analytical, _trajectory, _penalty = optimizer.objective_and_gradient(
        variables,
        head_pva,
        tail_pva,
    )
    finite_difference = np.zeros_like(analytical)
    for index in range(variables.size):
        plus_variables = variables.copy()
        minus_variables = variables.copy()
        plus_variables[index] += eps
        minus_variables[index] -= eps
        plus, _grad, _trajectory, _penalty = optimizer.objective_and_gradient(
            plus_variables,
            head_pva,
            tail_pva,
        )
        minus, _grad, _trajectory, _penalty = optimizer.objective_and_gradient(
            minus_variables,
            head_pva,
            tail_pva,
        )
        finite_difference[index] = (plus - minus) / (2.0 * eps)
    return float(
        np.linalg.norm(analytical - finite_difference) /
        max(1.0, np.linalg.norm(finite_difference))
    )


def main():
    output = Path("output/minco_offline_validation/metrics.json")
    results = run_offline_validation(output_json=output)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
