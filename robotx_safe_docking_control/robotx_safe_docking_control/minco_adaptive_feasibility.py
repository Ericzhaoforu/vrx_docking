from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Optional

import numpy as np

from .minco_feasible_reference import minco_to_feasible_reference
from .minco_offline_validation import minco_residuals
from .minco_usv_optimizer import MincoUsvOptimizer
from .minco_usv_optimizer import MincoUsvOptimizerConfig
from .minco_usv_penalties import MincoUsvPenaltyConfig
from .usv_flatness import UsvModelParams
from .usv_flatness import wrap_angle


@dataclass
class MincoAdaptiveCase:
    name: str
    head_pva: np.ndarray
    tail_pva: np.ndarray


@dataclass
class MincoAdaptiveConfig:
    segment_candidates: tuple[int, ...] = (1, 2, 3, 4, 6)
    duration_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
    tau_v_schedule: tuple[float, ...] = (250.0, 50.0, 20.0, 10.0, 5.0)
    target_tau_v_bar: float = 5.0
    sample_dt: float = 0.2
    reference_speed: float = 0.45
    time_dilation: float = 1.8
    min_total_time: float = 8.0
    min_segment_time: float = 0.3
    max_iterations: int = 30
    smooth_weight: float = 1.0
    time_weight: float = 0.0
    velocity_bounds: tuple[float, float, float] = (2.0, 0.5, 0.6)
    acceleration_bounds: tuple[float, float, float] = (0.9, 0.5, 0.8)
    lambda_tau_v: float = 1.0
    lambda_velocity: float = 1.0
    lambda_acceleration: float = 1.0
    lambda_actuator: float = 10.0
    quadrature_order: int = 8
    penalty_mu: float = 20.0
    feasibility_tolerance: float = 1e-6
    sampled_tau_v_tolerance: float = 0.25
    residual_tolerance: float = 1e-6


@dataclass
class MincoAttemptSummary:
    case: str
    piece_count: int
    total_time: float
    tau_v_bar: float
    optimizer_success: bool
    optimizer_iterations: int
    objective_before: float
    objective_after: float
    max_abs_tau_v: float
    rms_tau_v: float
    velocity_violation: float
    acceleration_violation: float
    actuator_bound_violation: float
    pva_residual: float
    continuity_residual: float
    accepted: bool
    message: str


@dataclass
class MincoAdaptiveSelection:
    case: str
    accepted: bool
    piece_count: Optional[int]
    total_time: Optional[float]
    tau_v_bar: Optional[float]
    metrics: dict
    attempts: list[MincoAttemptSummary]


def default_adaptive_cases() -> list[MincoAdaptiveCase]:
    rest = np.zeros((3, 3), dtype=float)
    moving_start = rest.copy()
    moving_start[1] = np.array([0.20, 0.00, 0.02], dtype=float)
    moving_terminal = rest.copy()
    moving_terminal[1] = np.array([0.18, 0.05, 0.00], dtype=float)
    return [
        MincoAdaptiveCase(
            name="local_offset_rest",
            head_pva=rest.copy(),
            tail_pva=_pva([2.0, 0.4, 0.25]),
        ),
        MincoAdaptiveCase(
            name="straight_moving_start",
            head_pva=moving_start.copy(),
            tail_pva=_pva([3.0, 0.0, 0.0]),
        ),
        MincoAdaptiveCase(
            name="diagonal_rest",
            head_pva=rest.copy(),
            tail_pva=_pva([2.5, 1.0, 0.45]),
        ),
        MincoAdaptiveCase(
            name="large_yaw_rest",
            head_pva=rest.copy(),
            tail_pva=_pva([1.5, 0.2, 0.80]),
        ),
        MincoAdaptiveCase(
            name="moving_terminal",
            head_pva=moving_start.copy(),
            tail_pva=_pva([3.0, 0.5, 0.25], z_dot=moving_terminal[1]),
        ),
    ]


def run_adaptive_feasibility(
    cases: Iterable[MincoAdaptiveCase] | None = None,
    config: MincoAdaptiveConfig | None = None,
    params: UsvModelParams | None = None,
) -> dict[str, MincoAdaptiveSelection]:
    config = config or MincoAdaptiveConfig()
    params = params or UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    results = {}
    for case in cases or default_adaptive_cases():
        results[case.name] = solve_case(case, config, params)
    return results


def solve_case(
    case: MincoAdaptiveCase,
    config: MincoAdaptiveConfig,
    params: UsvModelParams,
) -> MincoAdaptiveSelection:
    attempts: list[MincoAttemptSummary] = []
    for piece_count in config.segment_candidates:
        for total_time in duration_candidates(case, config, piece_count):
            warm_inPs = None
            warm_ts = None
            final_reference = None
            final_result = None
            final_residuals = None
            failed = False
            for tau_v_bar in config.tau_v_schedule:
                penalty_config = MincoUsvPenaltyConfig(
                    params=params,
                    tau_v_bar=float(tau_v_bar),
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
                        piece_count=int(piece_count),
                        fixed_total_time=float(total_time),
                        reference_speed=float(config.reference_speed),
                        time_dilation=float(config.time_dilation),
                        min_segment_time=float(config.min_segment_time),
                        smooth_weight=float(config.smooth_weight),
                        time_weight=float(config.time_weight),
                        max_iterations=int(config.max_iterations),
                    ),
                )
                result = optimizer.optimize(
                    case.head_pva,
                    case.tail_pva,
                    initial_inPs=warm_inPs,
                    initial_ts=warm_ts,
                )
                reference = minco_to_feasible_reference(
                    f"adaptive_{case.name}",
                    result.trajectory,
                    params,
                    dt=float(config.sample_dt),
                    penalty_config=penalty_config,
                    optimizer_result=result,
                )
                residuals = minco_residuals(
                    result.trajectory,
                    case.head_pva,
                    case.tail_pva,
                )
                accepted = is_feasible(
                    result,
                    reference.diagnostics,
                    residuals,
                    config,
                    target_tau_v_bar=float(tau_v_bar),
                )
                attempts.append(attempt_summary(
                    case.name,
                    piece_count,
                    total_time,
                    tau_v_bar,
                    result,
                    reference.diagnostics,
                    residuals,
                    accepted,
                ))
                final_reference = reference
                final_result = result
                final_residuals = residuals
                if not result.success:
                    failed = True
                    break
                warm_inPs = result.inPs
                warm_ts = result.ts

            if failed or final_reference is None or final_result is None:
                continue
            if is_feasible(
                final_result,
                final_reference.diagnostics,
                final_residuals,
                config,
                target_tau_v_bar=float(config.target_tau_v_bar),
            ):
                return MincoAdaptiveSelection(
                    case=case.name,
                    accepted=True,
                    piece_count=int(piece_count),
                    total_time=float(total_time),
                    tau_v_bar=float(config.target_tau_v_bar),
                    metrics=selection_metrics(final_reference, final_result),
                    attempts=attempts,
                )

    return MincoAdaptiveSelection(
        case=case.name,
        accepted=False,
        piece_count=None,
        total_time=None,
        tau_v_bar=None,
        metrics={},
        attempts=attempts,
    )


def duration_candidates(
    case: MincoAdaptiveCase,
    config: MincoAdaptiveConfig,
    piece_count: int,
) -> list[float]:
    start = np.asarray(case.head_pva[0], dtype=float)
    goal = np.asarray(case.tail_pva[0], dtype=float)
    distance = float(np.linalg.norm(goal[:2] - start[:2]))
    yaw_delta = abs(wrap_angle(float(goal[2] - start[2])))
    base_time = max(
        float(config.min_total_time),
        config.time_dilation * distance / max(float(config.reference_speed), 1e-6),
        yaw_delta / 0.12,
        piece_count * float(config.min_segment_time),
    )
    candidates = []
    for multiplier in config.duration_multipliers:
        value = max(
            base_time * float(multiplier),
            piece_count * float(config.min_segment_time),
        )
        if not candidates or abs(value - candidates[-1]) > 1e-9:
            candidates.append(float(value))
    return candidates


def is_feasible(
    result,
    diagnostics: dict,
    residuals: dict,
    config: MincoAdaptiveConfig,
    target_tau_v_bar: float,
) -> bool:
    return bool(
        result.success and
        diagnostics["max_abs_tau_v"] <= (
            float(target_tau_v_bar) + float(config.sampled_tau_v_tolerance)
        ) and
        diagnostics["velocity_violation"] <= config.feasibility_tolerance and
        diagnostics["acceleration_violation"] <= config.feasibility_tolerance and
        diagnostics["actuator_bound_violation"] <= config.feasibility_tolerance and
        residuals["pva_residual"] <= config.residual_tolerance and
        residuals["continuity_residual"] <= config.residual_tolerance
    )


def attempt_summary(
    case_name,
    piece_count,
    total_time,
    tau_v_bar,
    result,
    diagnostics,
    residuals,
    accepted,
) -> MincoAttemptSummary:
    return MincoAttemptSummary(
        case=str(case_name),
        piece_count=int(piece_count),
        total_time=float(total_time),
        tau_v_bar=float(tau_v_bar),
        optimizer_success=bool(result.success),
        optimizer_iterations=int(result.iterations),
        objective_before=float(result.objective_before),
        objective_after=float(result.objective_after),
        max_abs_tau_v=float(diagnostics["max_abs_tau_v"]),
        rms_tau_v=float(diagnostics["rms_tau_v"]),
        velocity_violation=float(diagnostics["velocity_violation"]),
        acceleration_violation=float(diagnostics["acceleration_violation"]),
        actuator_bound_violation=float(diagnostics["actuator_bound_violation"]),
        pva_residual=float(residuals["pva_residual"]),
        continuity_residual=float(residuals["continuity_residual"]),
        accepted=bool(accepted),
        message=str(result.message),
    )


def selection_metrics(reference, result) -> dict:
    diagnostics = dict(reference.diagnostics)
    return {
        "duration": float(reference.duration),
        "sample_count": float(len(reference.t)),
        "piece_count": float(result.trajectory.piece_count()),
        "segment_times": [float(value) for value in result.ts],
        "intermediate_points": np.asarray(result.inPs, dtype=float).tolist(),
        "optimizer_iterations": float(result.iterations),
        "objective_before": float(result.objective_before),
        "objective_after": float(result.objective_after),
        "max_abs_tau_v": float(diagnostics["max_abs_tau_v"]),
        "rms_tau_v": float(diagnostics["rms_tau_v"]),
        "velocity_violation": float(diagnostics["velocity_violation"]),
        "acceleration_violation": float(diagnostics["acceleration_violation"]),
        "actuator_bound_violation": float(diagnostics["actuator_bound_violation"]),
        "max_abs_tau_u": float(diagnostics["max_abs_tau_u"]),
        "max_abs_tau_r": float(diagnostics["max_abs_tau_r"]),
        "max_abs_left_thrust": float(diagnostics["max_abs_left_thrust"]),
        "max_abs_right_thrust": float(diagnostics["max_abs_right_thrust"]),
        "max_speed": float(diagnostics["max_speed"]),
        "max_yaw_rate": float(diagnostics["max_yaw_rate"]),
        "final_z": np.asarray(reference.z[-1], dtype=float).tolist(),
        "final_z_dot": np.asarray(reference.z_dot[-1], dtype=float).tolist(),
    }


def write_results(
    selections: dict[str, MincoAdaptiveSelection],
    output_dir: Path,
    config: MincoAdaptiveConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: {
            "case": selection.case,
            "accepted": selection.accepted,
            "piece_count": selection.piece_count,
            "total_time": selection.total_time,
            "tau_v_bar": selection.tau_v_bar,
            "metrics": selection.metrics,
            "attempts": [asdict(attempt) for attempt in selection.attempts],
        }
        for name, selection in selections.items()
    }
    (output_dir / "adaptive_minco_feasibility.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "results": serializable,
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_summary_plot(selections, output_dir)


def write_summary_plot(
    selections: dict[str, MincoAdaptiveSelection],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    names = list(selections.keys())
    max_tau_v = [
        selection.metrics.get("max_abs_tau_v", np.nan)
        for selection in selections.values()
    ]
    piece_counts = [
        np.nan if selection.piece_count is None else selection.piece_count
        for selection in selections.values()
    ]
    durations = [
        np.nan if selection.total_time is None else selection.total_time
        for selection in selections.values()
    ]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].bar(x, max_tau_v)
    axes[0].set_ylabel("max |tau_v| [N]")
    axes[0].grid(True, axis="y")
    axes[1].bar(x, piece_counts)
    axes[1].set_ylabel("selected pieces")
    axes[1].grid(True, axis="y")
    axes[2].bar(x, durations)
    axes[2].set_ylabel("total time [s]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=25, ha="right")
    axes[2].grid(True, axis="y")
    fig.suptitle("Adaptive MINCO feasibility sweep")
    fig.savefig(output_dir / "adaptive_minco_feasibility_summary.png", dpi=160)


def _pva(z, z_dot=None, z_ddot=None) -> np.ndarray:
    return np.vstack((
        np.asarray(z, dtype=float).reshape(3),
        np.zeros(3, dtype=float) if z_dot is None else np.asarray(
            z_dot, dtype=float).reshape(3),
        np.zeros(3, dtype=float) if z_ddot is None else np.asarray(
            z_ddot, dtype=float).reshape(3),
    ))


def _case_filter(cases, selected_names: set[str] | None):
    if not selected_names:
        return cases
    return [case for case in cases if case.name in selected_names]


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive MINCO segment/time/tau_v feasibility sweeps.")
    parser.add_argument(
        "--output-dir",
        default="/home/zjy/vrx_docking/output/minco_adaptive_feasibility",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this case name. Can be passed multiple times.",
    )
    parser.add_argument("--target-tau-v-bar", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--segments",
        default="1,2,3,4,6",
        help="Comma-separated segment candidates.",
    )
    parser.add_argument(
        "--duration-multipliers",
        default="1.0,1.25,1.5,2.0",
        help="Comma-separated duration multipliers.",
    )
    args = parser.parse_args()

    config = MincoAdaptiveConfig(
        segment_candidates=_parse_int_tuple(args.segments),
        duration_multipliers=_parse_float_tuple(args.duration_multipliers),
        target_tau_v_bar=float(args.target_tau_v_bar),
        tau_v_schedule=tuple(
            value for value in (250.0, 50.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0)
            if value >= float(args.target_tau_v_bar)
        ),
        max_iterations=int(args.max_iterations),
    )
    if config.tau_v_schedule[-1] != config.target_tau_v_bar:
        config = MincoAdaptiveConfig(
            **{
                **asdict(config),
                "tau_v_schedule": (
                    *config.tau_v_schedule,
                    config.target_tau_v_bar,
                ),
            }
        )
    cases = _case_filter(default_adaptive_cases(), set(args.case))
    selections = run_adaptive_feasibility(cases, config)
    output_dir = Path(args.output_dir)
    write_results(selections, output_dir, config)
    for name, selection in selections.items():
        if selection.accepted:
            metrics = selection.metrics
            print(
                f"{name}: accepted M={selection.piece_count} "
                f"T={selection.total_time:.2f}s "
                f"max_tau_v={metrics['max_abs_tau_v']:.3f}N "
                f"max_thrust=max({metrics['max_abs_left_thrust']:.2f},"
                f"{metrics['max_abs_right_thrust']:.2f})N"
            )
        else:
            print(f"{name}: no feasible selection")
    print(f"Wrote {output_dir}")


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(
        float(part.strip())
        for part in str(text).split(",")
        if part.strip()
    )
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(
        int(part.strip())
        for part in str(text).split(",")
        if part.strip()
    )
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


if __name__ == "__main__":
    main()
