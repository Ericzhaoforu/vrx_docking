from robotx_safe_docking_control.minco_adaptive_feasibility import (
    MincoAdaptiveConfig,
    default_adaptive_cases,
    duration_candidates,
)


def test_default_adaptive_cases_cover_distinct_boundary_statuses():
    names = {case.name for case in default_adaptive_cases()}
    assert {
        "local_offset_rest",
        "straight_moving_start",
        "diagonal_rest",
        "large_yaw_rest",
        "moving_terminal",
    } == names


def test_duration_candidates_respect_segment_minimum_and_are_ordered():
    case = default_adaptive_cases()[0]
    config = MincoAdaptiveConfig(
        min_total_time=4.0,
        duration_multipliers=(1.0, 1.5, 2.0),
        min_segment_time=0.5,
    )
    durations = duration_candidates(case, config, piece_count=3)
    assert durations == sorted(durations)
    assert all(duration >= 1.5 for duration in durations)
