from robotx_safe_docking_control.minco_offline_validation import (
    run_offline_validation,
)


def test_offline_validation_straight_and_gentle_arc_pass():
    results = run_offline_validation()
    assert set(results.keys()) == {"straight", "gentle_arc"}
    for metrics in results.values():
        assert metrics["optimizer_success"] == 1.0
        assert metrics["objective_after"] < metrics["objective_before"]
        assert metrics["pva_residual"] < 1e-9
        assert metrics["intermediate_residual"] < 1e-9
        assert metrics["continuity_residual"] < 1e-8
        assert metrics["gradient_check_error"] < 1e-4
        assert metrics["velocity_violation"] <= 1e-9
        assert metrics["acceleration_violation"] <= 1e-9
        assert metrics["actuator_bound_violation"] <= 1e-9
        assert metrics["terminal_pose_error"] < 1e-9
        assert metrics["terminal_velocity_error"] < 1e-9
        assert metrics["feasible_reference_exported"] == 1.0
