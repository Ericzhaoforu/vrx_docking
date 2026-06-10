import math

import numpy as np

from robotx_safe_docking_control.usv_flatness import UsvModelParams
from robotx_safe_docking_control.usv_lattice_frontend import CircularObstacle
from robotx_safe_docking_control.usv_lattice_frontend import UsvLatticeConfig
from robotx_safe_docking_control.usv_lattice_frontend import local_planning_goal
from robotx_safe_docking_control.usv_lattice_frontend import parse_circular_obstacles
from robotx_safe_docking_control.usv_lattice_frontend import plan_lattice_terminal
from robotx_safe_docking_control.usv_lattice_frontend import reach_horizon
from robotx_safe_docking_control import usv_lattice_frontend as lattice_frontend


def test_parse_circular_obstacles():
    obstacles = parse_circular_obstacles("1.0,2.0,0.5; -3, 4, 1.2")
    assert obstacles == (
        CircularObstacle(x=1.0, y=2.0, radius=0.5),
        CircularObstacle(x=-3.0, y=4.0, radius=1.2),
    )


def test_local_planning_goal_keeps_terminal_pose_for_far_goal():
    config = UsvLatticeConfig(
        goal_z=(6.0, 8.0, 2.4),
        max_local_goal_distance=2.5,
    )
    z = np.array([0.0, 0.0, 0.0], dtype=float)
    goal = np.array(config.goal_z, dtype=float)
    local_goal, local_goal_dot, terminal_stop = local_planning_goal(
        z, goal, config)

    assert np.allclose(local_goal[:2], goal[:2])
    assert math.isclose(local_goal[2], 2.4)
    assert np.allclose(local_goal_dot, 0.0)
    assert terminal_stop


def test_analytic_expansion_produces_exact_terminal_stop():
    params = UsvModelParams(min_thrust=-1000.0, max_thrust=1000.0)
    config = UsvLatticeConfig(
        goal_z=(0.5, 0.0, 0.2),
        primitive_duration=2.0,
        flat_accel_bounds=(2.0, 2.0, 2.0),
        flat_velocity_bounds=(3.0, 3.0, 3.0),
        tau_v_bar=1000.0,
        max_local_goal_distance=1.0,
        analytic_expansion=True,
    )
    plan = plan_lattice_terminal(
        z=np.zeros(3, dtype=float),
        z_dot=np.zeros(3, dtype=float),
        params=params,
        config=config,
    )

    assert plan.accepted
    assert np.allclose(plan.terminal_z, np.array(config.goal_z))
    assert np.allclose(plan.terminal_z_dot, np.zeros(3))
    assert np.allclose(plan.terminal_z_ddot, np.zeros(3))
    assert plan.diagnostics["frontend_terminal_stop"] == 1.0
    assert plan.diagnostics["frontend_goal_inside_horizon"] == 1.0
    assert plan.diagnostics["frontend_analytic_attempted"] == 1.0
    assert plan.diagnostics["frontend_analytic_success"] == 1.0


def test_cpp_lattice_matches_python_analytic_expansion(monkeypatch):
    if lattice_frontend._minco_cpp is None:
        return
    params = UsvModelParams(min_thrust=-1000.0, max_thrust=1000.0)
    config = UsvLatticeConfig(
        goal_z=(0.5, 0.0, 0.2),
        primitive_duration=2.0,
        flat_accel_bounds=(2.0, 2.0, 2.0),
        flat_velocity_bounds=(3.0, 3.0, 3.0),
        tau_v_bar=1000.0,
        max_local_goal_distance=1.0,
        analytic_expansion=True,
    )
    z = np.zeros(3, dtype=float)
    z_dot = np.zeros(3, dtype=float)

    monkeypatch.setenv("ROBOTX_LATTICE_USE_CPP", "0")
    py_plan = plan_lattice_terminal(z, z_dot, params, config)
    monkeypatch.setenv("ROBOTX_LATTICE_USE_CPP", "1")
    cpp_plan = plan_lattice_terminal(z, z_dot, params, config)

    assert cpp_plan.accepted == py_plan.accepted
    assert np.allclose(cpp_plan.terminal_z, py_plan.terminal_z)
    assert np.allclose(cpp_plan.terminal_z_dot, py_plan.terminal_z_dot)
    assert np.allclose(cpp_plan.terminal_z_ddot, py_plan.terminal_z_ddot)
    assert np.allclose(cpp_plan.path, py_plan.path)
    assert np.allclose(cpp_plan.path_times, py_plan.path_times)
    assert cpp_plan.diagnostics["frontend_analytic_success"] == (
        py_plan.diagnostics["frontend_analytic_success"])


def test_reach_goal_does_not_force_exact_terminal_stop():
    params = UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    config = UsvLatticeConfig(
        goal_z=(0.1, 0.0, 0.05),
        final_goal_radius=0.5,
        goal_tolerance=0.2,
        yaw_tolerance=0.1,
        analytic_expansion=False,
    )
    plan = plan_lattice_terminal(
        z=np.array([0.0, 0.0, 0.0], dtype=float),
        z_dot=np.array([0.2, 0.0, 0.0], dtype=float),
        params=params,
        config=config,
    )

    assert plan.accepted
    assert not np.allclose(plan.terminal_z, np.array(config.goal_z))
    assert plan.diagnostics["frontend_terminal_stop"] == 0.0
    assert plan.diagnostics["frontend_analytic_success"] == 0.0
    assert plan.diagnostics["frontend_reach_goal"] == 1.0
    assert np.linalg.norm(plan.terminal_z[:2] - np.array(config.goal_z[:2])) <= (
        config.goal_tolerance + 1e-9)
    assert abs(plan.terminal_z[2] - config.goal_z[2]) <= (
        config.yaw_tolerance + 1e-9)


def test_final_radius_alone_does_not_force_terminal_stop():
    params = UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    config = UsvLatticeConfig(
        goal_z=(0.1, 0.0, 1.0),
        final_goal_radius=0.5,
        goal_tolerance=0.05,
        yaw_tolerance=0.05,
        search_depth=0,
        analytic_expansion=False,
    )
    plan = plan_lattice_terminal(
        z=np.array([0.0, 0.0, 0.0], dtype=float),
        z_dot=np.zeros(3, dtype=float),
        params=params,
        config=config,
    )

    assert not plan.accepted
    assert plan.diagnostics["frontend_terminal_stop"] == 0.0


def test_tight_reach_goal_rejects_large_yaw_error():
    params = UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    config = UsvLatticeConfig(
        goal_z=(0.05, 0.0, 0.49),
        goal_tolerance=0.15,
        yaw_tolerance=0.12,
        search_depth=0,
        analytic_expansion=False,
    )
    plan = plan_lattice_terminal(
        z=np.zeros(3, dtype=float),
        z_dot=np.zeros(3, dtype=float),
        params=params,
        config=config,
    )

    assert not plan.accepted
    assert plan.diagnostics["frontend_reach_goal"] == 0.0
    assert plan.diagnostics["frontend_goal_position_error"] <= 0.15
    assert plan.diagnostics["frontend_goal_yaw_error"] > 0.12


def test_analytic_expansion_skipped_when_goal_outside_horizon():
    params = UsvModelParams(min_thrust=-1000.0, max_thrust=1000.0)
    config = UsvLatticeConfig(
        goal_z=(2.0, 0.0, 0.0),
        primitive_duration=0.5,
        check_num=5,
        search_depth=4,
        max_expansions=500,
        control_discretization=2,
        flat_accel_bounds=(1.0, 1.0, 1.0),
        flat_velocity_bounds=(3.0, 3.0, 3.0),
        tau_v_bar=1000.0,
        max_local_goal_distance=0.5,
        analytic_expansion=True,
    )
    plan = plan_lattice_terminal(
        z=np.zeros(3, dtype=float),
        z_dot=np.zeros(3, dtype=float),
        params=params,
        config=config,
    )

    assert plan.accepted
    assert plan.diagnostics["frontend_goal_inside_horizon"] == 0.0
    assert plan.diagnostics["frontend_analytic_attempted"] == 0.0
    assert plan.diagnostics["frontend_analytic_success"] == 0.0
    assert plan.diagnostics["frontend_reach_horizon"] == 1.0
    assert plan.diagnostics["frontend_terminal_stop"] == 0.0


def test_reach_horizon_uses_configured_distance():
    config = UsvLatticeConfig(max_local_goal_distance=1.0)

    assert reach_horizon(
        np.zeros(3, dtype=float), np.array([1.01, 0.0, 0.0]), config)
    assert not reach_horizon(
        np.zeros(3, dtype=float), np.array([0.99, 0.0, 0.0]), config)


def test_plan_returns_fast_planner_horizon_node_without_forced_stop():
    params = UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    config = UsvLatticeConfig(
        goal_z=(3.0, 0.0, 0.0),
        primitive_duration=0.5,
        primitive_dt=0.1,
        check_num=5,
        search_depth=4,
        max_expansions=500,
        control_discretization=2,
        flat_accel_bounds=(0.45, 0.2, 0.25),
        flat_velocity_bounds=(2.0, 0.5, 0.7),
        max_local_goal_distance=0.5,
        final_goal_radius=0.3,
        analytic_expansion=False,
    )
    plan = plan_lattice_terminal(
        z=np.zeros(3, dtype=float),
        z_dot=np.zeros(3, dtype=float),
        params=params,
        config=config,
    )

    assert plan.accepted
    assert plan.terminal_z[0] > 0.0
    assert plan.diagnostics["frontend_terminal_stop"] == 0.0
    assert np.linalg.norm(plan.terminal_z_dot) > 0.0
    assert plan.diagnostics["frontend_selected_depth"] > 0.0
    assert plan.diagnostics["frontend_candidate_count"] > 0.0
    assert plan.diagnostics["frontend_collision_free"] == 1.0
    assert math.isclose(plan.diagnostics["frontend_check_num"], 5.0)
    assert np.allclose(np.diff(plan.path_times[:6]), 0.1)


def test_plan_rejects_when_all_primitives_collide():
    params = UsvModelParams(min_thrust=-100.0, max_thrust=100.0)
    config = UsvLatticeConfig(
        goal_z=(3.0, 0.0, 0.0),
        primitive_duration=0.5,
        primitive_dt=0.1,
        search_depth=1,
        max_expansions=100,
        control_discretization=1,
        flat_accel_bounds=(0.45, 0.1, 0.1),
        obstacles=(CircularObstacle(x=0.05, y=0.0, radius=0.5),),
        obstacle_margin=0.0,
        analytic_expansion=False,
    )
    plan = plan_lattice_terminal(
        z=np.zeros(3, dtype=float),
        z_dot=np.zeros(3, dtype=float),
        params=params,
        config=config,
    )

    assert not plan.accepted
    assert plan.diagnostics["frontend_collision_free"] == 0.0
    assert plan.diagnostics["frontend_collision_reject_count"] > 0.0
