from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    import robotx_minco_cpp as _minco_cpp
except Exception:  # pragma: no cover - optional compiled acceleration
    _minco_cpp = None

from .usv_flatness import UsvModelParams
from .usv_flatness import body_to_flat_acceleration
from .usv_flatness import body_to_flat_velocity
from .usv_flatness import flat_to_body_velocity_at_yaw
from .usv_flatness import generalized_force_to_thrust
from .usv_flatness import nominal_coriolis_times_velocity
from .usv_flatness import nominal_damping
from .usv_flatness import theta_tau_nominal
from .usv_flatness import wrap_angle


@dataclass(frozen=True)
class CircularObstacle:
    x: float
    y: float
    radius: float


@dataclass
class UsvLatticeConfig:
    goal_z: tuple[float, float, float] = (6.0, 12.0, 2.4)
    primitive_duration: float = 2.0
    primitive_dt: float = 0.25
    check_num: int = 5
    search_depth: int = 24
    max_expansions: int = 2000
    control_discretization: int = 1
    sample_thrust_limit: float = 0.0
    minco_piece_count: int = 2
    minco_piece_count_min: int = 3
    minco_piece_count_max: int = 8
    flat_accel_bounds: tuple[float, float, float] = (0.5, 0.5, 0.12)
    flat_velocity_bounds: tuple[float, float, float] = (2.0, 2.0, 0.4)
    time_weight: float = 1.0
    heuristic_weight: float = 3.0
    tau_v_bar: float = 12.0
    max_local_goal_distance: float = 5.0
    local_goal_speed: float = 0.45
    final_goal_radius: float = 0.8
    goal_tolerance: float = 0.15
    yaw_tolerance: float = 0.12
    obstacle_margin: float = 0.5
    obstacles: tuple[CircularObstacle, ...] = ()
    grid_resolution_xy: float = 0.25
    grid_resolution_yaw: float = 0.25
    grid_resolution_vxy: float = 0.25
    grid_resolution_yaw_rate: float = 0.05
    analytic_expansion: bool = True


@dataclass
class LatticeNode:
    z: np.ndarray
    z_dot: np.ndarray
    z_ddot: np.ndarray
    nu: np.ndarray
    g_cost: float
    h_cost: float
    f_cost: float
    depth: int
    parent: LatticeNode | None
    primitive_u: np.ndarray
    segment_path: np.ndarray
    segment_z_dot: np.ndarray
    segment_z_ddot: np.ndarray
    segment_nu: np.ndarray
    segment_times: np.ndarray
    key: tuple[int, ...]


@dataclass
class LatticePlan:
    accepted: bool
    terminal_z: np.ndarray
    terminal_z_dot: np.ndarray
    terminal_z_ddot: np.ndarray
    path: np.ndarray
    path_times: np.ndarray
    first_tau: np.ndarray
    initial_inPs: np.ndarray | None
    initial_ts: np.ndarray | None
    diagnostics: dict[str, float]


def _params_to_cpp_dict(params: UsvModelParams) -> dict[str, float]:
    return {
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
    }


def _config_to_cpp_dict(config: UsvLatticeConfig) -> dict:
    return {
        "goal_z": tuple(float(item) for item in config.goal_z),
        "primitive_duration": float(config.primitive_duration),
        "primitive_dt": float(config.primitive_dt),
        "check_num": int(config.check_num),
        "search_depth": int(config.search_depth),
        "max_expansions": int(config.max_expansions),
        "control_discretization": int(config.control_discretization),
        "sample_thrust_limit": float(config.sample_thrust_limit),
        "minco_piece_count_min": int(config.minco_piece_count_min),
        "minco_piece_count_max": int(config.minco_piece_count_max),
        "flat_accel_bounds": tuple(float(item) for item in config.flat_accel_bounds),
        "flat_velocity_bounds": tuple(float(item) for item in config.flat_velocity_bounds),
        "time_weight": float(config.time_weight),
        "heuristic_weight": float(config.heuristic_weight),
        "tau_v_bar": float(config.tau_v_bar),
        "max_local_goal_distance": float(config.max_local_goal_distance),
        "local_goal_speed": float(config.local_goal_speed),
        "final_goal_radius": float(config.final_goal_radius),
        "goal_tolerance": float(config.goal_tolerance),
        "yaw_tolerance": float(config.yaw_tolerance),
        "obstacle_margin": float(config.obstacle_margin),
        "grid_resolution_xy": float(config.grid_resolution_xy),
        "grid_resolution_yaw": float(config.grid_resolution_yaw),
        "grid_resolution_vxy": float(config.grid_resolution_vxy),
        "grid_resolution_yaw_rate": float(config.grid_resolution_yaw_rate),
        "analytic_expansion": bool(config.analytic_expansion),
        "obstacles": [
            (float(obstacle.x), float(obstacle.y), float(obstacle.radius))
            for obstacle in config.obstacles
        ],
    }


def _optional_array_from_cpp(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    return array.copy()


def _lattice_plan_from_cpp(result) -> LatticePlan:
    diagnostics = {
        str(key): float(value)
        for key, value in dict(result["diagnostics"]).items()
    }
    return LatticePlan(
        accepted=bool(result["accepted"]),
        terminal_z=np.asarray(result["terminal_z"], dtype=float).reshape(3),
        terminal_z_dot=np.asarray(result["terminal_z_dot"], dtype=float).reshape(3),
        terminal_z_ddot=np.asarray(result["terminal_z_ddot"], dtype=float).reshape(3),
        path=np.asarray(result["path"], dtype=float),
        path_times=np.asarray(result["path_times"], dtype=float),
        first_tau=np.asarray(result["first_tau"], dtype=float).reshape(3),
        initial_inPs=_optional_array_from_cpp(result["initial_inPs"]),
        initial_ts=_optional_array_from_cpp(result["initial_ts"]),
        diagnostics=diagnostics,
    )


def plan_lattice_terminal(
    z,
    z_dot,
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> LatticePlan:
    use_cpp = (
        _minco_cpp is not None and
        os.environ.get("ROBOTX_LATTICE_USE_CPP", "1") != "0")
    if use_cpp:
        result = _minco_cpp.plan_lattice_terminal(
            np.asarray(z, dtype=float).reshape(3),
            np.asarray(z_dot, dtype=float).reshape(3),
            _params_to_cpp_dict(params),
            _config_to_cpp_dict(config),
        )
        return _lattice_plan_from_cpp(result)
    return _plan_lattice_terminal_python(z, z_dot, params, config)


def _plan_lattice_terminal_python(
    z,
    z_dot,
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> LatticePlan:
    """Fast-Planner-style kinodynamic A* in USV flat state space.

    The search follows Fast-Planner Section III's A* structure, actual
    cost-plus-heuristic ranking, and cubic minimum-control heuristic. Primitive
    expansion is USV-specific: sampled constant [tau_u, tau_r] inputs are
    integrated through the nominal differential-thrust USV model with
    tau_v = 0.
    """
    z = np.asarray(z, dtype=float).reshape(3)
    z_dot = np.asarray(z_dot, dtype=float).reshape(3)
    goal = np.asarray(config.goal_z, dtype=float).reshape(3)
    search_goal, search_goal_dot, terminal_stop = local_planning_goal(
        z, goal, config)
    global_goal_distance = float(np.linalg.norm(goal[:2] - z[:2]))

    controls = discretized_generalized_inputs(params, config)
    root_nu = flat_to_body_velocity_at_yaw(float(z[2]), z_dot)
    heuristic_weight = max(float(config.heuristic_weight), 0.0)
    root_h, _root_t = heuristic_cost(z, z_dot, search_goal, search_goal_dot, config)
    root = LatticeNode(
        z=z.copy(),
        z_dot=z_dot.copy(),
        z_ddot=np.zeros(3, dtype=float),
        nu=root_nu.copy(),
        g_cost=0.0,
        h_cost=root_h,
        f_cost=heuristic_weight * root_h,
        depth=0,
        parent=None,
        primitive_u=np.zeros(2, dtype=float),
        segment_path=np.asarray([z.copy()], dtype=float),
        segment_z_dot=np.asarray([z_dot.copy()], dtype=float),
        segment_z_ddot=np.asarray([np.zeros(3, dtype=float)], dtype=float),
        segment_nu=np.asarray([root_nu.copy()], dtype=float),
        segment_times=np.asarray([0.0], dtype=float),
        key=grid_key(z, root_nu, config),
    )

    open_heap: list[tuple[float, int, LatticeNode]] = []
    open_best = {root.key: 0.0}
    closed: set[tuple[int, ...]] = set()
    heapq.heappush(open_heap, (root.f_cost, 0, root))
    serial = 1
    candidate_count = 0
    collision_rejects = 0
    infeasible_rejects = 0
    expansions = 0
    analytic_attempts = 0
    best_seen = root
    analytic_allowed = goal_inside_local_horizon(z, search_goal, config)

    while open_heap and expansions < int(config.max_expansions):
        _fc, _serial, current = heapq.heappop(open_heap)
        if current.key in closed:
            continue
        closed.add(current.key)
        expansions += 1
        if current.f_cost < best_seen.f_cost:
            best_seen = current

        if bool(config.analytic_expansion) and analytic_allowed:
            analytic_attempts += 1
            analytic = analytic_expand(
                current, search_goal, search_goal_dot, z, params, config)
            if analytic is not None:
                return plan_from_node(
                    analytic, params, config, goal, search_goal, terminal_stop,
                    candidate_count, collision_rejects, infeasible_rejects,
                    expansions, analytic_attempts=analytic_attempts,
                    analytic_success=True)

        if reach_goal(current.z, search_goal, terminal_stop, config):
            return plan_from_node(
                current, params, config, goal, search_goal,
                terminal_stop=False, candidate_count=candidate_count,
                collision_rejects=collision_rejects,
                infeasible_rejects=infeasible_rejects, expansions=expansions,
                analytic_attempts=analytic_attempts,
                analytic_success=False, reached_goal=True)

        if (
                not analytic_allowed and
                reach_horizon_toward_goal(z, current.z, search_goal, config)):
            return plan_from_node(
                current, params, config, goal, search_goal,
                terminal_stop=False, candidate_count=candidate_count,
                collision_rejects=collision_rejects,
                infeasible_rejects=infeasible_rejects, expansions=expansions,
                analytic_attempts=analytic_attempts,
                analytic_success=False, reached_horizon=True)

        if current.depth >= int(config.search_depth):
            continue

        best_children: dict[tuple[int, ...], LatticeNode] = {}
        for control in controls:
            candidate_count += 1
            child = rollout_usv_primitive(current, control, params, config)
            if collides(child.segment_path[1:, :2], config.obstacles,
                        config.obstacle_margin):
                collision_rejects += 1
                continue
            if not check_feasible(child, params, config):
                infeasible_rejects += 1
                continue
            if child.key in closed:
                continue
            edge = edge_cost(control, params, config)
            h_cost, _h_time = heuristic_cost(
                child.z, child.z_dot, search_goal, search_goal_dot, config)
            child.g_cost = current.g_cost + edge
            child.h_cost = h_cost
            child.f_cost = child.g_cost + heuristic_weight * child.h_cost
            previous = best_children.get(child.key)
            if previous is None or child.f_cost < previous.f_cost:
                best_children[child.key] = child

        for child in best_children.values():
            previous_g = open_best.get(child.key)
            if previous_g is not None and child.g_cost >= previous_g:
                continue
            open_best[child.key] = child.g_cost
            heapq.heappush(open_heap, (child.f_cost, serial, child))
            serial += 1

    diagnostics = _diagnostics(
        config, z, goal, search_goal, best_seen.z, global_goal_distance,
        selected_depth=best_seen.depth, candidate_count=candidate_count,
        collision_rejects=collision_rejects,
        infeasible_rejects=infeasible_rejects,
        selected_cost=best_seen.f_cost, terminal_stop=False,
        analytic_attempts=analytic_attempts, analytic_success=False,
        expansions=expansions)
    diagnostics["frontend_collision_free"] = 0.0
    return LatticePlan(
        accepted=False,
        terminal_z=z.copy(),
        terminal_z_dot=z_dot.copy(),
        terminal_z_ddot=np.zeros(3, dtype=float),
        path=np.asarray([z.copy()], dtype=float),
        path_times=np.asarray([0.0], dtype=float),
        first_tau=np.zeros(3, dtype=float),
        initial_inPs=None,
        initial_ts=None,
        diagnostics=diagnostics,
    )


def discretized_generalized_inputs(
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> np.ndarray:
    """Sample constant underactuated generalized inputs for primitive rollout.

    The front end keeps Fast-Planner's sampled-control expansion, but samples
    the physical actuator commands first.  Each discrete pair [T_L, T_R] maps
    to tau = [tau_u, 0, tau_r], so every primitive is actuator-feasible by
    construction.
    """
    r = max(int(config.control_discretization), 1)
    sample_limit = float(config.sample_thrust_limit)
    if sample_limit > 0.0:
        lower = max(float(params.min_thrust), -abs(sample_limit))
        upper = min(float(params.max_thrust), abs(sample_limit))
    else:
        lower = float(params.min_thrust)
        upper = float(params.max_thrust)
    left_levels = np.linspace(lower, upper, 2 * r + 1)
    right_levels = np.linspace(lower, upper, 2 * r + 1)
    controls = []
    seen: set[tuple[float, float]] = set()
    for left in left_levels:
        for right in right_levels:
            tau_u = float(left + right)
            tau_r = float(params.thruster_half_spacing * (right - left))
            key = (round(tau_u, 12), round(tau_r, 12))
            if key in seen:
                continue
            seen.add(key)
            controls.append([tau_u, tau_r])
    if not controls:
        controls.append([0.0, 0.0])
    return np.asarray(controls, dtype=float)


def local_planning_goal(
    z,
    goal,
    config: UsvLatticeConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return the terminal flat goal state for kinodynamic search.

    Fast-Planner's kinodynamic heuristic estimates cost-to-go from the current
    state to the requested terminal state. Keep the terminal yaw in that target;
    replacing it with a line-of-sight yaw creates a late terminal spin.
    """
    z = np.asarray(z, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    del config
    goal[2] = z[2] + wrap_angle(float(goal[2] - z[2]))
    return goal.copy(), np.zeros(3, dtype=float), True


def rollout_usv_primitive(
    node: LatticeNode,
    control,
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> LatticeNode:
    control = np.asarray(control, dtype=float).reshape(2)
    duration = max(float(config.primitive_duration), 1e-6)
    steps = max(int(config.check_num), 1)
    times = np.linspace(0.0, duration, steps + 1)
    segment = []
    segment_z_dot = []
    segment_z_ddot = []
    segment_nu = []
    state = np.concatenate((node.z, node.nu)).astype(float)
    previous_t = 0.0
    for t in times:
        dt = float(t - previous_t)
        if dt > 0.0:
            state = rk4_usv_step(state, control, params, dt)
            previous_t = float(t)
        z_t = state[:3].copy()
        nu_t = state[3:].copy()
        z_dot_t = body_to_flat_velocity(float(z_t[2]), nu_t)
        nu_dot_t = usv_body_acceleration(nu_t, control, params)
        z_ddot_t = body_to_flat_acceleration(float(z_t[2]), nu_t, nu_dot_t)
        z_t[2] = wrap_angle(float(z_t[2]))
        segment.append(z_t)
        segment_z_dot.append(z_dot_t)
        segment_z_ddot.append(z_ddot_t)
        segment_nu.append(nu_t)
    segment = np.asarray(segment, dtype=float)
    segment_z_dot = np.asarray(segment_z_dot, dtype=float)
    segment_z_ddot = np.asarray(segment_z_ddot, dtype=float)
    segment_nu = np.asarray(segment_nu, dtype=float)
    z_next = segment[-1].copy()
    z_dot_next = segment_z_dot[-1].copy()
    return LatticeNode(
        z=z_next,
        z_dot=z_dot_next,
        z_ddot=segment_z_ddot[-1].copy(),
        nu=segment_nu[-1].copy(),
        g_cost=0.0,
        h_cost=0.0,
        f_cost=0.0,
        depth=node.depth + 1,
        parent=node,
        primitive_u=control.copy(),
        segment_path=segment,
        segment_z_dot=segment_z_dot,
        segment_z_ddot=segment_z_ddot,
        segment_nu=segment_nu,
        segment_times=times,
        key=grid_key(z_next, segment_nu[-1], config),
    )


def usv_body_acceleration(nu, control, params: UsvModelParams) -> np.ndarray:
    nu = np.asarray(nu, dtype=float).reshape(3)
    tau_u, tau_r = np.asarray(control, dtype=float).reshape(2)
    tau = np.array([float(tau_u), 0.0, float(tau_r)], dtype=float)
    rhs = (
        tau -
        nominal_coriolis_times_velocity(nu, params) -
        nominal_damping(nu, params)
    )
    return rhs / np.array([params.mass, params.mass, params.iz], dtype=float)


def usv_state_derivative(state, control, params: UsvModelParams) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(6)
    z = state[:3].copy()
    nu = state[3:].copy()
    z_dot = body_to_flat_velocity(float(z[2]), nu)
    nu_dot = usv_body_acceleration(nu, control, params)
    return np.concatenate((z_dot, nu_dot))


def rk4_usv_step(state, control, params: UsvModelParams, dt: float) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(6)
    dt = float(dt)
    k1 = usv_state_derivative(state, control, params)
    k2 = usv_state_derivative(state + 0.5 * dt * k1, control, params)
    k3 = usv_state_derivative(state + 0.5 * dt * k2, control, params)
    k4 = usv_state_derivative(state + dt * k3, control, params)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[2] = wrap_angle(float(next_state[2]))
    return next_state


def analytic_expand(
    node: LatticeNode,
    goal,
    goal_dot,
    root_z,
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> LatticeNode | None:
    cost, duration = heuristic_cost(node.z, node.z_dot, goal, goal_dot, config)
    if not np.isfinite(cost) or not np.isfinite(duration):
        return None
    duration = max(float(duration), max(float(config.primitive_duration), 1e-3))
    coeffs = cubic_connection_coefficients(
        node.z, node.z_dot, goal, goal_dot, duration)
    steps = max(10, int(config.check_num), 2)
    times = np.linspace(0.0, duration, steps + 1)
    segment = []
    z_dot_samples = []
    z_ddot_samples = []
    for t in times:
        z_t, z_dot_t, z_ddot_t = evaluate_cubic_connection(
            node.z, node.z_dot, coeffs, t)
        z_t[2] = wrap_angle(float(z_t[2]))
        segment.append(z_t)
        z_dot_samples.append(z_dot_t)
        z_ddot_samples.append(z_ddot_t)
    segment = np.asarray(segment, dtype=float)
    z_dot_samples = np.asarray(z_dot_samples, dtype=float)
    z_ddot_samples = np.asarray(z_ddot_samples, dtype=float)
    if not path_inside_local_horizon(root_z, segment, config):
        return None
    if collides(segment[:, :2], config.obstacles, config.obstacle_margin):
        return None
    for z_t, z_dot_t, z_ddot_t in zip(segment, z_dot_samples, z_ddot_samples):
        if not check_flat_bounds(z_dot_t, z_ddot_t, config):
            return None
        if not check_usv_flat_feasible(z_t, z_dot_t, z_ddot_t, params, config):
            return None
    child = LatticeNode(
        z=segment[-1].copy(),
        z_dot=z_dot_samples[-1].copy(),
        z_ddot=z_ddot_samples[-1].copy(),
        nu=flat_to_body_velocity_at_yaw(
            float(segment[-1, 2]), z_dot_samples[-1]).copy(),
        g_cost=node.g_cost + cost,
        h_cost=0.0,
        f_cost=node.g_cost + cost,
        depth=node.depth + 1,
        parent=node,
        primitive_u=np.zeros(2, dtype=float),
        segment_path=segment,
        segment_z_dot=z_dot_samples,
        segment_z_ddot=z_ddot_samples,
        segment_nu=np.asarray([
            flat_to_body_velocity_at_yaw(float(z_t[2]), z_dot_t)
            for z_t, z_dot_t in zip(segment, z_dot_samples)
        ], dtype=float),
        segment_times=times,
        key=grid_key(segment[-1], flat_to_body_velocity_at_yaw(
            float(segment[-1, 2]), z_dot_samples[-1]), config),
    )
    return child


def edge_cost(control, params: UsvModelParams, config: UsvLatticeConfig) -> float:
    tau_u, tau_r = np.asarray(control, dtype=float).reshape(2)
    left, right = generalized_force_to_thrust(tau_u, tau_r, params)
    thrust_scale = max(
        abs(float(params.min_thrust)), abs(float(params.max_thrust)), 1e-6)
    effort = (float(left) / thrust_scale)**2 + (
        float(right) / thrust_scale)**2
    return float((effort + float(config.time_weight)) *
                 max(float(config.primitive_duration), 1e-6))


def heuristic_cost(
    z,
    z_dot,
    goal,
    goal_dot,
    config: UsvLatticeConfig,
) -> tuple[float, float]:
    z = np.asarray(z, dtype=float).reshape(3)
    z_dot = np.asarray(z_dot, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    goal_dot = np.asarray(goal_dot, dtype=float).reshape(3)
    goal = goal.copy()
    goal[2] = z[2] + wrap_angle(float(goal[2] - z[2]))
    p_delta = goal - z
    v_delta = goal_dot - z_dot
    rho = max(float(config.time_weight), 1e-9)
    a = float(np.sum(12.0 * p_delta * p_delta))
    b = float(np.sum(-24.0 * p_delta * z_dot - 12.0 * p_delta * v_delta))
    c = float(np.sum(12.0 * z_dot * z_dot + 12.0 * z_dot * v_delta +
                     4.0 * v_delta * v_delta))
    roots = np.roots([rho, 0.0, -c, -2.0 * b, -3.0 * a])
    candidates = [
        float(root.real) for root in roots
        if abs(root.imag) < 1e-7 and root.real > 1e-6
    ]
    velocity_bounds = np.maximum(
        np.asarray(config.flat_velocity_bounds, dtype=float).reshape(3), 1e-6)
    lower_bound = max(
        float(config.primitive_duration),
        float(np.max(np.abs(p_delta) / (0.5 * velocity_bounds))),
        1e-3,
    )
    if not candidates:
        distance = float(np.linalg.norm(p_delta[:2]))
        speed = max(float(config.local_goal_speed), 0.1)
        candidates.append(max(distance / speed, lower_bound))
    candidates.append(lower_bound)
    best_cost = float("inf")
    best_time = lower_bound
    for duration in candidates:
        duration = max(float(duration), lower_bound)
        cost = cubic_connection_cost(z, z_dot, goal, goal_dot, duration, rho)
        if cost < best_cost:
            best_cost = cost
            best_time = duration
    return best_cost, best_time


def cubic_connection_coefficients(z, z_dot, goal, goal_dot, duration: float):
    z = np.asarray(z, dtype=float).reshape(3)
    z_dot = np.asarray(z_dot, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    goal_dot = np.asarray(goal_dot, dtype=float).reshape(3)
    goal = goal.copy()
    goal[2] = z[2] + wrap_angle(float(goal[2] - z[2]))
    duration = max(float(duration), 1e-9)
    d = goal - z - z_dot * duration
    v = goal_dot - z_dot
    alpha = -12.0 * d / duration**3 + 6.0 * v / duration**2
    beta = 6.0 * d / duration**2 - 2.0 * v / duration
    return alpha, beta


def cubic_connection_cost(
    z,
    z_dot,
    goal,
    goal_dot,
    duration: float,
    rho: float,
) -> float:
    alpha, beta = cubic_connection_coefficients(
        z, z_dot, goal, goal_dot, duration)
    return float(
        np.sum(
            alpha * alpha * duration**3 / 3.0 +
            alpha * beta * duration**2 +
            beta * beta * duration
        ) + rho * duration
    )


def evaluate_cubic_connection(z, z_dot, coeffs, t: float):
    z = np.asarray(z, dtype=float).reshape(3)
    z_dot = np.asarray(z_dot, dtype=float).reshape(3)
    alpha, beta = coeffs
    t = float(t)
    z_t = z + z_dot * t + 0.5 * beta * t * t + (alpha * t**3) / 6.0
    z_dot_t = z_dot + beta * t + 0.5 * alpha * t * t
    z_ddot_t = beta + alpha * t
    return z_t, z_dot_t, z_ddot_t


def check_feasible(
    node: LatticeNode,
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> bool:
    del params
    if len(node.segment_path) < 2:
        return False
    for nu_t in node.segment_nu:
        if not check_body_velocity_bounds(nu_t, config):
            return False
    return True


def check_body_velocity_bounds(nu, config: UsvLatticeConfig) -> bool:
    velocity_bounds = np.asarray(config.flat_velocity_bounds, dtype=float).reshape(3)
    nu = np.asarray(nu, dtype=float).reshape(3)
    return bool(np.all(np.abs(nu) <= velocity_bounds + 1e-9))


def check_flat_bounds(z_dot, z_ddot, config: UsvLatticeConfig) -> bool:
    velocity_bounds = np.asarray(config.flat_velocity_bounds, dtype=float).reshape(3)
    accel_bounds = np.asarray(config.flat_accel_bounds, dtype=float).reshape(3)
    return bool(
        np.all(np.abs(z_dot) <= velocity_bounds + 1e-9) and
        np.all(np.abs(z_ddot) <= accel_bounds + 1e-9)
    )


def check_usv_flat_feasible(
    z,
    z_dot,
    z_ddot,
    params: UsvModelParams,
    config: UsvLatticeConfig,
) -> bool:
    tau = theta_tau_nominal(z, z_dot, z_ddot, params)
    if abs(float(tau[1])) > float(config.tau_v_bar) + 1e-9:
        return False
    left, right = generalized_force_to_thrust(float(tau[0]), float(tau[2]), params)
    return bool(
        params.min_thrust <= left <= params.max_thrust and
        params.min_thrust <= right <= params.max_thrust
    )


def plan_from_node(
    node: LatticeNode,
    params: UsvModelParams,
    config: UsvLatticeConfig,
    global_goal,
    search_goal,
    terminal_stop: bool,
    candidate_count: int,
    collision_rejects: int,
    infeasible_rejects: int,
    expansions: int,
    analytic_attempts: int,
    analytic_success: bool,
    reached_goal: bool = False,
    reached_horizon: bool = False,
) -> LatticePlan:
    path, path_times = retrieve_path(node)
    terminal_z = node.z.copy()
    terminal_z[2] = wrap_angle(float(terminal_z[2]))
    terminal_z_dot = node.z_dot.copy()
    terminal_z_ddot = node.z_ddot.copy()
    if terminal_stop:
        terminal_z = np.asarray(search_goal, dtype=float).reshape(3)
        terminal_z_dot = np.zeros(3, dtype=float)
        terminal_z_ddot = np.zeros(3, dtype=float)
    first_tau = first_feasible_tau(path, path_times, node, params)
    inPs, ts = minco_seed_from_path(path, path_times, config)
    goal_distance = float(np.linalg.norm(np.asarray(global_goal)[:2] -
                                         terminal_z[:2]))
    diagnostics = _diagnostics(
        config, path[0], np.asarray(global_goal, dtype=float), search_goal,
        terminal_z, goal_distance, selected_depth=node.depth,
        candidate_count=candidate_count, collision_rejects=collision_rejects,
        infeasible_rejects=infeasible_rejects, selected_cost=node.f_cost,
        terminal_stop=terminal_stop, analytic_attempts=analytic_attempts,
        analytic_success=analytic_success, expansions=expansions,
        reached_goal=reached_goal, reached_horizon=reached_horizon)
    return _accepted_plan(
        terminal_z=terminal_z,
        terminal_z_dot=terminal_z_dot,
        terminal_z_ddot=terminal_z_ddot,
        path=path,
        path_times=path_times,
        first_tau=first_tau,
        initial_inPs=inPs,
        initial_ts=ts,
        diagnostics=diagnostics,
    )


def retrieve_path(node: LatticeNode) -> tuple[np.ndarray, np.ndarray]:
    nodes = []
    current = node
    while current is not None:
        nodes.append(current)
        current = current.parent
    nodes.reverse()
    path_segments = []
    time_segments = []
    offset = 0.0
    for index, item in enumerate(nodes):
        segment = item.segment_path
        times = item.segment_times + offset
        if index > 0:
            segment = segment[1:]
            times = times[1:]
        path_segments.append(segment)
        time_segments.append(times)
        offset = float(time_segments[-1][-1])
    return np.vstack(path_segments), np.concatenate(time_segments)


def first_feasible_tau(path, path_times, node: LatticeNode, params: UsvModelParams):
    del path, path_times
    chain = []
    current = node
    while current is not None:
        chain.append(current)
        current = current.parent
    chain.reverse()
    for item in chain:
        if item.parent is None:
            continue
        return theta_tau_nominal(
            item.segment_path[0],
            item.segment_z_dot[0],
            item.segment_z_ddot[0],
            params,
        )
    return theta_tau_nominal(node.z, node.z_dot, node.z_ddot, params)


def minco_seed_from_path(path, path_times, config: UsvLatticeConfig):
    path = np.asarray(path, dtype=float)
    path_times = np.asarray(path_times, dtype=float)
    total_time = max(float(path_times[-1] - path_times[0]), 1e-6)
    selected_depth = max(
        int(round(total_time / max(float(config.primitive_duration), 1e-6))),
        1,
    )
    piece_min = max(int(getattr(config, "minco_piece_count_min", 1)), 1)
    piece_max = max(int(getattr(config, "minco_piece_count_max", piece_min)),
                    piece_min)
    piece_count = min(max(selected_depth, piece_min), piece_max)
    if piece_count <= 1:
        return None, None
    sample_times = np.linspace(path_times[0], path_times[-1], piece_count + 1)
    samples = interpolate_path(path, path_times, sample_times)
    return samples[1:-1].copy(), np.full(piece_count, total_time / piece_count)


def interpolate_path(path, path_times, sample_times):
    path = np.asarray(path, dtype=float)
    path_times = np.asarray(path_times, dtype=float)
    yaws = np.unwrap(path[:, 2])
    out = np.empty((len(sample_times), 3), dtype=float)
    for dim in range(2):
        out[:, dim] = np.interp(sample_times, path_times, path[:, dim])
    out[:, 2] = np.interp(sample_times, path_times, yaws)
    out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
    return out


def reach_goal(z, goal, terminal_stop: bool, config: UsvLatticeConfig) -> bool:
    del terminal_stop
    distance = float(np.linalg.norm(np.asarray(goal)[:2] - np.asarray(z)[:2]))
    yaw_error = abs(wrap_angle(float(goal[2] - z[2])))
    return bool(
        distance <= float(config.goal_tolerance) and
        yaw_error <= float(config.yaw_tolerance))


def goal_inside_local_horizon(start_z, goal, config: UsvLatticeConfig) -> bool:
    start_z = np.asarray(start_z, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    distance = float(np.linalg.norm(goal[:2] - start_z[:2]))
    return bool(distance <= float(config.max_local_goal_distance) + 1e-9)


def path_inside_local_horizon(start_z, path, config: UsvLatticeConfig) -> bool:
    start_z = np.asarray(start_z, dtype=float).reshape(3)
    path = np.asarray(path, dtype=float)
    if path.size == 0:
        return False
    distances = np.linalg.norm(path[:, :2] - start_z[:2], axis=1)
    return bool(np.all(
        distances <= float(config.max_local_goal_distance) + 1e-9))


def reach_horizon(start_z, current_z, config: UsvLatticeConfig) -> bool:
    start_z = np.asarray(start_z, dtype=float).reshape(3)
    current_z = np.asarray(current_z, dtype=float).reshape(3)
    distance = float(np.linalg.norm(current_z[:2] - start_z[:2]))
    return bool(distance >= float(config.max_local_goal_distance))


def reach_horizon_toward_goal(
    start_z,
    current_z,
    goal_z,
    config: UsvLatticeConfig,
) -> bool:
    if not reach_horizon(start_z, current_z, config):
        return False
    start_z = np.asarray(start_z, dtype=float).reshape(3)
    current_z = np.asarray(current_z, dtype=float).reshape(3)
    goal_z = np.asarray(goal_z, dtype=float).reshape(3)
    goal_delta = goal_z[:2] - start_z[:2]
    goal_distance = float(np.linalg.norm(goal_delta))
    if goal_distance <= 1e-9:
        return True
    direction = goal_delta / goal_distance
    offset = current_z[:2] - start_z[:2]
    progress = float(np.dot(offset, direction))
    lateral_error = float(np.linalg.norm(offset - progress * direction))
    horizon = float(config.max_local_goal_distance)
    return bool(progress >= 0.75 * horizon and lateral_error <= 0.55 * horizon)


def grid_key(z, nu, config: UsvLatticeConfig) -> tuple[int, ...]:
    z = np.asarray(z, dtype=float).reshape(3)
    nu = np.asarray(nu, dtype=float).reshape(3)
    return (
        int(math.floor(z[0] / max(float(config.grid_resolution_xy), 1e-6))),
        int(math.floor(z[1] / max(float(config.grid_resolution_xy), 1e-6))),
        int(math.floor(wrap_angle(float(z[2])) /
                       max(float(config.grid_resolution_yaw), 1e-6))),
        int(math.floor(nu[0] / max(float(config.grid_resolution_vxy), 1e-6))),
        int(math.floor(nu[2] /
                       max(float(config.grid_resolution_yaw_rate), 1e-6))),
    )


def collides(path, obstacles, margin: float) -> bool:
    if not obstacles:
        return False
    points = np.asarray(path, dtype=float)
    for obstacle in obstacles:
        center = np.array([obstacle.x, obstacle.y], dtype=float)
        distances = np.linalg.norm(points - center, axis=1)
        if np.any(distances <= obstacle.radius + float(margin)):
            return True
    return False


def _accepted_plan(
    terminal_z,
    terminal_z_dot,
    terminal_z_ddot,
    path,
    path_times,
    first_tau,
    initial_inPs,
    initial_ts,
    diagnostics,
) -> LatticePlan:
    return LatticePlan(
        accepted=True,
        terminal_z=np.asarray(terminal_z, dtype=float).reshape(3),
        terminal_z_dot=np.asarray(terminal_z_dot, dtype=float).reshape(3),
        terminal_z_ddot=np.asarray(terminal_z_ddot, dtype=float).reshape(3),
        path=np.asarray(path, dtype=float),
        path_times=np.asarray(path_times, dtype=float),
        first_tau=np.asarray(first_tau, dtype=float).reshape(3),
        initial_inPs=initial_inPs,
        initial_ts=initial_ts,
        diagnostics=diagnostics,
    )


def _diagnostics(
    config: UsvLatticeConfig,
    start_z,
    global_goal,
    search_goal,
    terminal_z,
    goal_distance: float,
    selected_depth: int,
    candidate_count: int,
    collision_rejects: int,
    infeasible_rejects: int,
    selected_cost: float,
    terminal_stop: bool,
    analytic_attempts: int,
    analytic_success: bool,
    expansions: int,
    reached_goal: bool = False,
    reached_horizon: bool = False,
) -> dict[str, float]:
    start_z = np.asarray(start_z, dtype=float).reshape(3)
    global_goal = np.asarray(global_goal, dtype=float).reshape(3)
    search_goal = np.asarray(search_goal, dtype=float).reshape(3)
    terminal_z = np.asarray(terminal_z, dtype=float).reshape(3)
    goal_position_error = float(np.linalg.norm(global_goal[:2] - terminal_z[:2]))
    goal_yaw_error = abs(wrap_angle(float(global_goal[2] - terminal_z[2])))
    return {
        "frontend_mode": 2.0,
        "frontend_goal_distance": float(goal_distance),
        "frontend_goal_inside_horizon": (
            1.0 if goal_inside_local_horizon(
                start_z, global_goal, config) else 0.0),
        "frontend_goal_position_error": goal_position_error,
        "frontend_goal_yaw_error": goal_yaw_error,
        "frontend_local_goal_error": float(
            np.linalg.norm(search_goal[:2] - terminal_z[:2])),
        "frontend_selected_depth": float(selected_depth),
        "frontend_collision_free": 1.0,
        "frontend_terminal_stop": 1.0 if terminal_stop else 0.0,
        "frontend_candidate_count": float(candidate_count),
        "frontend_collision_reject_count": float(collision_rejects),
        "frontend_infeasible_reject_count": float(infeasible_rejects),
        "frontend_selected_cost": float(selected_cost),
        "frontend_expansion_count": float(expansions),
        "frontend_analytic_attempted": 1.0 if analytic_attempts > 0 else 0.0,
        "frontend_analytic_attempt_count": float(analytic_attempts),
        "frontend_analytic_success": 1.0 if analytic_success else 0.0,
        "frontend_reach_goal": 1.0 if reached_goal else 0.0,
        "frontend_reach_horizon": 1.0 if reached_horizon else 0.0,
        "frontend_tau_v_bar": float(config.tau_v_bar),
        "frontend_heuristic_weight": float(config.heuristic_weight),
        "frontend_control_discretization": float(config.control_discretization),
        "frontend_sample_thrust_limit": float(config.sample_thrust_limit),
        "frontend_check_num": float(config.check_num),
        "frontend_terminal_x": float(terminal_z[0]),
        "frontend_terminal_y": float(terminal_z[1]),
        "frontend_terminal_psi": float(terminal_z[2]),
        "frontend_global_goal_x": float(global_goal[0]),
        "frontend_global_goal_y": float(global_goal[1]),
        "frontend_global_goal_psi": float(global_goal[2]),
        "frontend_local_goal_x": float(search_goal[0]),
        "frontend_local_goal_y": float(search_goal[1]),
        "frontend_local_goal_psi": float(search_goal[2]),
    }


def parse_circular_obstacles(text: str) -> tuple[CircularObstacle, ...]:
    text = (text or "").strip()
    if not text:
        return ()
    obstacles = []
    for chunk in text.split(";"):
        parts = [item.strip() for item in chunk.split(",") if item.strip()]
        if len(parts) != 3:
            raise ValueError(
                "Each obstacle must be formatted as x,y,r and separated by ';'.")
        x, y, radius = (float(item) for item in parts)
        obstacles.append(CircularObstacle(x=x, y=y, radius=radius))
    return tuple(obstacles)
