import math
from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rot2d(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def sqrt_p(error: float, kp: float, accel_max: float) -> float:
    if abs(error) < 1e-9:
        return 0.0
    linear_dist = accel_max / max(kp * kp, 1e-9)
    if abs(error) <= linear_dist:
        return kp * error
    return math.copysign(
        math.sqrt(max(0.0, 2.0 * accel_max *
                      (abs(error) - 0.5 * linear_dist))),
        error)


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


class QuinticHermitePath:
    def __init__(self, coeffs: np.ndarray):
        self.coeffs = coeffs

    @classmethod
    def from_boundary(cls, p0: np.ndarray, psi0: float, r0: float,
                      speed0: float, pf: np.ndarray, psif: float,
                      handle_scale_start: float, handle_scale_goal: float,
                      min_handle: float, u_min_curvature: float,
                      kappa_max: float,
                      start_handle: Optional[float] = None,
                      goal_handle: Optional[float] = None):
        distance = float(np.linalg.norm(pf - p0))
        d0 = (
            max(min_handle, handle_scale_start * distance)
            if start_handle is None else max(min_handle, start_handle)
        )
        df = (
            max(min_handle, handle_scale_goal * distance)
            if goal_handle is None else max(min_handle, goal_handle)
        )

        t0 = np.array([math.cos(psi0), math.sin(psi0)], dtype=float)
        tf = np.array([math.cos(psif), math.sin(psif)], dtype=float)
        n0 = np.array([-math.sin(psi0), math.cos(psi0)], dtype=float)
        nf = np.array([-math.sin(psif), math.cos(psif)], dtype=float)

        kappa0 = clamp(r0 / max(abs(speed0), u_min_curvature),
                       -kappa_max, kappa_max)
        kappaf = 0.0

        p_s0 = d0 * t0
        p_sf = df * tf
        p_ss0 = d0 * d0 * kappa0 * n0
        p_ssf = df * df * kappaf * nf

        coeffs = np.zeros((2, 6), dtype=float)
        coeffs[:, 0] = p0
        coeffs[:, 1] = p_s0
        coeffs[:, 2] = 0.5 * p_ss0

        lhs = np.array([
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [6.0, 12.0, 20.0],
        ])
        rhs = np.vstack([
            pf - coeffs[:, 0] - coeffs[:, 1] - coeffs[:, 2],
            p_sf - coeffs[:, 1] - 2.0 * coeffs[:, 2],
            p_ssf - 2.0 * coeffs[:, 2],
        ])
        solved = np.linalg.solve(lhs, rhs)
        coeffs[:, 3] = solved[0, :]
        coeffs[:, 4] = solved[1, :]
        coeffs[:, 5] = solved[2, :]
        return cls(coeffs)

    def evaluate(self, s: float):
        s = clamp(s, 0.0, 1.0)
        powers = np.array([1.0, s, s**2, s**3, s**4, s**5])
        dpowers = np.array([0.0, 1.0, 2.0 * s, 3.0 * s**2,
                            4.0 * s**3, 5.0 * s**4])
        ddpowers = np.array([0.0, 0.0, 2.0, 6.0 * s,
                             12.0 * s**2, 20.0 * s**3])
        p = self.coeffs @ powers
        p_s = self.coeffs @ dpowers
        p_ss = self.coeffs @ ddpowers
        speed_s = float(np.linalg.norm(p_s))
        psi = math.atan2(p_s[1], p_s[0])
        cross = float(p_s[0] * p_ss[1] - p_s[1] * p_ss[0])
        kappa = cross / max(speed_s**3, 1e-6)
        return p, p_s, p_ss, psi, kappa


class MultiSegmentPath:
    def __init__(self, segments: List[QuinticHermitePath]):
        self.segments = segments
        self.max_s = float(max(len(segments), 1))
        self.max_abs_kappa = 0.0
        self.smoothing_attempts = 0
        self.curvature_feasible = True

    def evaluate(self, path_s: float):
        if not self.segments:
            raise RuntimeError('Cannot evaluate an empty path.')
        path_s = clamp(path_s, 0.0, self.max_s)
        if path_s >= self.max_s:
            segment_index = len(self.segments) - 1
            local_s = 1.0
        else:
            segment_index = int(math.floor(path_s))
            local_s = path_s - segment_index
        return self.segments[segment_index].evaluate(local_s)


class PathTable:
    def __init__(self, path: MultiSegmentPath, samples: int):
        self.path = path
        self.s = np.linspace(0.0, path.max_s, max(samples, 20))
        evaluated = [path.evaluate(float(si)) for si in self.s]
        points = [item[0] for item in evaluated]
        self.kappas = np.asarray([item[4] for item in evaluated])
        self.max_abs_kappa = float(np.max(np.abs(self.kappas)))
        self.points = np.asarray(points)
        diffs = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        self.arc = np.concatenate(([0.0], np.cumsum(lengths)))
        self.total_length = float(self.arc[-1])

    def closest_index(self, position: np.ndarray) -> int:
        distances = np.linalg.norm(self.points - position, axis=1)
        return int(np.argmin(distances))

    def s_at_arc(self, arc_value: float) -> float:
        if self.total_length <= 1e-6:
            return 1.0
        arc_value = clamp(arc_value, 0.0, self.total_length)
        index = int(np.searchsorted(self.arc, arc_value))
        if index <= 0:
            return float(self.s[0])
        if index >= len(self.arc):
            return float(self.s[-1])
        denom = self.arc[index] - self.arc[index - 1]
        if denom <= 1e-9:
            return float(self.s[index])
        ratio = (arc_value - self.arc[index - 1]) / denom
        return float(self.s[index - 1] +
                     ratio * (self.s[index] - self.s[index - 1]))


class CascadedPidController(Node):
    """Path-guided surge/yaw-rate controller for the two-thruster WAM-V.

    It consumes EKF odometry and publishes direct left/right thrust commands.
    It does not consume Gazebo ground truth.
    """

    def __init__(self):
        super().__init__('cascaded_pid_controller')

        self.declare_parameter('odom_topic', '/safe_docking/odometry')
        self.declare_parameter(
            'left_thrust_topic', '/wamv/thrusters/left/thrust')
        self.declare_parameter(
            'right_thrust_topic', '/wamv/thrusters/right/thrust')
        self.declare_parameter(
            'command_topic', '/safe_docking/controller/command')
        self.declare_parameter(
            'debug_topic', '/safe_docking/controller/debug')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('command_timeout_sec', 0.5)

        self.declare_parameter('target_x', 6.0)
        self.declare_parameter('target_y', 12.0)
        self.declare_parameter('target_yaw', 1.0)
        self.declare_parameter('waypoint_xs', [6.0])
        self.declare_parameter('waypoint_ys', [12.0])
        self.declare_parameter('waypoint_yaws', [1.0])
        self.declare_parameter('position_tolerance', 0.4)
        self.declare_parameter('yaw_tolerance', 0.15)
        self.declare_parameter('waypoint_tolerance', 0.75)

        self.declare_parameter('min_path_handle', 1.5)
        self.declare_parameter('handle_scale_start', 0.35)
        self.declare_parameter('handle_scale_goal', 0.35)
        self.declare_parameter('u_min_curvature', 0.2)
        self.declare_parameter('kappa_max', 0.7)
        self.declare_parameter('curvature_check_samples', 240)
        self.declare_parameter('max_path_smoothing_iterations', 6)
        self.declare_parameter('path_handle_growth', 1.45)
        self.declare_parameter('max_path_handle_scale', 8.0)
        self.declare_parameter('kappa_feasibility_tolerance', 1.05)
        self.declare_parameter('num_path_samples', 180)
        self.declare_parameter('lookahead_base', 1.2)
        self.declare_parameter('lookahead_speed_gain', 1.4)
        self.declare_parameter('lookahead_min', 0.8)
        self.declare_parameter('lookahead_max', 3.0)
        self.declare_parameter('goal_lookahead_distance', 1.2)
        self.declare_parameter('replan_cross_track_threshold', 3.0)
        self.declare_parameter('replan_interval_sec', 30.0)

        self.declare_parameter('along_track_kp', 0.35)
        self.declare_parameter('along_track_accel_max', 0.08)
        self.declare_parameter('cross_track_kp', 0.45)
        self.declare_parameter('cross_track_accel_max', 0.10)
        self.declare_parameter('max_cross_track_speed', 0.45)
        self.declare_parameter('max_course_correction', 0.7)
        self.declare_parameter('final_heading_blend_distance', 2.0)
        self.declare_parameter('u_min_guidance', 0.2)
        self.declare_parameter('enable_sideslip_compensation', False)
        self.declare_parameter('u_beta_min', 0.25)
        self.declare_parameter('beta_max', 0.6)

        self.declare_parameter('cruise_speed', 0.65)
        self.declare_parameter('min_speed', -0.25)
        self.declare_parameter('max_speed', 0.75)
        self.declare_parameter('max_lateral_accel', 0.12)
        self.declare_parameter('max_stopping_accel', 0.08)
        self.declare_parameter('kappa_eps', 1e-3)

        self.declare_parameter('heading_kp', 1.2)
        self.declare_parameter('yaw_accel_max', 0.35)
        self.declare_parameter('max_yaw_rate_ff', 0.45)
        self.declare_parameter('max_yaw_rate', 0.8)

        self.declare_parameter('surge_kp', 220.0)
        self.declare_parameter('surge_ki', 12.0)
        self.declare_parameter('surge_drag_linear', 70.0)
        self.declare_parameter('surge_drag_quadratic', 140.0)
        self.declare_parameter('max_surge_force', 650.0)
        self.declare_parameter('surge_integral_limit', 10.0)

        self.declare_parameter('yaw_rate_kp', 900.0)
        self.declare_parameter('yaw_rate_ki', 25.0)
        self.declare_parameter('yaw_rate_kd', 0.0)
        self.declare_parameter('max_yaw_moment', 1300.0)
        self.declare_parameter('yaw_rate_integral_limit', 2.5)

        self.declare_parameter('thruster_half_spacing', 1.027135)
        self.declare_parameter('min_thrust', -500.0)
        self.declare_parameter('max_thrust', 500.0)
        self.declare_parameter('thrust_rate_limit', 1000.0)
        self.declare_parameter('allocation_weight_force', 1.0)
        self.declare_parameter('allocation_weight_torque', 3.0)
        self.declare_parameter('allocation_smooth_weight', 0.03)
        self.declare_parameter('surge_aw_threshold', 80.0)
        self.declare_parameter('yaw_aw_threshold', 180.0)

        self.declare_parameter('stop_when_reached', False)

        self.odom_topic = self._str_param('odom_topic')
        self.command_topic = self._str_param('command_topic')
        self.debug_topic = self._str_param('debug_topic')
        self.control_period = 1.0 / max(self._float_param('control_rate_hz'), 1.0)
        self.command_timeout_sec = self._float_param('command_timeout_sec')

        self.waypoints = self.load_waypoints()
        self.position_tolerance = self._float_param('position_tolerance')
        self.yaw_tolerance = self._float_param('yaw_tolerance')
        self.waypoint_tolerance = self._float_param('waypoint_tolerance')

        self.min_path_handle = self._float_param('min_path_handle')
        self.handle_scale_start = self._float_param('handle_scale_start')
        self.handle_scale_goal = self._float_param('handle_scale_goal')
        self.u_min_curvature = self._float_param('u_min_curvature')
        self.kappa_max = self._float_param('kappa_max')
        self.curvature_check_samples = self._int_param(
            'curvature_check_samples')
        self.max_path_smoothing_iterations = self._int_param(
            'max_path_smoothing_iterations')
        self.path_handle_growth = self._float_param('path_handle_growth')
        self.max_path_handle_scale = self._float_param('max_path_handle_scale')
        self.kappa_feasibility_tolerance = self._float_param(
            'kappa_feasibility_tolerance')
        self.num_path_samples = self._int_param('num_path_samples')
        self.lookahead_base = self._float_param('lookahead_base')
        self.lookahead_speed_gain = self._float_param('lookahead_speed_gain')
        self.lookahead_min = self._float_param('lookahead_min')
        self.lookahead_max = self._float_param('lookahead_max')
        self.goal_lookahead_distance = self._float_param(
            'goal_lookahead_distance')
        self.replan_cross_track_threshold = self._float_param(
            'replan_cross_track_threshold')
        self.replan_interval_sec = self._float_param('replan_interval_sec')

        self.along_track_kp = self._float_param('along_track_kp')
        self.along_track_accel_max = self._float_param('along_track_accel_max')
        self.cross_track_kp = self._float_param('cross_track_kp')
        self.cross_track_accel_max = self._float_param('cross_track_accel_max')
        self.max_cross_track_speed = self._float_param('max_cross_track_speed')
        self.max_course_correction = self._float_param('max_course_correction')
        self.final_heading_blend_distance = self._float_param(
            'final_heading_blend_distance')
        self.u_min_guidance = self._float_param('u_min_guidance')
        self.enable_sideslip_compensation = self.get_parameter(
            'enable_sideslip_compensation').get_parameter_value().bool_value
        self.u_beta_min = self._float_param('u_beta_min')
        self.beta_max = self._float_param('beta_max')

        self.cruise_speed = self._float_param('cruise_speed')
        self.min_speed = self._float_param('min_speed')
        self.max_speed = self._float_param('max_speed')
        self.max_lateral_accel = self._float_param('max_lateral_accel')
        self.max_stopping_accel = self._float_param('max_stopping_accel')
        self.kappa_eps = self._float_param('kappa_eps')

        self.heading_kp = self._float_param('heading_kp')
        self.yaw_accel_max = self._float_param('yaw_accel_max')
        self.max_yaw_rate_ff = self._float_param('max_yaw_rate_ff')
        self.max_yaw_rate = self._float_param('max_yaw_rate')

        self.surge_kp = self._float_param('surge_kp')
        self.surge_ki = self._float_param('surge_ki')
        self.surge_drag_linear = self._float_param('surge_drag_linear')
        self.surge_drag_quadratic = self._float_param('surge_drag_quadratic')
        self.max_surge_force = self._float_param('max_surge_force')
        self.surge_integral_limit = self._float_param('surge_integral_limit')

        self.yaw_rate_kp = self._float_param('yaw_rate_kp')
        self.yaw_rate_ki = self._float_param('yaw_rate_ki')
        self.yaw_rate_kd = self._float_param('yaw_rate_kd')
        self.max_yaw_moment = self._float_param('max_yaw_moment')
        self.yaw_rate_integral_limit = self._float_param(
            'yaw_rate_integral_limit')

        self.thruster_half_spacing = self._float_param('thruster_half_spacing')
        self.min_thrust = self._float_param('min_thrust')
        self.max_thrust = self._float_param('max_thrust')
        self.thrust_rate_limit = self._float_param('thrust_rate_limit')
        self.allocation_weight_force = self._float_param(
            'allocation_weight_force')
        self.allocation_weight_torque = self._float_param(
            'allocation_weight_torque')
        self.allocation_smooth_weight = self._float_param(
            'allocation_smooth_weight')
        self.surge_aw_threshold = self._float_param('surge_aw_threshold')
        self.yaw_aw_threshold = self._float_param('yaw_aw_threshold')
        self.stop_when_reached = self.get_parameter(
            'stop_when_reached').get_parameter_value().bool_value

        self.latest_odom: Optional[Odometry] = None
        self.last_control_time: Optional[float] = None
        self.path_created_time: Optional[float] = None
        self.path: Optional[MultiSegmentPath] = None
        self.path_table: Optional[PathTable] = None
        self.path_max_abs_kappa = 0.0
        self.path_smoothing_attempts = 0
        self.path_curvature_feasible = True
        self.current_path_segment = 0
        self.last_reference_arc = 0.0
        self.goal_reached = False
        self.surge_integral = 0.0
        self.yaw_rate_integral = 0.0
        self.last_yaw_rate_error = 0.0
        self.prev_left_thrust = 0.0
        self.prev_right_thrust = 0.0

        self.left_pub = self.create_publisher(
            Float64, self._str_param('left_thrust_topic'), 10)
        self.right_pub = self.create_publisher(
            Float64, self._str_param('right_thrust_topic'), 10)
        self.command_pub = self.create_publisher(
            Float64MultiArray, self.command_topic, 10)
        self.debug_pub = self.create_publisher(
            Float64MultiArray, self.debug_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.create_timer(self.control_period, self.on_timer)

        goal = self.waypoints[-1]
        self.get_logger().info(
            'Path-guided PID controller loaded with '
            f'{len(self.waypoints)} waypoint(s); final target '
            f'x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.yaw:.2f}.')

    def _str_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _float_param(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _int_param(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _float_array_param(self, name: str) -> List[float]:
        return list(self.get_parameter(name).get_parameter_value().double_array_value)

    def load_waypoints(self) -> List[Waypoint]:
        xs = self._float_array_param('waypoint_xs')
        ys = self._float_array_param('waypoint_ys')
        yaws = self._float_array_param('waypoint_yaws')
        if not xs or len(xs) != len(ys) or len(xs) != len(yaws):
            self.get_logger().warn(
                'Invalid waypoint arrays; falling back to target_x/y/yaw.')
            return [Waypoint(
                self._float_param('target_x'),
                self._float_param('target_y'),
                self._float_param('target_yaw'))]
        return [
            Waypoint(float(x), float(y), float(yaw))
            for x, y, yaw in zip(xs, ys, yaws)
        ]

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def on_timer(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.latest_odom is None:
            self.publish_thrust(0.0, 0.0)
            return

        stamp = stamp_to_sec(self.latest_odom.header.stamp)
        if stamp > 0.0 and abs(now - stamp) > self.command_timeout_sec:
            self.publish_thrust(0.0, 0.0)
            self.reset_integrators()
            return

        if self.last_control_time is None:
            self.last_control_time = now
            return

        dt = now - self.last_control_time
        if dt <= 0.0:
            return
        dt = min(dt, 0.2)
        self.last_control_time = now

        state = self.extract_state(self.latest_odom)
        cmd = self.compute_command(state, dt, now)
        self.publish_thrust(cmd['left_thrust'], cmd['right_thrust'])
        self.publish_debug(state, cmd)

    def extract_state(self, odom: Odometry):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        psi = yaw_from_quaternion(odom.pose.pose.orientation)
        x_dot = odom.twist.twist.linear.x
        y_dot = odom.twist.twist.linear.y
        psi_dot = odom.twist.twist.angular.z
        return {
            'position': np.array([x, y], dtype=float),
            'velocity': np.array([x_dot, y_dot], dtype=float),
            'yaw': psi,
            'yaw_rate': psi_dot,
        }

    def compute_command(self, state, dt: float, now: float):
        final_goal = self.waypoints[-1]
        final_distance = float(
            np.linalg.norm(final_goal.position - state['position']))
        final_yaw_error = wrap_angle(final_goal.yaw - state['yaw'])

        if (self.stop_when_reached and
                final_distance < self.position_tolerance and
                abs(final_yaw_error) < self.yaw_tolerance and
                np.linalg.norm(state['velocity']) < 0.15):
            self.goal_reached = True

        if self.goal_reached:
            self.reset_integrators()
            return self.zero_command(state, final_goal, final_distance,
                                     final_yaw_error)

        if self.should_replan(state, now):
            self.make_path(state, now)

        guidance = self.path_guidance(state, final_goal)
        body_velocity = rot2d(state['yaw']).T @ state['velocity']
        surge_speed = float(body_velocity[0])

        surge_error = guidance['u_ref'] - surge_speed
        surge_ff = (
            self.surge_drag_linear * guidance['u_ref'] +
            self.surge_drag_quadratic * abs(guidance['u_ref']) *
            guidance['u_ref'])
        surge_force_d = (
            surge_ff +
            self.surge_kp * surge_error +
            self.surge_ki * self.surge_integral)
        surge_force_d = clamp(
            surge_force_d, -self.max_surge_force, self.max_surge_force)

        yaw_rate_error = guidance['r_ref'] - state['yaw_rate']
        yaw_rate_derivative = (
            yaw_rate_error - self.last_yaw_rate_error) / max(dt, 1e-6)
        self.last_yaw_rate_error = yaw_rate_error
        yaw_moment_d = (
            self.yaw_rate_kp * yaw_rate_error +
            self.yaw_rate_ki * self.yaw_rate_integral +
            self.yaw_rate_kd * yaw_rate_derivative)
        yaw_moment_d = clamp(
            yaw_moment_d, -self.max_yaw_moment, self.max_yaw_moment)

        left, right, alloc = self.allocate_thrusters(
            surge_force_d, yaw_moment_d, dt)
        surge_force_c = left + right
        tau_c = self.thruster_half_spacing * (right - left)

        if abs(surge_force_d - surge_force_c) < self.surge_aw_threshold:
            self.surge_integral = clamp(
                self.surge_integral + surge_error * dt,
                -self.surge_integral_limit,
                self.surge_integral_limit)
        if abs(yaw_moment_d - tau_c) < self.yaw_aw_threshold:
            self.yaw_rate_integral = clamp(
                self.yaw_rate_integral + yaw_rate_error * dt,
                -self.yaw_rate_integral_limit,
                self.yaw_rate_integral_limit)

        self.prev_left_thrust = left
        self.prev_right_thrust = right

        return {
            'left_thrust': left,
            'right_thrust': right,
            'force': surge_force_d,
            'torque': yaw_moment_d,
            'surge_force_c': surge_force_c,
            'tau_c': tau_c,
            'desired_yaw': guidance['psi_cmd'],
            'desired_yaw_rate': guidance['r_ref'],
            'distance': final_distance,
            'yaw_error': final_yaw_error,
            'velocity_error_norm': abs(surge_error),
            'waypoint_index': self.current_path_segment,
            's_ref': guidance['s_ref'],
            'x_ref': guidance['p_ref'][0],
            'y_ref': guidance['p_ref'][1],
            'psi_ref': guidance['psi_ref'],
            'kappa_ref': guidance['kappa_ref'],
            'e_s': guidance['e_s'],
            'e_y': guidance['e_y'],
            'u_ref': guidance['u_ref'],
            'surge_speed': surge_speed,
            'beta_hat': guidance['beta_hat'],
            'allocation_cost': alloc['cost'],
            'path_max_abs_kappa': self.path_max_abs_kappa,
            'path_smoothing_attempts': self.path_smoothing_attempts,
            'path_curvature_feasible': float(self.path_curvature_feasible),
        }

    def zero_command(self, state, goal: Waypoint, distance: float,
                     yaw_error: float):
        return {
            'left_thrust': 0.0,
            'right_thrust': 0.0,
            'force': 0.0,
            'torque': 0.0,
            'surge_force_c': 0.0,
            'tau_c': 0.0,
            'desired_yaw': goal.yaw,
            'desired_yaw_rate': 0.0,
            'distance': distance,
            'yaw_error': yaw_error,
            'velocity_error_norm': float(np.linalg.norm(state['velocity'])),
            'waypoint_index': self.current_path_segment,
            's_ref': 1.0,
            'x_ref': goal.x,
            'y_ref': goal.y,
            'psi_ref': goal.yaw,
            'kappa_ref': 0.0,
            'e_s': 0.0,
            'e_y': 0.0,
            'u_ref': 0.0,
            'surge_speed': 0.0,
            'beta_hat': 0.0,
            'allocation_cost': 0.0,
            'path_max_abs_kappa': self.path_max_abs_kappa,
            'path_smoothing_attempts': self.path_smoothing_attempts,
            'path_curvature_feasible': float(self.path_curvature_feasible),
        }

    def should_replan(self, state, now: float) -> bool:
        if self.path is None or self.path_table is None:
            return True
        if (self.replan_interval_sec > 0.0 and
                self.path_created_time is not None and
                now - self.path_created_time > self.replan_interval_sec):
            return True
        closest = self.path_table.closest_index(state['position'])
        cross_track = float(
            np.linalg.norm(self.path_table.points[closest] - state['position']))
        if cross_track > self.replan_cross_track_threshold:
            return True
        return False

    def make_path(self, state, now: float):
        body_velocity = rot2d(state['yaw']).T @ state['velocity']
        speed0 = float(body_velocity[0])
        path_points = [state['position']] + [wp.position for wp in self.waypoints]
        path_yaws = [state['yaw']] + [wp.yaw for wp in self.waypoints]
        base_handle_lengths = self.path_handle_lengths(path_points)
        best_path = None
        best_table = None
        best_scale = 1.0
        best_attempt = 0
        best_max_kappa = float('inf')
        feasible = False
        scale = 1.0
        max_scale = max(self.max_path_handle_scale, 1.0)
        max_attempts = max(self.max_path_smoothing_iterations, 0)

        for attempt in range(max_attempts + 1):
            handle_lengths = [
                max(self.min_path_handle, h * min(scale, max_scale))
                for h in base_handle_lengths
            ]
            candidate_path = self.build_path_from_handles(
                path_points, path_yaws, handle_lengths, state['yaw_rate'],
                speed0)
            candidate_table = PathTable(
                candidate_path, self.curvature_check_samples)
            max_kappa = candidate_table.max_abs_kappa
            if max_kappa < best_max_kappa:
                best_path = candidate_path
                best_table = candidate_table
                best_scale = scale
                best_attempt = attempt
                best_max_kappa = max_kappa
            if max_kappa <= self.kappa_max * self.kappa_feasibility_tolerance:
                feasible = True
                break
            scale *= max(self.path_handle_growth, 1.0)
            if scale > max_scale and attempt < max_attempts:
                scale = max_scale

        assert best_path is not None
        assert best_table is not None
        best_path.max_abs_kappa = best_max_kappa
        best_path.smoothing_attempts = best_attempt
        best_path.curvature_feasible = feasible
        self.path = best_path
        self.path_table = PathTable(self.path, self.num_path_samples)
        self.path_max_abs_kappa = self.path_table.max_abs_kappa
        self.path_smoothing_attempts = best_attempt
        self.path_curvature_feasible = feasible

        if best_attempt > 0 or not feasible:
            status = 'feasible' if feasible else 'still infeasible'
            self.get_logger().warn(
                'Path curvature check: '
                f'max |kappa|={self.path_max_abs_kappa:.3f} 1/m, '
                f'limit={self.kappa_max:.3f} 1/m, '
                f'handle scale={best_scale:.2f}, '
                f'attempts={best_attempt}, status={status}.')

        self.path_created_time = now
        self.current_path_segment = 0
        self.last_reference_arc = 0.0

    def build_path_from_handles(self, path_points: List[np.ndarray],
                                path_yaws: List[float],
                                handle_lengths: List[float],
                                initial_yaw_rate: float,
                                initial_speed: float) -> MultiSegmentPath:
        segments = []
        for index in range(len(path_points) - 1):
            r0 = initial_yaw_rate if index == 0 else 0.0
            u0 = initial_speed if index == 0 else self.cruise_speed
            segments.append(QuinticHermitePath.from_boundary(
                path_points[index],
                path_yaws[index],
                r0,
                u0,
                path_points[index + 1],
                path_yaws[index + 1],
                self.handle_scale_start,
                self.handle_scale_goal,
                self.min_path_handle,
                self.u_min_curvature,
                self.kappa_max,
                start_handle=handle_lengths[index],
                goal_handle=handle_lengths[index + 1]))
        return MultiSegmentPath(segments)

    def path_handle_lengths(self, points: List[np.ndarray]) -> List[float]:
        segment_lengths = [
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(len(points) - 1)
        ]
        handles = []
        for index in range(len(points)):
            if index == 0:
                length = self.handle_scale_start * segment_lengths[0]
            elif index == len(points) - 1:
                length = self.handle_scale_goal * segment_lengths[-1]
            else:
                length = 0.5 * self.handle_scale_goal * (
                    segment_lengths[index - 1] + segment_lengths[index])
            handles.append(max(self.min_path_handle, length))
        return handles

    def path_guidance(self, state, goal: Waypoint):
        assert self.path is not None
        assert self.path_table is not None

        position = state['position']
        velocity = state['velocity']
        speed_world = float(np.linalg.norm(velocity))
        distance_to_goal = float(np.linalg.norm(goal.position - position))

        closest = self.path_table.closest_index(position)
        lookahead = clamp(
            self.lookahead_base + self.lookahead_speed_gain * speed_world,
            self.lookahead_min,
            self.lookahead_max)
        if distance_to_goal < self.goal_lookahead_distance:
            lookahead = min(lookahead, max(0.15, distance_to_goal))
        target_arc = self.path_table.arc[closest] + lookahead
        target_arc = max(target_arc, self.last_reference_arc)
        self.last_reference_arc = min(target_arc, self.path_table.total_length)
        s_ref = self.path_table.s_at_arc(self.last_reference_arc)
        self.current_path_segment = min(
            int(math.floor(min(s_ref, self.path.max_s - 1e-6))),
            len(self.waypoints) - 1)

        p_ref, p_s, _p_ss, psi_ref, kappa_ref = self.path.evaluate(s_ref)
        tangent_norm = max(float(np.linalg.norm(p_s)), 1e-6)
        t_ref = p_s / tangent_norm
        n_ref = np.array([-t_ref[1], t_ref[0]], dtype=float)

        e_p = p_ref - position
        e_s = float(t_ref @ e_p)
        e_y = float(n_ref @ e_p)

        u_path = self.speed_profile(kappa_ref, distance_to_goal)
        u_corr = sqrt_p(e_s, self.along_track_kp,
                        self.along_track_accel_max)
        u_ref = clamp(u_path + u_corr, self.min_speed, self.max_speed)

        v_cross_cmd = sqrt_p(
            e_y, self.cross_track_kp, self.cross_track_accel_max)
        v_cross_cmd = clamp(
            v_cross_cmd, -self.max_cross_track_speed,
            self.max_cross_track_speed)
        chi_correction = math.atan2(
            v_cross_cmd, max(abs(u_ref), self.u_min_guidance))
        chi_correction = clamp(
            chi_correction, -self.max_course_correction,
            self.max_course_correction)
        final_heading_weight = clamp(
            distance_to_goal / max(self.final_heading_blend_distance, 1e-6),
            0.0,
            1.0)
        chi_correction *= final_heading_weight
        chi_cmd = wrap_angle(psi_ref + chi_correction)

        beta_hat = 0.0
        if self.enable_sideslip_compensation and speed_world > self.u_beta_min:
            chi_meas = math.atan2(velocity[1], velocity[0])
            beta_hat = clamp(wrap_angle(chi_meas - state['yaw']),
                             -self.beta_max, self.beta_max)
        psi_cmd = wrap_angle(chi_cmd - beta_hat)

        r_ff = clamp(
            u_ref * kappa_ref, -self.max_yaw_rate_ff, self.max_yaw_rate_ff)
        e_psi = wrap_angle(psi_cmd - state['yaw'])
        r_fb = sqrt_p(e_psi, self.heading_kp, self.yaw_accel_max)
        r_ref = clamp(r_ff + r_fb, -self.max_yaw_rate, self.max_yaw_rate)

        return {
            's_ref': s_ref,
            'p_ref': p_ref,
            'psi_ref': psi_ref,
            'kappa_ref': kappa_ref,
            'e_s': e_s,
            'e_y': e_y,
            'u_ref': u_ref,
            'psi_cmd': psi_cmd,
            'r_ref': r_ref,
            'beta_hat': beta_hat,
        }

    def speed_profile(self, kappa_ref: float, distance_to_goal: float) -> float:
        u_kappa_max = math.sqrt(
            self.max_lateral_accel / max(abs(kappa_ref), self.kappa_eps))
        u_stop_max = math.sqrt(
            max(0.0, 2.0 * self.max_stopping_accel *
                max(distance_to_goal, 0.0)))
        u_path = min(self.cruise_speed, u_kappa_max, u_stop_max,
                     self.max_speed)
        if distance_to_goal < self.position_tolerance:
            u_path = 0.0
        return clamp(u_path, self.min_speed, self.max_speed)

    def allocate_thrusters(self, surge_force: float, yaw_moment: float,
                           dt: float):
        b = self.thruster_half_spacing
        if b <= 1e-6:
            half = 0.5 * clamp(surge_force, 2.0 * self.min_thrust,
                               2.0 * self.max_thrust)
            return half, half, {'cost': 0.0}

        rate_step = max(self.thrust_rate_limit, 0.0) * max(dt, 1e-3)
        lower = np.array([
            max(self.min_thrust, self.prev_left_thrust - rate_step),
            max(self.min_thrust, self.prev_right_thrust - rate_step),
        ])
        upper = np.array([
            min(self.max_thrust, self.prev_left_thrust + rate_step),
            min(self.max_thrust, self.prev_right_thrust + rate_step),
        ])

        sqrt_force = math.sqrt(max(self.allocation_weight_force, 0.0))
        sqrt_torque = math.sqrt(max(self.allocation_weight_torque, 0.0))
        sqrt_smooth = math.sqrt(max(self.allocation_smooth_weight, 0.0))
        a_aug = np.array([
            [sqrt_force, sqrt_force],
            [-sqrt_torque * b, sqrt_torque * b],
            [sqrt_smooth, 0.0],
            [0.0, sqrt_smooth],
        ], dtype=float)
        b_aug = np.array([
            sqrt_force * surge_force,
            sqrt_torque * yaw_moment,
            sqrt_smooth * self.prev_left_thrust,
            sqrt_smooth * self.prev_right_thrust,
        ], dtype=float)

        candidates = []
        unconstrained, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
        candidates.append(unconstrained)

        for fixed_index in (0, 1):
            for fixed_value in (lower[fixed_index], upper[fixed_index]):
                candidates.append(self.solve_with_one_fixed(
                    a_aug, b_aug, fixed_index, fixed_value))
        for left in (lower[0], upper[0]):
            for right in (lower[1], upper[1]):
                candidates.append(np.array([left, right], dtype=float))

        best = None
        best_cost = float('inf')
        for candidate in candidates:
            feasible = np.clip(candidate, lower, upper)
            residual = a_aug @ feasible - b_aug
            cost = float(residual @ residual)
            if cost < best_cost:
                best_cost = cost
                best = feasible

        return float(best[0]), float(best[1]), {'cost': best_cost}

    def solve_with_one_fixed(self, a_aug: np.ndarray, b_aug: np.ndarray,
                             fixed_index: int,
                             fixed_value: float) -> np.ndarray:
        free_index = 1 - fixed_index
        a_free = a_aug[:, free_index]
        rhs = b_aug - a_aug[:, fixed_index] * fixed_value
        denom = float(a_free @ a_free)
        free_value = 0.0 if denom <= 1e-9 else float((a_free @ rhs) / denom)
        candidate = np.zeros(2, dtype=float)
        candidate[fixed_index] = fixed_value
        candidate[free_index] = free_value
        return candidate

    def publish_thrust(self, left: float, right: float):
        left_msg = Float64()
        right_msg = Float64()
        left_msg.data = float(left)
        right_msg.data = float(right)
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

        command = Float64MultiArray()
        command.data = [float(left), float(right)]
        self.command_pub.publish(command)

    def publish_debug(self, state, cmd):
        final_goal = self.waypoints[-1]
        debug = Float64MultiArray()
        debug.data = [
            float(state['position'][0]),
            float(state['position'][1]),
            float(state['yaw']),
            float(state['velocity'][0]),
            float(state['velocity'][1]),
            float(state['yaw_rate']),
            float(final_goal.x),
            float(final_goal.y),
            float(final_goal.yaw),
            float(cmd['desired_yaw']),
            float(cmd['desired_yaw_rate']),
            float(cmd['force']),
            float(cmd['torque']),
            float(cmd['left_thrust']),
            float(cmd['right_thrust']),
            float(cmd['distance']),
            float(cmd['yaw_error']),
            float(cmd['velocity_error_norm']),
            float(cmd['waypoint_index']),
            float(cmd['s_ref']),
            float(cmd['x_ref']),
            float(cmd['y_ref']),
            float(cmd['psi_ref']),
            float(cmd['kappa_ref']),
            float(cmd['e_s']),
            float(cmd['e_y']),
            float(cmd['u_ref']),
            float(cmd['surge_speed']),
            float(cmd['surge_force_c']),
            float(cmd['tau_c']),
            float(cmd['beta_hat']),
            float(cmd['allocation_cost']),
            float(cmd['path_max_abs_kappa']),
            float(cmd['path_smoothing_attempts']),
            float(cmd['path_curvature_feasible']),
        ]
        self.debug_pub.publish(debug)

    def reset_integrators(self):
        self.surge_integral = 0.0
        self.yaw_rate_integral = 0.0
        self.last_yaw_rate_error = 0.0

    def destroy_node(self):
        self.publish_thrust(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CascadedPidController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
