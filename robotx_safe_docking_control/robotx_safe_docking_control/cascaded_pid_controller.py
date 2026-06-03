import math
import warnings
from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

try:
    from scipy.interpolate import BSpline as scipy_bspline
    from scipy.optimize import minimize as scipy_minimize
except Exception:  # pragma: no cover - optional runtime dependency fallback
    scipy_bspline = None
    scipy_minimize = None


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


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: float
    yaw_specified: bool = True

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


class BSplinePath:
    def __init__(self, control_points: np.ndarray, knots: np.ndarray,
                 degree: int, max_s: float):
        if scipy_bspline is None:
            raise RuntimeError('SciPy BSpline is unavailable.')
        self.control_points = np.asarray(control_points, dtype=float)
        self.knots = np.asarray(knots, dtype=float)
        self.degree = int(degree)
        self.max_s = float(max(max_s, 1e-6))
        self.max_abs_kappa = 0.0
        self.smoothing_attempts = 0
        self.curvature_feasible = True
        self.spline = scipy_bspline(
            self.knots, self.control_points, self.degree, axis=0)
        self.spline_d1 = self.spline.derivative(1)
        self.spline_d2 = self.spline.derivative(2)

    def evaluate(self, path_s: float):
        path_s = clamp(path_s, 0.0, self.max_s)
        p = np.asarray(self.spline(path_s), dtype=float)
        p_s = np.asarray(self.spline_d1(path_s), dtype=float)
        p_ss = np.asarray(self.spline_d2(path_s), dtype=float)
        speed_s = float(np.linalg.norm(p_s))
        psi = math.atan2(p_s[1], p_s[0]) if speed_s > 1e-9 else 0.0
        cross = float(p_s[0] * p_ss[1] - p_s[1] * p_ss[0])
        kappa = cross / max(speed_s**3, 1e-6)
        return p, p_s, p_ss, psi, kappa


class PathTable:
    def __init__(self, path, samples: int):
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


class TrajectoryTable:
    def __init__(self, path, samples: int, initial_surge_speed: float,
                 cruise_speed: float, max_speed: float,
                 max_lateral_accel: float, max_yaw_rate: float,
                 accel_max: float, decel_max: float, kappa_eps: float):
        self.path = path
        self.s = np.linspace(0.0, path.max_s, max(samples, 30))
        evaluated = [path.evaluate(float(si)) for si in self.s]
        self.points = np.asarray([item[0] for item in evaluated], dtype=float)
        self.psi = np.asarray([item[3] for item in evaluated], dtype=float)
        self.psi_unwrapped = np.unwrap(self.psi)
        self.kappa = np.asarray([item[4] for item in evaluated], dtype=float)

        diffs = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        self.arc = np.concatenate(([0.0], np.cumsum(lengths)))
        self.total_length = float(self.arc[-1])

        curvature_abs = np.maximum(np.abs(self.kappa), kappa_eps)
        self.curvature_speed_limit = np.sqrt(
            max(max_lateral_accel, 0.0) / curvature_abs)
        self.yaw_rate_speed_limit = (
            max(max_yaw_rate, 0.0) / curvature_abs)
        self.u_limit = np.minimum.reduce([
            np.full_like(self.kappa, max(cruise_speed, 0.0), dtype=float),
            np.full_like(self.kappa, max(max_speed, 0.0), dtype=float),
            self.curvature_speed_limit,
            self.yaw_rate_speed_limit,
        ])

        self.u_d = self.forward_backward_speed_profile(
            initial_surge_speed, accel_max, decel_max)
        self.r_d = self.u_d * self.kappa
        self.valid = (
            self.total_length > 1e-6 and
            np.all(np.isfinite(self.points)) and
            np.all(np.isfinite(self.psi)) and
            np.all(np.isfinite(self.kappa)) and
            np.all(np.isfinite(self.u_d)) and
            np.all(np.isfinite(self.r_d)))

    def forward_backward_speed_profile(self, initial_surge_speed: float,
                                       accel_max: float,
                                       decel_max: float) -> np.ndarray:
        u = np.asarray(self.u_limit, dtype=float).copy()
        u[0] = min(max(initial_surge_speed, 0.0), u[0])

        accel = max(accel_max, 0.0)
        for index in range(1, len(u)):
            ds = max(self.arc[index] - self.arc[index - 1], 0.0)
            u[index] = min(u[index],
                           math.sqrt(max(0.0, u[index - 1]**2 +
                                         2.0 * accel * ds)))

        u[-1] = 0.0
        decel = max(decel_max, 0.0)
        for index in range(len(u) - 2, -1, -1):
            ds = max(self.arc[index + 1] - self.arc[index], 0.0)
            u[index] = min(u[index],
                           math.sqrt(max(0.0, u[index + 1]**2 +
                                         2.0 * decel * ds)))
        return u

    def closest_arc(self, position: np.ndarray) -> float:
        distances = np.linalg.norm(self.points - position, axis=1)
        return float(self.arc[int(np.argmin(distances))])

    def evaluate_at_arc(self, arc_value: float):
        if self.total_length <= 1e-6:
            index = 0
            return self.sample(index)
        arc_value = clamp(arc_value, 0.0, self.total_length)
        index = int(np.searchsorted(self.arc, arc_value))
        if index <= 0:
            return self.sample(0)
        if index >= len(self.arc):
            return self.sample(len(self.arc) - 1)
        denom = self.arc[index] - self.arc[index - 1]
        if denom <= 1e-9:
            return self.sample(index)
        ratio = (arc_value - self.arc[index - 1]) / denom
        return {
            'arc': arc_value,
            's': float(self.s[index - 1] +
                       ratio * (self.s[index] - self.s[index - 1])),
            'position': self.points[index - 1] +
            ratio * (self.points[index] - self.points[index - 1]),
            'psi': wrap_angle(float(self.psi_unwrapped[index - 1] +
                                    ratio * (
                                        self.psi_unwrapped[index] -
                                        self.psi_unwrapped[index - 1]))),
            'kappa': float(self.kappa[index - 1] +
                           ratio * (self.kappa[index] -
                                    self.kappa[index - 1])),
            'u_d': float(self.u_d[index - 1] +
                         ratio * (self.u_d[index] -
                                  self.u_d[index - 1])),
            'r_d': float(self.r_d[index - 1] +
                         ratio * (self.r_d[index] -
                                  self.r_d[index - 1])),
            'u_limit': float(self.u_limit[index - 1] +
                             ratio * (self.u_limit[index] -
                                      self.u_limit[index - 1])),
            'curvature_speed_limit': float(
                self.curvature_speed_limit[index - 1] +
                ratio * (self.curvature_speed_limit[index] -
                         self.curvature_speed_limit[index - 1])),
            'yaw_rate_speed_limit': float(
                self.yaw_rate_speed_limit[index - 1] +
                ratio * (self.yaw_rate_speed_limit[index] -
                         self.yaw_rate_speed_limit[index - 1])),
        }

    def sample(self, index: int):
        return {
            'arc': float(self.arc[index]),
            's': float(self.s[index]),
            'position': self.points[index],
            'psi': float(self.psi[index]),
            'kappa': float(self.kappa[index]),
            'u_d': float(self.u_d[index]),
            'r_d': float(self.r_d[index]),
            'u_limit': float(self.u_limit[index]),
            'curvature_speed_limit': float(
                self.curvature_speed_limit[index]),
            'yaw_rate_speed_limit': float(self.yaw_rate_speed_limit[index]),
        }


class CascadedPidController(Node):
    """B-spline trajectory tracker for the two-thruster WAM-V.

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
        self.declare_parameter('waypoint_yaw_specified', [True])
        self.declare_parameter('position_tolerance', 0.20)
        self.declare_parameter('yaw_tolerance', 0.08)
        self.declare_parameter('goal_speed_tolerance', 0.15)
        self.declare_parameter('waypoint_tolerance', 0.75)

        self.declare_parameter('kappa_max', 0.7)
        self.declare_parameter('curvature_check_samples', 240)
        self.declare_parameter('kappa_feasibility_tolerance', 1.01)
        self.declare_parameter('bspline_degree', 3)
        self.declare_parameter('bspline_control_points', 8)
        self.declare_parameter('bspline_enforce_waypoints', True)
        self.declare_parameter('bspline_waypoint_tolerance', 0.05)
        self.declare_parameter('bspline_waypoint_weight', 2500.0)
        self.declare_parameter('bspline_yaw_weight', 40.0)
        self.declare_parameter('bspline_smoothness_weight', 0.05)
        self.declare_parameter('bspline_control_regularization', 0.002)
        self.declare_parameter('bspline_control_margin', 20.0)
        self.declare_parameter('bspline_optimizer_samples', 300)
        self.declare_parameter('bspline_optimizer_max_iterations', 220)
        self.declare_parameter('bspline_max_handle', 40.0)
        self.declare_parameter('bspline_curvature_penalty', 20000.0)
        self.declare_parameter('bspline_peak_curvature_penalty', 50.0)
        self.declare_parameter('bspline_curvature_smoothness', 0.02)

        self.declare_parameter('trajectory_samples', 800)
        self.declare_parameter('trajectory_max_lead', 0.05)
        self.declare_parameter('trajectory_accel_max', 0.06)
        self.declare_parameter('trajectory_decel_max', 0.06)
        self.declare_parameter('tracking_kx', 0.35)
        self.declare_parameter('tracking_ky', 2.50)
        self.declare_parameter('tracking_kpsi', 0.80)
        self.declare_parameter('tracking_min_speed_for_lateral', 0.45)

        self.declare_parameter('cruise_speed', 0.30)
        self.declare_parameter('min_speed', -0.25)
        self.declare_parameter('max_speed', 0.45)
        self.declare_parameter('max_lateral_accel', 0.10)
        self.declare_parameter('kappa_eps', 1e-3)

        self.declare_parameter('max_yaw_rate', 0.55)

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
        self.goal_speed_tolerance = self._float_param('goal_speed_tolerance')
        self.waypoint_tolerance = self._float_param('waypoint_tolerance')

        self.kappa_max = self._float_param('kappa_max')
        self.curvature_check_samples = self._int_param(
            'curvature_check_samples')
        self.kappa_feasibility_tolerance = self._float_param(
            'kappa_feasibility_tolerance')
        self.bspline_degree = self._int_param('bspline_degree')
        self.bspline_control_points = self._int_param(
            'bspline_control_points')
        self.bspline_enforce_waypoints = self.get_parameter(
            'bspline_enforce_waypoints').get_parameter_value().bool_value
        self.bspline_waypoint_tolerance = self._float_param(
            'bspline_waypoint_tolerance')
        self.bspline_waypoint_weight = self._float_param(
            'bspline_waypoint_weight')
        self.bspline_yaw_weight = self._float_param('bspline_yaw_weight')
        self.bspline_smoothness_weight = self._float_param(
            'bspline_smoothness_weight')
        self.bspline_control_regularization = self._float_param(
            'bspline_control_regularization')
        self.bspline_control_margin = self._float_param(
            'bspline_control_margin')
        self.bspline_optimizer_samples = self._int_param(
            'bspline_optimizer_samples')
        self.bspline_optimizer_max_iterations = self._int_param(
            'bspline_optimizer_max_iterations')
        self.bspline_max_handle = self._float_param('bspline_max_handle')
        self.bspline_curvature_penalty = self._float_param(
            'bspline_curvature_penalty')
        self.bspline_peak_curvature_penalty = self._float_param(
            'bspline_peak_curvature_penalty')
        self.bspline_curvature_smoothness = self._float_param(
            'bspline_curvature_smoothness')

        self.trajectory_samples = self._int_param('trajectory_samples')
        self.trajectory_max_lead = self._float_param('trajectory_max_lead')
        self.trajectory_accel_max = self._float_param('trajectory_accel_max')
        self.trajectory_decel_max = self._float_param('trajectory_decel_max')
        self.tracking_kx = self._float_param('tracking_kx')
        self.tracking_ky = self._float_param('tracking_ky')
        self.tracking_kpsi = self._float_param('tracking_kpsi')
        self.tracking_min_speed_for_lateral = self._float_param(
            'tracking_min_speed_for_lateral')

        self.cruise_speed = self._float_param('cruise_speed')
        self.min_speed = self._float_param('min_speed')
        self.max_speed = self._float_param('max_speed')
        self.max_lateral_accel = self._float_param('max_lateral_accel')
        self.kappa_eps = self._float_param('kappa_eps')
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
        self.path = None
        self.path_table: Optional[PathTable] = None
        self.trajectory: Optional[TrajectoryTable] = None
        self.trajectory_valid = False
        self.path_max_abs_kappa = 0.0
        self.path_smoothing_attempts = 0
        self.path_curvature_feasible = True
        self.current_path_segment = 0
        self.path_waypoint_offset = 0
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
        yaw_mode = 'specified' if goal.yaw_specified else 'auto'
        self.get_logger().info(
            'B-spline trajectory tracker loaded with '
            f'{len(self.waypoints)} waypoint(s); final target '
            f'x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.yaw:.2f} '
            f'({yaw_mode} yaw).')

    def _str_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _float_param(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _int_param(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _float_array_param(self, name: str) -> List[float]:
        return list(self.get_parameter(name).get_parameter_value().double_array_value)

    def _bool_array_param(self, name: str) -> List[bool]:
        return list(self.get_parameter(name).get_parameter_value().bool_array_value)

    def load_waypoints(self) -> List[Waypoint]:
        xs = self._float_array_param('waypoint_xs')
        ys = self._float_array_param('waypoint_ys')
        yaws = self._float_array_param('waypoint_yaws')
        yaw_specified = self._bool_array_param('waypoint_yaw_specified')
        if not xs or len(xs) != len(ys) or len(xs) != len(yaws):
            self.get_logger().warn(
                'Invalid waypoint arrays; falling back to target_x/y/yaw.')
            return [Waypoint(
                self._float_param('target_x'),
                self._float_param('target_y'),
                self._float_param('target_yaw'))]
        if not yaw_specified:
            yaw_specified = [True] * len(xs)
        elif len(yaw_specified) != len(xs):
            self.get_logger().warn(
                'Invalid waypoint_yaw_specified length; treating all waypoint '
                'yaws as specified.')
            yaw_specified = [True] * len(xs)
        if not yaw_specified[-1]:
            self.get_logger().warn(
                'Final waypoint yaw should be specified for docking; using the '
                'configured final yaw value as the terminal heading.')
            yaw_specified[-1] = True
        return [
            Waypoint(float(x), float(y), float(yaw), bool(yaw_is_specified))
            for x, y, yaw, yaw_is_specified
            in zip(xs, ys, yaws, yaw_specified)
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
                np.linalg.norm(state['velocity']) <
                self.goal_speed_tolerance):
            self.goal_reached = True

        if self.goal_reached:
            self.reset_integrators()
            return self.zero_command(state, final_goal, final_distance,
                                     final_yaw_error)

        if self.path is None or self.trajectory is None:
            self.make_path(state, now)
        if not self.trajectory_valid or self.trajectory is None:
            self.reset_integrators()
            return self.zero_command(state, final_goal, final_distance,
                                     final_yaw_error)

        guidance = self.trajectory_tracking(state, final_goal, dt)
        body_velocity = rot2d(state['yaw']).T @ state['velocity']
        surge_speed = float(body_velocity[0])

        surge_error = guidance['u_cmd'] - surge_speed
        surge_ff = (
            self.surge_drag_linear * guidance['u_cmd'] +
            self.surge_drag_quadratic * abs(guidance['u_cmd']) *
            guidance['u_cmd'])
        surge_force_d = (
            surge_ff +
            self.surge_kp * surge_error +
            self.surge_ki * self.surge_integral)
        surge_force_d = clamp(
            surge_force_d, -self.max_surge_force, self.max_surge_force)

        yaw_rate_error = guidance['r_cmd'] - state['yaw_rate']
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
            'desired_yaw': guidance['psi_d'],
            'desired_yaw_rate': guidance['r_cmd'],
            'distance': final_distance,
            'yaw_error': final_yaw_error,
            'velocity_error_norm': abs(surge_error),
            'waypoint_index': self.current_path_segment,
            's_ref': guidance['s_ref'],
            'arc_ref': guidance['arc_ref'],
            'x_ref': guidance['p_d'][0],
            'y_ref': guidance['p_d'][1],
            'psi_ref': guidance['psi_d'],
            'kappa_ref': guidance['kappa_ref'],
            'e_x': guidance['e_x'],
            'e_y': guidance['e_y'],
            'e_psi': guidance['e_psi'],
            'u_ref': guidance['u_cmd'],
            'u_d': guidance['u_d'],
            'r_d': guidance['r_d'],
            'u_cmd': guidance['u_cmd'],
            'r_cmd': guidance['r_cmd'],
            'trajectory_u_limit': guidance['u_limit'],
            'trajectory_curvature_limit': guidance['curvature_speed_limit'],
            'trajectory_yaw_rate_limit': guidance['yaw_rate_speed_limit'],
            'trajectory_valid': float(self.trajectory_valid),
            'surge_speed': surge_speed,
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
            'arc_ref': 0.0,
            'x_ref': goal.x,
            'y_ref': goal.y,
            'psi_ref': goal.yaw,
            'kappa_ref': 0.0,
            'e_x': 0.0,
            'e_y': 0.0,
            'e_psi': yaw_error,
            'u_ref': 0.0,
            'u_d': 0.0,
            'r_d': 0.0,
            'u_cmd': 0.0,
            'r_cmd': 0.0,
            'trajectory_u_limit': 0.0,
            'trajectory_curvature_limit': 0.0,
            'trajectory_yaw_rate_limit': 0.0,
            'trajectory_valid': float(self.trajectory_valid),
            'surge_speed': 0.0,
            'allocation_cost': 0.0,
            'path_max_abs_kappa': self.path_max_abs_kappa,
            'path_smoothing_attempts': self.path_smoothing_attempts,
            'path_curvature_feasible': float(self.path_curvature_feasible),
        }

    def make_path(self, state, now: float):
        body_velocity = rot2d(state['yaw']).T @ state['velocity']
        speed0 = float(max(body_velocity[0], 0.0))
        waypoint_offset = 0
        remaining_waypoints = self.waypoints
        path_points = (
            [state['position']] +
            [wp.position for wp in remaining_waypoints])
        path_yaws = [state['yaw']] + [wp.yaw for wp in remaining_waypoints]
        path_yaw_specified = [True] + [
            wp.yaw_specified for wp in remaining_waypoints
        ]

        optimized = self.optimize_bspline_path(
            path_points, path_yaws, path_yaw_specified)
        if optimized is None:
            self.path = None
            self.path_table = None
            self.trajectory = None
            self.trajectory_valid = False
            self.path_curvature_feasible = False
            self.get_logger().error(
                'Unable to generate a feasible B-spline trajectory; '
                'publishing zero thrust.')
            return

        best_path, best_table, optimized_info = optimized
        best_attempt = int(optimized_info['iterations'])
        best_max_kappa = best_table.max_abs_kappa
        feasible = bool(optimized_info['feasible'])

        best_path.max_abs_kappa = best_max_kappa
        best_path.smoothing_attempts = best_attempt
        best_path.curvature_feasible = feasible
        self.path = best_path
        self.path_table = best_table
        self.path_max_abs_kappa = self.path_table.max_abs_kappa
        self.path_smoothing_attempts = best_attempt
        self.path_curvature_feasible = feasible
        self.trajectory = TrajectoryTable(
            self.path,
            self.trajectory_samples,
            speed0,
            self.cruise_speed,
            self.max_speed,
            self.max_lateral_accel,
            self.max_yaw_rate,
            self.trajectory_accel_max,
            self.trajectory_decel_max,
            self.kappa_eps)
        self.trajectory_valid = bool(feasible and self.trajectory.valid)

        status = 'feasible' if self.trajectory_valid else 'invalid'
        log = self.get_logger().info if self.trajectory_valid else (
            self.get_logger().error)
        log(
            'B-spline trajectory generation: '
            f'max |kappa|={self.path_max_abs_kappa:.3f} 1/m, '
            f'limit={self.kappa_max:.3f} 1/m, '
            f'iterations={best_attempt}, status={status}.')

        self.path_created_time = now
        self.path_waypoint_offset = waypoint_offset
        self.current_path_segment = waypoint_offset
        self.last_reference_arc = 0.0

    def optimize_bspline_path(self, path_points: List[np.ndarray],
                              path_yaws: List[float],
                              path_yaw_specified: List[bool]):
        if scipy_bspline is None or scipy_minimize is None:
            return None
        if len(path_points) < 2:
            return None

        degree = max(2, min(int(self.bspline_degree), 5))
        control_count = max(
            int(self.bspline_control_points),
            degree + 3,
            len(path_points) + degree + 1)
        max_s = float(max(len(path_points) - 1, 1))
        waypoint_params = self.path_parameters(path_points, max_s)
        knots = self.open_uniform_knots(control_count, degree, max_s)
        seed_controls = self.seed_bspline_controls(
            path_points, path_yaws, path_yaw_specified, control_count,
            max_s)
        specs, z_seed, bounds = self.bspline_variable_encoding(
            seed_controls, path_points, path_yaws, path_yaw_specified)
        initial_guesses = self.bspline_initial_guesses(z_seed, specs)

        samples = max(self.bspline_optimizer_samples, 30)
        max_iterations = max(self.bspline_optimizer_max_iterations, 1)
        limit = max(self.kappa_max, 1e-6)
        best = None

        def build_path(z: np.ndarray):
            controls = self.build_bspline_controls(
                z, specs, seed_controls, path_points)
            path = BSplinePath(controls, knots, degree, max_s)
            return path, controls

        def waypoint_error(path) -> float:
            if len(path_points) <= 2:
                return 0.0
            errors = []
            for point, param in zip(path_points[1:-1],
                                    waypoint_params[1:-1]):
                p_i, *_ = path.evaluate(float(param))
                errors.append(float(np.linalg.norm(p_i - point)))
            return float(np.sqrt(np.mean(np.asarray(errors)**2)))

        def specified_yaw_cost(path) -> float:
            total = 0.0
            count = 0
            for yaw, specified, param in zip(path_yaws,
                                             path_yaw_specified,
                                             waypoint_params):
                if not specified:
                    continue
                _p, _p_s, _p_ss, psi, _kappa = path.evaluate(float(param))
                total += wrap_angle(psi - yaw)**2
                count += 1
            return total / max(count, 1)

        def smoothness_cost(controls: np.ndarray) -> float:
            if len(controls) < 3:
                return 0.0
            second_diff = (
                controls[:-2] - 2.0 * controls[1:-1] + controls[2:])
            return float(np.mean(np.sum(second_diff**2, axis=1)))

        def evaluate_candidate(z: np.ndarray):
            path, controls = build_path(z)
            length, max_kappa, kappas = self.sampled_path_metrics(
                path, samples)
            overflow = np.maximum(np.abs(kappas) - limit, 0.0)
            curvature_smoothness = 0.0
            if len(kappas) > 1:
                curvature_smoothness = float(np.mean(np.diff(kappas)**2))
            wp_rms = waypoint_error(path)
            objective = (
                length +
                self.bspline_curvature_penalty *
                float(np.mean(overflow**2)) +
                self.bspline_peak_curvature_penalty *
                float(np.max(overflow)**2) +
                self.bspline_waypoint_weight * wp_rms**2 +
                self.bspline_yaw_weight * specified_yaw_cost(path) +
                self.bspline_smoothness_weight * smoothness_cost(controls) +
                self.bspline_control_regularization *
                float(np.mean(np.sum((controls - seed_controls)**2, axis=1))) +
                self.bspline_curvature_smoothness *
                curvature_smoothness)
            return path, controls, length, max_kappa, objective, wp_rms

        def objective(z: np.ndarray) -> float:
            return evaluate_candidate(z)[4]

        def curvature_constraint(z: np.ndarray) -> float:
            return limit - evaluate_candidate(z)[3]

        constraints = [{
            'type': 'ineq',
            'fun': curvature_constraint,
        }]

        if self.bspline_enforce_waypoints and len(path_points) > 2:
            def waypoint_constraint(z: np.ndarray) -> np.ndarray:
                path, _controls = build_path(z)
                residuals = []
                for point, param in zip(path_points[1:-1],
                                        waypoint_params[1:-1]):
                    p_i, *_ = path.evaluate(float(param))
                    residuals.extend((p_i - point).tolist())
                return np.asarray(residuals, dtype=float)
            constraints.append({
                'type': 'eq',
                'fun': waypoint_constraint,
            })

        for z0 in initial_guesses:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = scipy_minimize(
                        objective,
                        z0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            'maxiter': max_iterations,
                            'ftol': 1e-7,
                            'disp': False,
                        })
                z = np.asarray(result.x, dtype=float)
                path, _controls, length, max_kappa, objective_value, wp_rms = (
                    evaluate_candidate(z))
            except Exception as exc:  # pragma: no cover - optimizer fallback
                self.get_logger().warn(
                    f'B-spline optimization attempt failed: {exc}')
                continue

            feasible = (
                max_kappa <= limit * self.kappa_feasibility_tolerance and
                wp_rms <= max(self.bspline_waypoint_tolerance, 1e-6))
            candidate = {
                'path': path,
                'length': length,
                'max_kappa': max_kappa,
                'objective': objective_value,
                'iterations': int(getattr(result, 'nit', 0)),
                'feasible': feasible,
                'waypoint_rms': wp_rms,
                'method': 'bspline-slsqp',
            }
            if best is None:
                best = candidate
                continue
            if candidate['feasible'] and not best['feasible']:
                best = candidate
            elif candidate['feasible'] == best['feasible']:
                if candidate['max_kappa'] < best['max_kappa']:
                    best = candidate
                elif math.isclose(candidate['max_kappa'],
                                  best['max_kappa']):
                    if candidate['waypoint_rms'] < best['waypoint_rms']:
                        best = candidate
                    elif math.isclose(candidate['waypoint_rms'],
                                      best['waypoint_rms']):
                        if candidate['objective'] < best['objective']:
                            best = candidate

        if best is None:
            return None
        table = PathTable(best['path'], self.curvature_check_samples)
        best['max_kappa'] = table.max_abs_kappa
        best['feasible'] = (
            best['max_kappa'] <=
            self.kappa_max * self.kappa_feasibility_tolerance and
            best['waypoint_rms'] <= max(self.bspline_waypoint_tolerance, 1e-6))
        return best['path'], table, best

    def path_parameters(self, points: List[np.ndarray],
                        max_s: float) -> np.ndarray:
        if len(points) < 2:
            return np.array([0.0], dtype=float)
        segment_lengths = np.asarray([
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(len(points) - 1)
        ], dtype=float)
        total = float(np.sum(segment_lengths))
        if total <= 1e-9:
            return np.linspace(0.0, max_s, len(points))
        params = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        return params / total * max_s

    def open_uniform_knots(self, control_count: int, degree: int,
                           max_s: float) -> np.ndarray:
        internal_count = control_count - degree - 1
        if internal_count > 0:
            internal = np.linspace(
                0.0, max_s, internal_count + 2, dtype=float)[1:-1]
        else:
            internal = np.asarray([], dtype=float)
        return np.concatenate((
            np.zeros(degree + 1, dtype=float),
            internal,
            np.full(degree + 1, max_s, dtype=float)))

    def seed_bspline_controls(self, points: List[np.ndarray],
                              yaws: List[float],
                              yaw_specified: List[bool],
                              control_count: int,
                              max_s: float) -> np.ndarray:
        waypoint_params = self.path_parameters(points, max_s)
        control_params = np.linspace(0.0, max_s, control_count)
        point_array = np.asarray(points, dtype=float)
        controls = np.column_stack((
            np.interp(control_params, waypoint_params, point_array[:, 0]),
            np.interp(control_params, waypoint_params, point_array[:, 1]),
        ))
        controls[0] = points[0]
        controls[-1] = points[-1]

        handles = self.path_handle_lengths(points)
        if yaw_specified[0] and control_count > 2:
            direction = np.array([
                math.cos(yaws[0]), math.sin(yaws[0])
            ], dtype=float)
            controls[1] = points[0] + handles[0] * direction
        if yaw_specified[-1] and control_count > 3:
            direction = np.array([
                math.cos(yaws[-1]), math.sin(yaws[-1])
            ], dtype=float)
            controls[-2] = points[-1] - handles[-1] * direction
        return controls

    def bspline_variable_encoding(self, seed_controls: np.ndarray,
                                  points: List[np.ndarray],
                                  yaws: List[float],
                                  yaw_specified: List[bool]):
        specs = []
        z0 = []
        bounds = []
        max_handle = max(self.bspline_max_handle, 0.3)
        point_array = np.asarray(points, dtype=float)
        margin = max(self.bspline_control_margin, 1.0)
        lower_xy = np.min(point_array, axis=0) - margin
        upper_xy = np.max(point_array, axis=0) + margin
        control_count = len(seed_controls)

        for index in range(control_count):
            if index == 0 or index == control_count - 1:
                specs.append(('fixed', index, None))
                continue
            if index == 1 and yaw_specified[0]:
                direction = np.array([
                    math.cos(yaws[0]), math.sin(yaws[0])
                ], dtype=float)
                magnitude = float((seed_controls[index] - points[0]) @
                                  direction)
                specs.append(('start_handle', index, direction))
                z0.append(clamp(magnitude, 0.3, max_handle))
                bounds.append((0.3, max_handle))
                continue
            if index == control_count - 2 and yaw_specified[-1]:
                direction = np.array([
                    math.cos(yaws[-1]), math.sin(yaws[-1])
                ], dtype=float)
                magnitude = float((points[-1] - seed_controls[index]) @
                                  direction)
                specs.append(('goal_handle', index, direction))
                z0.append(clamp(magnitude, 0.3, max_handle))
                bounds.append((0.3, max_handle))
                continue

            specs.append(('point', index, None))
            z0.extend([
                float(seed_controls[index, 0]),
                float(seed_controls[index, 1]),
            ])
            bounds.extend([
                (float(lower_xy[0]), float(upper_xy[0])),
                (float(lower_xy[1]), float(upper_xy[1])),
            ])
        return specs, np.asarray(z0, dtype=float), bounds

    def build_bspline_controls(self, z: np.ndarray, specs,
                               seed_controls: np.ndarray,
                               points: List[np.ndarray]) -> np.ndarray:
        controls = np.asarray(seed_controls, dtype=float).copy()
        cursor = 0
        for kind, index, direction in specs:
            if kind == 'fixed':
                continue
            if kind == 'start_handle':
                controls[index] = points[0] + float(z[cursor]) * direction
                cursor += 1
            elif kind == 'goal_handle':
                controls[index] = points[-1] - float(z[cursor]) * direction
                cursor += 1
            else:
                controls[index] = np.array([
                    float(z[cursor]),
                    float(z[cursor + 1]),
                ], dtype=float)
                cursor += 2
        controls[0] = points[0]
        controls[-1] = points[-1]
        return controls

    def bspline_initial_guesses(self, z_seed: np.ndarray, specs):
        guesses = [np.asarray(z_seed, dtype=float)]
        guesses.append(self.scale_bspline_handle_guess(z_seed, specs, 1.5))
        guesses.append(self.scale_bspline_handle_guess(z_seed, specs, 2.5))
        unique = []
        for guess in guesses:
            if not any(np.allclose(guess, existing) for existing in unique):
                unique.append(guess)
        return unique

    def scale_bspline_handle_guess(self, z: np.ndarray, specs,
                                   scale: float) -> np.ndarray:
        scaled = np.asarray(z, dtype=float).copy()
        cursor = 0
        max_handle = max(self.bspline_max_handle, 0.3)
        for kind, _index, _direction in specs:
            if kind == 'fixed':
                continue
            if kind in ('start_handle', 'goal_handle'):
                scaled[cursor] = clamp(
                    scaled[cursor] * scale, 0.3, max_handle)
                cursor += 1
            else:
                cursor += 2
        return scaled

    def sampled_path_metrics(self, path, samples: int):
        sample_s = np.linspace(0.0, path.max_s, max(samples, 20))
        evaluated = [path.evaluate(float(si)) for si in sample_s]
        kappas = np.asarray([item[4] for item in evaluated])
        points = np.asarray([item[0] for item in evaluated])
        if len(points) < 2:
            length = 0.0
        else:
            length = float(np.sum(np.linalg.norm(
                np.diff(points, axis=0), axis=1)))
        max_kappa = float(np.max(np.abs(kappas)))
        return length, max_kappa, kappas

    def path_handle_lengths(self, points: List[np.ndarray]) -> List[float]:
        segment_lengths = [
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(len(points) - 1)
        ]
        if not segment_lengths:
            return [0.3]
        handles = []
        for index in range(len(points)):
            if index == 0:
                length = 0.35 * segment_lengths[0]
            elif index == len(points) - 1:
                length = 0.35 * segment_lengths[-1]
            else:
                length = 0.175 * (
                    segment_lengths[index - 1] + segment_lengths[index])
            handles.append(clamp(length, 0.3, self.bspline_max_handle))
        return handles

    def trajectory_tracking(self, state, goal: Waypoint, _dt: float):
        assert self.trajectory is not None

        position = state['position']
        closest_arc = self.trajectory.closest_arc(position)
        lead = clamp(self.trajectory_max_lead, 0.0,
                     self.trajectory.total_length)
        target_arc = min(closest_arc + lead, self.trajectory.total_length)
        arc_ref = max(target_arc, self.last_reference_arc)
        self.last_reference_arc = arc_ref
        ref = self.trajectory.evaluate_at_arc(arc_ref)

        local_segment = int(math.floor(min(ref['s'], self.path.max_s - 1e-6)))
        self.current_path_segment = min(
            self.path_waypoint_offset + local_segment,
            len(self.waypoints) - 1)

        p_d = np.asarray(ref['position'], dtype=float)
        psi_d = float(ref['psi'])
        e_body = rot2d(state['yaw']).T @ (p_d - position)
        e_x = float(e_body[0])
        e_y = float(e_body[1])
        e_psi = wrap_angle(psi_d - state['yaw'])

        u_d = float(ref['u_d'])
        r_d = float(ref['r_d'])
        u_cmd = u_d * math.cos(e_psi) + self.tracking_kx * e_x
        u_cmd = clamp(u_cmd, self.min_speed, self.max_speed)

        u_eff = max(abs(u_d), self.tracking_min_speed_for_lateral)
        r_cmd = (
            r_d +
            self.tracking_ky * u_eff * e_y +
            self.tracking_kpsi * math.sin(e_psi))
        r_cmd = clamp(r_cmd, -self.max_yaw_rate, self.max_yaw_rate)

        return {
            's_ref': float(ref['s']),
            'arc_ref': arc_ref,
            'p_d': p_d,
            'psi_d': psi_d,
            'kappa_ref': float(ref['kappa']),
            'e_x': e_x,
            'e_y': e_y,
            'e_psi': e_psi,
            'u_d': u_d,
            'r_d': r_d,
            'u_cmd': u_cmd,
            'r_cmd': r_cmd,
            'u_limit': float(ref['u_limit']),
            'curvature_speed_limit': float(ref['curvature_speed_limit']),
            'yaw_rate_speed_limit': float(ref['yaw_rate_speed_limit']),
        }

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
            # 0-8: measured state and final target.
            float(state['position'][0]),
            float(state['position'][1]),
            float(state['yaw']),
            float(state['velocity'][0]),
            float(state['velocity'][1]),
            float(state['yaw_rate']),
            float(final_goal.x),
            float(final_goal.y),
            float(final_goal.yaw),
            # 9-24: trajectory reference, commands, and tracking errors.
            float(cmd['x_ref']),
            float(cmd['y_ref']),
            float(cmd['psi_ref']),
            float(cmd['u_d']),
            float(cmd['r_d']),
            float(cmd['u_cmd']),
            float(cmd['r_cmd']),
            float(cmd['e_x']),
            float(cmd['e_y']),
            float(cmd['e_psi']),
            float(cmd['s_ref']),
            float(cmd['arc_ref']),
            float(cmd['kappa_ref']),
            float(cmd['trajectory_u_limit']),
            float(cmd['trajectory_curvature_limit']),
            float(cmd['trajectory_yaw_rate_limit']),
            # 25-32: inner-loop and allocation outputs.
            float(cmd['surge_speed']),
            float(cmd['force']),
            float(cmd['torque']),
            float(cmd['left_thrust']),
            float(cmd['right_thrust']),
            float(cmd['surge_force_c']),
            float(cmd['tau_c']),
            float(cmd['allocation_cost']),
            # 33-39: path validity and terminal metrics.
            float(cmd['path_max_abs_kappa']),
            float(cmd['path_curvature_feasible']),
            float(cmd['trajectory_valid']),
            float(cmd['distance']),
            float(cmd['yaw_error']),
            float(cmd['velocity_error_norm']),
            float(cmd['waypoint_index']),
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
