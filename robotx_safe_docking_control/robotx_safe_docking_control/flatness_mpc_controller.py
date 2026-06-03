import math
import os
import time
import ctypes
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:  # pragma: no cover - runtime dependency guard
    scipy_minimize = None

try:
    import casadi as ca
    from acados_template import AcadosModel
    from acados_template import AcadosOcp
    from acados_template import AcadosOcpSolver
except Exception:  # pragma: no cover - runtime dependency guard
    ca = None
    AcadosModel = None
    AcadosOcp = None
    AcadosOcpSolver = None

from .feasible_references import available_reference_names
from .feasible_references import make_synthetic_reference
from .usv_flatness import UsvModelParams
from .usv_flatness import flat_to_body_velocity
from .usv_flatness import generalized_force_to_thrust
from .usv_flatness import theta_tau_nominal
from .usv_flatness import thrust_to_generalized_force
from .usv_flatness import wrap_angle


DEBUG_FIELDS = [
    'elapsed',
    'reference_duration',
    'solver_success',
    'solver_backend',
    'solver_status',
    'solve_time_ms',
    'objective',
    'x_ref',
    'y_ref',
    'psi_ref',
    'x',
    'y',
    'psi',
    'e_x',
    'e_y',
    'e_psi',
    'x_dot_ref',
    'y_dot_ref',
    'psi_dot_ref',
    'x_dot',
    'y_dot',
    'psi_dot',
    'tau_u',
    'tau_v',
    'tau_v_slack',
    'tau_r',
    'left_raw',
    'right_raw',
    'left_cmd',
    'right_cmd',
    'thrust_saturated',
    'max_pred_tau_v',
    'rms_pred_tau_v',
    'max_pred_tau_v_slack',
    'rms_pred_tau_v_slack',
    'max_pred_tau_v_violation',
    'max_pred_tau_u',
    'max_pred_tau_r',
    'max_pred_left_abs',
    'max_pred_right_abs',
    'terminal_pos_error',
    'terminal_yaw_error',
    'terminal_x_ref',
    'terminal_y_ref',
    'terminal_psi_ref',
    'fail_reason',
    'allocation_tau_u_residual',
    'allocation_tau_r_residual',
    'scenario_complete',
]


FAIL_OK = 0.0
FAIL_NO_ODOM = 1.0
FAIL_NO_REFERENCE = 2.0
FAIL_OPTIMIZER_UNAVAILABLE = 3.0
FAIL_OPTIMIZER_FAILED = 4.0

SOLVER_BACKEND_SLSQP = 0.0
SOLVER_BACKEND_ACADOS = 1.0


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class FlatnessMpcController(Node):
    """RK4 dynamics-MPC tracker for nominally feasible USV references.

    The acados backend optimizes the local-frame state
    [x, y, psi, x_dot, y_dot, psi_dot], generalized input
    [tau_u, tau_v, tau_r], and one constant tau_v slack state. Two extra
    bookkeeping states carry the previous tau_u and tau_r so the NLP can
    penalize command steps. tau_v is constrained tightly through the slack,
    while only tau_u and tau_r are allocated to the differential-thrust WAM-V
    actuators. There is no PID fallback.
    """

    def __init__(self):
        super().__init__('flatness_mpc_controller')

        self.declare_parameter('odom_topic', '/safe_docking/odometry')
        self.declare_parameter(
            'left_thrust_topic', '/wamv/thrusters/left/thrust')
        self.declare_parameter(
            'right_thrust_topic', '/wamv/thrusters/right/thrust')
        self.declare_parameter(
            'command_topic', '/safe_docking/flatness_mpc/command')
        self.declare_parameter(
            'debug_topic', '/safe_docking/flatness_mpc/debug')
        self.declare_parameter('control_rate_hz', 5.0)
        self.declare_parameter('command_timeout_sec', 0.8)

        self.declare_parameter('reference_name', 'arc')
        self.declare_parameter('reference_dt', 0.2)
        self.declare_parameter('solver_backend', 'acados')
        self.declare_parameter('acados_source_dir', '/home/zjy/acados')
        self.declare_parameter(
            'acados_codegen_dir',
            '/tmp/robotx_safe_docking_acados')
        self.declare_parameter(
            'acados_qp_solver', 'PARTIAL_CONDENSING_HPIPM')
        self.declare_parameter('acados_nlp_solver_type', 'SQP')
        self.declare_parameter('acados_integrator_type', 'DISCRETE')
        self.declare_parameter('acados_max_iter', 80)
        self.declare_parameter('acados_levenberg_marquardt', 1e-3)
        self.declare_parameter('acados_regularize_method', 'CONVEXIFY')
        self.declare_parameter('acados_globalization', 'MERIT_BACKTRACKING')
        self.declare_parameter('horizon_steps', 12)
        self.declare_parameter('horizon_dt', 0.25)
        self.declare_parameter(
            'horizon_dt_sequence',
            [0.08, 0.08, 0.10, 0.12, 0.15, 0.20,
             0.25, 0.35, 0.45, 0.55, 0.70, 0.97])
        self.declare_parameter('solver_max_iterations', 60)
        self.declare_parameter('solver_ftol', 1e-3)
        self.declare_parameter('accel_bound_xy', 0.9)
        self.declare_parameter('yaw_accel_bound', 0.8)

        self.declare_parameter('q_position', 4.0)
        self.declare_parameter('q_yaw', 3.0)
        self.declare_parameter('q_velocity', 2.0)
        self.declare_parameter('q_yaw_rate', 1.5)
        self.declare_parameter('q_terminal_position', 700.0)
        self.declare_parameter('q_terminal_yaw', 320.0)
        self.declare_parameter('q_terminal_velocity', 200.0)
        self.declare_parameter('q_terminal_yaw_rate', 100.0)
        self.declare_parameter('q_tau_u', 0.0005)
        self.declare_parameter('q_tau_r', 0.00008)
        self.declare_parameter('q_tau_v_slack', 0.01)
        self.declare_parameter('q_delta_tau', 0.0002)
        self.declare_parameter('tau_v_slack_max', 250.0)
        self.declare_parameter('tau_v_slack_warm_start', 1.0)
        self.declare_parameter('max_yaw_moment', 150.0)
        self.declare_parameter('constraint_violation_tolerance', 5e-3)

        self.declare_parameter('mass', 180.0)
        self.declare_parameter('iz', 446.0)
        self.declare_parameter('du', 100.0)
        self.declare_parameter('duu', 150.0)
        self.declare_parameter('dv', 100.0)
        self.declare_parameter('dvv', 100.0)
        self.declare_parameter('dr', 800.0)
        self.declare_parameter('drr', 800.0)
        self.declare_parameter('thruster_half_spacing', 1.027135)
        self.declare_parameter('min_thrust', -100.0)
        self.declare_parameter('max_thrust', 100.0)
        self.declare_parameter('thrust_rate_limit', 1000.0)

        self.odom_topic = self._str_param('odom_topic')
        self.command_topic = self._str_param('command_topic')
        self.debug_topic = self._str_param('debug_topic')
        self.control_period = 1.0 / max(self._float_param('control_rate_hz'), 0.1)
        self.command_timeout_sec = self._float_param('command_timeout_sec')
        self.reference_name = self._str_param('reference_name')
        self.reference_dt = self._float_param('reference_dt')
        self.solver_backend = self._str_param('solver_backend').strip().lower()
        self.acados_source_dir = self._str_param('acados_source_dir')
        self.acados_codegen_dir = self._str_param('acados_codegen_dir')
        self.acados_qp_solver = self._str_param('acados_qp_solver')
        self.acados_nlp_solver_type = self._str_param('acados_nlp_solver_type')
        self.acados_integrator_type = self._str_param('acados_integrator_type')
        self.acados_max_iter = self._int_param('acados_max_iter')
        self.acados_levenberg_marquardt = self._float_param(
            'acados_levenberg_marquardt')
        self.acados_regularize_method = self._str_param(
            'acados_regularize_method')
        self.acados_globalization = self._str_param('acados_globalization')
        self.horizon_steps = self._int_param('horizon_steps')
        self.horizon_dt = self._float_param('horizon_dt')
        self.horizon_dt_sequence = self.normalized_horizon_dt_sequence()
        self.horizon_offsets = np.concatenate(
            ([0.0], np.cumsum(self.horizon_dt_sequence)))
        self.horizon_node_weights = self.node_quadrature_weights()
        self.solver_max_iterations = self._int_param('solver_max_iterations')
        self.solver_ftol = self._float_param('solver_ftol')
        self.accel_bound_xy = self._float_param('accel_bound_xy')
        self.yaw_accel_bound = self._float_param('yaw_accel_bound')
        self.tau_v_slack_max = self._float_param('tau_v_slack_max')
        self.tau_v_slack_warm_start = self._float_param(
            'tau_v_slack_warm_start')
        self.max_yaw_moment = self._float_param('max_yaw_moment')
        self.constraint_violation_tolerance = self._float_param(
            'constraint_violation_tolerance')

        self.weights = {
            'position': self._float_param('q_position'),
            'yaw': self._float_param('q_yaw'),
            'velocity': self._float_param('q_velocity'),
            'yaw_rate': self._float_param('q_yaw_rate'),
            'terminal_position': self._float_param('q_terminal_position'),
            'terminal_yaw': self._float_param('q_terminal_yaw'),
            'terminal_velocity': self._float_param('q_terminal_velocity'),
            'terminal_yaw_rate': self._float_param('q_terminal_yaw_rate'),
            'tau_u': self._float_param('q_tau_u'),
            'tau_r': self._float_param('q_tau_r'),
            'tau_v_slack': self._float_param('q_tau_v_slack'),
            'delta_tau': self._float_param('q_delta_tau'),
        }

        self.model_params = UsvModelParams(
            mass=self._float_param('mass'),
            iz=self._float_param('iz'),
            du=self._float_param('du'),
            duu=self._float_param('duu'),
            dv=self._float_param('dv'),
            dvv=self._float_param('dvv'),
            dr=self._float_param('dr'),
            drr=self._float_param('drr'),
            thruster_half_spacing=self._float_param(
                'thruster_half_spacing'),
            min_thrust=self._float_param('min_thrust'),
            max_thrust=self._float_param('max_thrust'),
            thrust_rate_limit=self._float_param('thrust_rate_limit'),
        )

        self.latest_odom: Optional[Odometry] = None
        self.last_control_time: Optional[float] = None
        self.reference_start_time: Optional[float] = None
        self.reference = None
        self.last_solution = None
        self.last_acados_x = None
        self.last_acados_u = None
        self.acados_solver = None
        self.acados_ready = False
        self.last_tau = np.zeros(3, dtype=float)
        self.prev_left_thrust = 0.0
        self.prev_right_thrust = 0.0
        self.last_debug = {field: 0.0 for field in DEBUG_FIELDS}

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

        available = ', '.join(available_reference_names(self.model_params))
        self.get_logger().info(
            'Flatness MPC loaded. Synthetic reference='
            f'{self.reference_name}; available references: {available}.')
        self.configure_solver_backend()

    def _str_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _float_param(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _int_param(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _float_array_param(self, name: str):
        value = self.get_parameter(name).get_parameter_value()
        if len(value.double_array_value):
            return [float(item) for item in value.double_array_value]
        if value.string_value:
            return [
                float(item.strip())
                for item in value.string_value.split(',')
                if item.strip()
            ]
        return []

    def normalized_horizon_dt_sequence(self):
        sequence = self._float_array_param('horizon_dt_sequence')
        if len(sequence) != self.horizon_steps:
            sequence = [self.horizon_dt] * self.horizon_steps
        return np.asarray([
            max(float(item), 1e-3)
            for item in sequence
        ], dtype=float)

    def node_quadrature_weights(self):
        weights = np.zeros(self.horizon_steps + 1, dtype=float)
        weights[0] = 0.5 * self.horizon_dt_sequence[0]
        weights[-1] = 0.5 * self.horizon_dt_sequence[-1]
        for index in range(1, self.horizon_steps):
            weights[index] = 0.5 * (
                self.horizon_dt_sequence[index - 1] +
                self.horizon_dt_sequence[index])
        return weights

    def configure_solver_backend(self):
        if self.solver_backend == 'acados':
            self.acados_ready = self.create_acados_solver()
            return
        if self.solver_backend == 'slsqp':
            if scipy_minimize is None:
                self.get_logger().error(
                    'solver_backend=slsqp requested, but scipy.optimize '
                    'is unavailable.')
            else:
                self.get_logger().warn(
                    'Using diagnostic SLSQP backend. This is slower than '
                    'the acados flatness MPC backend.')
            return
        self.get_logger().error(
            f'Unknown solver_backend=[{self.solver_backend}]. Expected '
            'acados or slsqp.')

    def load_acados_shared_libraries(self) -> bool:
        lib_dir = os.path.join(self.acados_source_dir, 'lib')
        os.environ.setdefault('ACADOS_SOURCE_DIR', self.acados_source_dir)
        existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_dir not in existing_ld_path.split(':'):
            os.environ['LD_LIBRARY_PATH'] = (
                lib_dir + (':' + existing_ld_path if existing_ld_path else ''))
        libraries = [
            'libblasfeo.so',
            'libhpipm.so',
            'libqpOASES_e.so',
            'libqpdunes.so',
            'libdaqp.so',
            'libosqp.so',
            'libacados.so',
        ]
        try:
            for library in libraries:
                path = os.path.join(lib_dir, library)
                if os.path.exists(path):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            self.get_logger().error(
                f'Unable to load acados shared libraries from {lib_dir}: '
                f'{exc}')
            return False
        return True

    def create_acados_solver(self) -> bool:
        if ca is None or AcadosOcpSolver is None:
            self.get_logger().error(
                'solver_backend=acados requested, but acados_template or '
                'casadi is unavailable.')
            return False
        if not os.path.exists(self.acados_source_dir):
            self.get_logger().error(
                f'acados_source_dir does not exist: {self.acados_source_dir}')
            return False
        if not self.load_acados_shared_libraries():
            return False

        try:
            ocp = self.build_acados_ocp()
            self.acados_solver = AcadosOcpSolver(
                ocp, json_file=ocp.code_gen_opts.json_file, verbose=False)
        except Exception as exc:
            self.get_logger().error(f'Failed to create acados solver: {exc}')
            self.acados_solver = None
            return False

        self.get_logger().info(
            'acados RK4 dynamics MPC backend ready: '
            f'N={self.horizon_steps}, dt={self.horizon_dt_sequence.tolist()}, '
            f'nlp={self.acados_nlp_solver_type}, qp={self.acados_qp_solver}.')
        return True

    def build_acados_ocp(self):
        params = self.model_params
        model = AcadosModel()
        x = ca.SX.sym('x', 9)
        u = ca.SX.sym('u', 3)
        p = ca.SX.sym('p', 15)

        px, py, psi, vx, vy, r = x[0], x[1], x[2], x[3], x[4], x[5]
        tau_v_slack = x[6]
        prev_tau_u, prev_tau_r = x[7], x[8]
        tau_u_cmd, tau_v_cmd, tau_r_cmd = u[0], u[1], u[2]
        h_step = p[14]

        smooth = 1e-6

        def continuous_dynamics(state, control):
            px_i, py_i, psi_i = state[0], state[1], state[2]
            vx_i, vy_i, r_i = state[3], state[4], state[5]
            slack_i = state[6]
            tau_u_i, tau_v_i, tau_r_i = control[0], control[1], control[2]
            c_i = ca.cos(psi_i)
            s_i = ca.sin(psi_i)
            nu_u_i = c_i * vx_i + s_i * vy_i
            nu_v_i = -s_i * vx_i + c_i * vy_i
            abs_u_i = ca.sqrt(nu_u_i * nu_u_i + smooth)
            abs_v_i = ca.sqrt(nu_v_i * nu_v_i + smooth)
            abs_r_i = ca.sqrt(r_i * r_i + smooth)
            damp_u_i = params.du * nu_u_i + params.duu * abs_u_i * nu_u_i
            damp_v_i = params.dv * nu_v_i + params.dvv * abs_v_i * nu_v_i
            damp_r_i = params.dr * r_i + params.drr * abs_r_i * r_i
            nu_dot_u_i = (
                tau_u_i + params.mass * nu_v_i * r_i - damp_u_i) / params.mass
            nu_dot_v_i = (
                tau_v_i - params.mass * nu_u_i * r_i - damp_v_i) / params.mass
            nu_dot_r_i = (tau_r_i - damp_r_i) / params.iz
            ax_i = c_i * nu_dot_u_i - s_i * nu_dot_v_i - r_i * vy_i
            ay_i = s_i * nu_dot_u_i + c_i * nu_dot_v_i + r_i * vx_i
            return ca.vertcat(
                vx_i,
                vy_i,
                r_i,
                ax_i,
                ay_i,
                nu_dot_r_i,
                0.0 * slack_i,
                0.0,
                0.0,
            )

        c = ca.cos(psi)
        s = ca.sin(psi)
        nu_u = c * vx + s * vy
        nu_v = -s * vx + c * vy
        nu_r = r

        tau_u = tau_u_cmd
        tau_v = tau_v_cmd
        tau_r = tau_r_cmd
        left = 0.5 * (tau_u - tau_r / params.thruster_half_spacing)
        right = 0.5 * (tau_u + tau_r / params.thruster_half_spacing)

        k1 = continuous_dynamics(x, u)
        k2 = continuous_dynamics(x + 0.5 * h_step * k1, u)
        k3 = continuous_dynamics(x + 0.5 * h_step * k2, u)
        k4 = continuous_dynamics(x + h_step * k3, u)
        rk4_next = x + (h_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        disc_dyn = ca.vertcat(
            rk4_next[0:7],
            tau_u_cmd,
            tau_r_cmd,
        )

        model.name = 'robotx_dynamics_rk4_mpc'
        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = disc_dyn
        model.con_h_expr = ca.vertcat(
            left,
            right,
            tau_r,
            tau_v_slack - tau_v,
            tau_v_slack + tau_v,
        )
        model.con_h_expr_0 = model.con_h_expr

        ref_z = p[0:3]
        ref_z_dot = p[3:6]
        q_pos = p[6]
        q_yaw = p[7]
        q_vel = p[8]
        q_yaw_rate = p[9]
        q_tau_u = p[10]
        q_tau_r = p[11]
        q_slack = p[12]
        q_delta_tau = p[13]
        yaw_error = ca.atan2(ca.sin(psi - ref_z[2]), ca.cos(psi - ref_z[2]))
        vel_error_x = vx - ref_z_dot[0]
        vel_error_y = vy - ref_z_dot[1]
        vel_error_r = nu_r - ref_z_dot[2]
        model.cost_expr_ext_cost = (
            q_pos * ((px - ref_z[0])**2 + (py - ref_z[1])**2) +
            q_yaw * yaw_error**2 +
            q_vel * (vel_error_x**2 + vel_error_y**2) +
            q_yaw_rate * vel_error_r**2 +
            q_tau_u * tau_u**2 +
            q_tau_r * tau_r**2 +
            q_delta_tau * (
                (tau_u - prev_tau_u)**2 +
                (tau_r - prev_tau_r)**2) +
            q_slack * tau_v_slack**2)
        model.cost_expr_ext_cost_e = (
            q_pos * ((px - ref_z[0])**2 + (py - ref_z[1])**2) +
            q_yaw * yaw_error**2 +
            q_vel * (vel_error_x**2 + vel_error_y**2) +
            q_yaw_rate * vel_error_r**2)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.horizon_steps
        ocp.solver_options.tf = float(np.sum(self.horizon_dt_sequence))
        ocp.solver_options.time_steps = self.horizon_dt_sequence
        ocp.parameter_values = np.zeros(15, dtype=float)
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        ocp.constraints.idxbx_0 = np.arange(9, dtype=np.int64)
        ocp.constraints.lbx_0 = np.zeros(9, dtype=float)
        ocp.constraints.ubx_0 = np.zeros(9, dtype=float)
        ocp.constraints.idxbx = np.array([6], dtype=np.int64)
        ocp.constraints.lbx = np.array([0.0], dtype=float)
        ocp.constraints.ubx = np.array([self.tau_v_slack_max], dtype=float)
        ocp.constraints.lh = np.array([
            params.min_thrust,
            params.min_thrust,
            -self.max_yaw_moment,
            0.0,
            0.0,
        ], dtype=float)
        ocp.constraints.uh = np.array([
            params.max_thrust,
            params.max_thrust,
            self.max_yaw_moment,
            1e6,
            1e6,
        ], dtype=float)
        ocp.constraints.lh_0 = ocp.constraints.lh
        ocp.constraints.uh_0 = ocp.constraints.uh

        ocp.solver_options.qp_solver = self.acados_qp_solver
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = self.acados_nlp_solver_type
        ocp.solver_options.nlp_solver_max_iter = max(self.acados_max_iter, 1)
        ocp.solver_options.levenberg_marquardt = self.acados_levenberg_marquardt
        ocp.solver_options.regularize_method = self.acados_regularize_method
        ocp.solver_options.globalization = self.acados_globalization
        ocp.solver_options.globalization_alpha_min = 1e-4
        ocp.solver_options.globalization_alpha_reduction = 0.7
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.print_level = 0

        os.makedirs(self.acados_codegen_dir, exist_ok=True)
        codegen_name = (
            'robotx_dynamics_rk4_mpc_const_tau_v_slack_'
            f'tau_step_tau_r_bound_N{self.horizon_steps}_'
            f'{self.acados_nlp_solver_type.lower()}')
        code_dir = os.path.join(self.acados_codegen_dir, codegen_name)
        ocp.code_gen_opts.code_export_directory = code_dir
        ocp.code_gen_opts.json_file = os.path.join(
            self.acados_codegen_dir, f'{codegen_name}.json')
        return ocp

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def on_timer(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.latest_odom is None:
            self.publish_zero(FAIL_NO_ODOM)
            return

        stamp = stamp_to_sec(self.latest_odom.header.stamp)
        if stamp > 0.0 and abs(now - stamp) > self.command_timeout_sec:
            self.publish_zero(FAIL_NO_ODOM)
            return

        if self.last_control_time is None:
            self.last_control_time = now
            return
        dt = max(now - self.last_control_time, 0.0)
        if dt <= 0.0:
            return
        dt = min(dt, 0.5)
        self.last_control_time = now

        state = self.extract_state(self.latest_odom)
        if self.reference is None:
            if not self.create_reference(state, now):
                self.publish_zero(FAIL_NO_REFERENCE)
                return

        elapsed = max(now - self.reference_start_time, 0.0)
        solve = self.solve_mpc(state, elapsed)
        if not solve['success']:
            self.get_logger().warn(
                'MPC solve failed; publishing zero thrust. '
                f"reason={solve['message']}")
            self.publish_zero(FAIL_OPTIMIZER_FAILED, state, solve)
            return

        left_raw, right_raw = generalized_force_to_thrust(
            solve['tau'][0], solve['tau'][2], self.model_params)
        left_cmd, right_cmd = self.apply_thrust_limits(
            left_raw, right_raw, dt)
        tau_u_c, tau_r_c = thrust_to_generalized_force(
            left_cmd, right_cmd, self.model_params)

        self.publish_thrust(left_cmd, right_cmd)
        self.publish_command(left_cmd, right_cmd, solve['tau'])
        self.publish_debug(
            state, elapsed, solve, left_raw, right_raw, left_cmd, right_cmd,
            tau_u_c, tau_r_c, FAIL_OK)
        self.last_tau = solve['tau']

    def extract_state(self, odom: Odometry):
        z = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            yaw_from_quaternion(odom.pose.pose.orientation),
        ], dtype=float)
        z_dot = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.angular.z,
        ], dtype=float)
        return {
            'z': z,
            'z_dot': z_dot,
            'nu': flat_to_body_velocity(z, z_dot),
        }

    def create_reference(self, state, now: float) -> bool:
        try:
            self.reference = make_synthetic_reference(
                self.reference_name, state['z'], self.model_params,
                dt=self.reference_dt, initial_nu=state['nu'])
        except Exception as exc:
            self.get_logger().error(
                f'Unable to create synthetic reference: {exc}')
            self.reference = None
            return False
        self.reference_start_time = now
        self.get_logger().info(
            f'Created feasible reference [{self.reference.name}] '
            f'duration={self.reference.duration:.2f}s, '
            f'diagnostics={self.reference.diagnostics}')
        return True

    def solve_mpc(self, state, elapsed: float):
        if self.solver_backend == 'acados':
            if not self.acados_ready or self.acados_solver is None:
                return {
                    'success': False,
                    'message': 'acados backend unavailable',
                    'status': -1.0,
                    'backend': SOLVER_BACKEND_ACADOS,
                    'objective': float('inf'),
                    'solve_time_ms': 0.0,
                    'tau': np.zeros(3, dtype=float),
                    'tau_pred': np.zeros((1, 3), dtype=float),
                    'slack_pred': np.zeros(1, dtype=float),
                    'reference': state['z'],
                    'reference_dot': np.zeros(3, dtype=float),
                    'terminal_reference': state['z'],
                    'terminal_pos_error': 0.0,
                    'terminal_yaw_error': 0.0,
                    'scenario_complete': 0.0,
                }
            return self.solve_mpc_acados(state, elapsed)
        if self.solver_backend == 'slsqp':
            return self.solve_mpc_slsqp(state, elapsed)
        return {
            'success': False,
            'message': f'unknown backend {self.solver_backend}',
            'status': -2.0,
            'backend': -1.0,
            'objective': float('inf'),
            'solve_time_ms': 0.0,
            'tau': np.zeros(3, dtype=float),
            'tau_pred': np.zeros((1, 3), dtype=float),
            'slack_pred': np.zeros(1, dtype=float),
            'reference': state['z'],
            'reference_dot': np.zeros(3, dtype=float),
            'terminal_reference': state['z'],
            'terminal_pos_error': 0.0,
            'terminal_yaw_error': 0.0,
            'scenario_complete': 0.0,
        }

    def solve_mpc_slsqp(self, state, elapsed: float):
        if scipy_minimize is None:
            return {
                'success': False,
                'message': 'scipy.optimize unavailable',
                'status': -1.0,
                'backend': SOLVER_BACKEND_SLSQP,
                'objective': float('inf'),
                'solve_time_ms': 0.0,
                'tau': np.zeros(3, dtype=float),
                'tau_pred': np.zeros((1, 3), dtype=float),
                'slack_pred': np.zeros(1, dtype=float),
                'reference': state['z'],
                'reference_dot': np.zeros(3, dtype=float),
                'terminal_reference': state['z'],
                'terminal_pos_error': 0.0,
                'terminal_yaw_error': 0.0,
                'scenario_complete': 0.0,
            }
        horizon = self.reference.horizon_at_offsets(
            elapsed, self.horizon_offsets)
        initial_guess = self.initial_guess(horizon, state)
        bounds = self.decision_bounds(state)

        start_time = time.monotonic()

        def objective(decision):
            return self.mpc_cost(decision, state, horizon)

        def feasibility(decision):
            return self.path_margin(decision, state)

        result = scipy_minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': feasibility}],
            options={
                'maxiter': self.solver_max_iterations,
                'ftol': self.solver_ftol,
                'disp': False,
            })
        solve_time_ms = (time.monotonic() - start_time) * 1000.0

        decision = np.asarray(result.x, dtype=float)
        z_pred, z_dot_pred, z_ddot_pred, tau_pred, slack_pred = (
            self.rollout_prediction(decision, state))
        margins = self.path_margin(decision, state)
        feasible = bool(np.all(margins >= -1e-3))
        success = bool(
            result.success and feasible and np.all(np.isfinite(tau_pred)))
        if success:
            self.last_solution = decision
        tau0 = tau_pred[0] if len(tau_pred) else np.zeros(3, dtype=float)
        ref = horizon['z'][1]
        terminal_ref = horizon['z'][-1]
        terminal_pos_error = float(
            np.linalg.norm(terminal_ref[:2] - z_pred[-1, :2]))
        terminal_yaw_error = wrap_angle(float(terminal_ref[2] - z_pred[-1, 2]))

        return {
            'success': success,
            'message': str(result.message),
            'status': float(result.status),
            'backend': SOLVER_BACKEND_SLSQP,
            'objective': float(result.fun) if np.isfinite(result.fun) else float('inf'),
            'solve_time_ms': solve_time_ms,
            'decision': decision,
            'z_pred': z_pred,
            'z_dot_pred': z_dot_pred,
            'z_ddot_pred': z_ddot_pred,
            'tau_pred': tau_pred,
            'slack_pred': slack_pred,
            'tau': tau0,
            'reference': ref,
            'reference_dot': horizon['z_dot'][1],
            'terminal_reference': terminal_ref,
            'terminal_pos_error': terminal_pos_error,
            'terminal_yaw_error': terminal_yaw_error,
            'scenario_complete': 1.0 if elapsed >= self.reference.duration else 0.0,
        }

    def solve_mpc_acados(self, state, elapsed: float):
        horizon = self.reference.horizon_at_offsets(
            elapsed, self.horizon_offsets)
        start_time = time.monotonic()
        state_vector = np.concatenate([state['z'], state['z_dot']])
        state_vector[2] = wrap_angle(float(state_vector[2]))
        initial_lbx = np.array([
            *state_vector,
            0.0,
            self.last_tau[0],
            self.last_tau[2],
        ], dtype=float)
        initial_ubx = np.array([
            *state_vector,
            self.tau_v_slack_max,
            self.last_tau[0],
            self.last_tau[2],
        ], dtype=float)

        final_result = None
        retried = False
        for attempt in range(2):
            if attempt:
                retried = True
                self.last_acados_x = None
                self.last_acados_u = None
                try:
                    self.acados_solver.reset()
                except Exception as exc:
                    self.get_logger().warn(
                        f'Unable to reset acados solver before retry: {exc}')

            self.acados_solver.set(0, 'lbx', initial_lbx)
            self.acados_solver.set(0, 'ubx', initial_ubx)
            self.set_acados_references(horizon)
            self.set_acados_initial_guess(horizon, state_vector)

            status = int(self.acados_solver.solve())
            final_result = self.extract_acados_solution(status, horizon, elapsed)
            if final_result['success']:
                break

        solve_time_ms = (time.monotonic() - start_time) * 1000.0
        if final_result is None:
            final_result = self.empty_acados_result(state, horizon)
        final_result['solve_time_ms'] = solve_time_ms
        if retried:
            final_result['message'] += ' after reset retry'
        return final_result

    def empty_acados_result(self, state, horizon):
        return {
            'success': False,
            'message': 'acados solve unavailable',
            'status': -1.0,
            'backend': SOLVER_BACKEND_ACADOS,
            'objective': float('inf'),
            'solve_time_ms': 0.0,
            'decision': np.zeros(6 + 3 * self.horizon_steps +
                                 1, dtype=float),
            'z_pred': np.asarray([state['z']], dtype=float),
            'z_dot_pred': np.asarray([state['z_dot']], dtype=float),
            'z_ddot_pred': np.zeros((1, 3), dtype=float),
            'tau_pred': np.zeros((1, 3), dtype=float),
            'slack_pred': np.zeros(1, dtype=float),
            'tau': np.zeros(3, dtype=float),
            'reference': horizon['z'][1],
            'reference_dot': horizon['z_dot'][1],
            'terminal_reference': horizon['z'][-1],
            'terminal_pos_error': 0.0,
            'terminal_yaw_error': 0.0,
            'scenario_complete': 0.0,
        }

    def extract_acados_solution(self, status: int, horizon, elapsed: float):
        x_pred = np.asarray([
            self.acados_solver.get(index, 'x')
            for index in range(self.horizon_steps + 1)
        ], dtype=float)
        u_pred = np.asarray([
            self.acados_solver.get(index, 'u')
            for index in range(self.horizon_steps)
        ], dtype=float)
        if not len(u_pred):
            u_pred = np.zeros((1, 3), dtype=float)

        z_pred = x_pred[:, :3]
        z_pred[:, 2] = np.asarray([
            wrap_angle(float(item)) for item in z_pred[:, 2]
        ])
        z_dot_pred = x_pred[:, 3:6]
        z_ddot_pred = np.zeros((self.horizon_steps + 1, 3), dtype=float)
        slack_pred = x_pred[:, 6]
        tau_pred = np.zeros((self.horizon_steps + 1, 3), dtype=float)
        tau_pred[:self.horizon_steps] = u_pred[:self.horizon_steps]
        if self.horizon_steps > 0:
            tau_pred[-1] = tau_pred[-2]

        constrained_tau = tau_pred[:self.horizon_steps]
        constrained_slack = slack_pred[:self.horizon_steps]
        lateral_violation = (
            float(np.max(np.maximum(
                np.abs(constrained_tau[:, 1]) - constrained_slack, 0.0)))
            if len(constrained_tau) else 0.0)
        thrust_violation = self.max_thrust_violation(constrained_tau)
        yaw_moment_violation = self.max_yaw_moment_violation(constrained_tau)
        acceptable_status = status in (0, 2)
        success = bool(
            acceptable_status and
            np.all(np.isfinite(tau_pred)) and
            lateral_violation <= self.constraint_violation_tolerance and
            thrust_violation <= self.constraint_violation_tolerance and
            yaw_moment_violation <= self.constraint_violation_tolerance)
        if success:
            self.last_acados_x = x_pred
            self.last_acados_u = u_pred
        tau0 = tau_pred[0] if len(u_pred) else np.zeros(3, dtype=float)
        ref = horizon['z'][1]
        terminal_ref = horizon['z'][-1]
        terminal_pos_error = float(
            np.linalg.norm(terminal_ref[:2] - z_pred[-1, :2]))
        terminal_yaw_error = wrap_angle(float(terminal_ref[2] - z_pred[-1, 2]))
        objective = self.evaluate_prediction_cost(
            z_pred, z_dot_pred, tau_pred, slack_pred, horizon)

        return {
            'success': success,
            'message': f'acados status {status}',
            'status': float(status),
            'backend': SOLVER_BACKEND_ACADOS,
            'objective': objective,
            'solve_time_ms': 0.0,
            'decision': np.concatenate([
                x_pred[0, :6],
                u_pred.reshape(-1),
                np.asarray([slack_pred[0]], dtype=float),
            ]),
            'z_pred': z_pred,
            'z_dot_pred': z_dot_pred,
            'z_ddot_pred': z_ddot_pred,
            'tau_pred': tau_pred,
            'slack_pred': slack_pred,
            'tau': tau0,
            'reference': ref,
            'reference_dot': horizon['z_dot'][1],
            'terminal_reference': terminal_ref,
            'terminal_pos_error': terminal_pos_error,
            'terminal_yaw_error': terminal_yaw_error,
            'scenario_complete': 1.0 if (
                self.reference is not None and
                elapsed >= self.reference.duration) else 0.0,
        }

    def max_thrust_violation(self, tau_pred) -> float:
        violation = 0.0
        for tau_step in np.asarray(tau_pred, dtype=float):
            left, right = generalized_force_to_thrust(
                tau_step[0], tau_step[2], self.model_params)
            violation = max(
                violation,
                self.model_params.min_thrust - left,
                left - self.model_params.max_thrust,
                self.model_params.min_thrust - right,
                right - self.model_params.max_thrust,
                0.0)
        return float(violation)

    def max_yaw_moment_violation(self, tau_pred) -> float:
        tau_pred = np.asarray(tau_pred, dtype=float)
        if not len(tau_pred):
            return 0.0
        return float(max(np.max(np.abs(tau_pred[:, 2])) -
                         self.max_yaw_moment, 0.0))

    def acados_stage_parameter(self, ref_z, ref_z_dot, weight_scale: float,
                               h_step: float = 1.0,
                               terminal: bool = False):
        params = np.zeros(15, dtype=float)
        params[0:3] = np.asarray(ref_z, dtype=float)
        params[3:6] = np.asarray(ref_z_dot, dtype=float)
        params[14] = float(h_step)
        if terminal:
            params[6] = self.weights['terminal_position']
            params[7] = self.weights['terminal_yaw']
            params[8] = self.weights['terminal_velocity']
            params[9] = self.weights['terminal_yaw_rate']
        else:
            params[6] = weight_scale * self.weights['position']
            params[7] = weight_scale * self.weights['yaw']
            params[8] = weight_scale * self.weights['velocity']
            params[9] = weight_scale * self.weights['yaw_rate']
            params[10] = weight_scale * self.weights['tau_u']
            params[11] = weight_scale * self.weights['tau_r']
            params[12] = weight_scale * self.weights['tau_v_slack']
            params[13] = weight_scale * self.weights['delta_tau']
        return params

    def set_acados_references(self, horizon):
        for index in range(self.horizon_steps):
            params = self.acados_stage_parameter(
                horizon['z'][index],
                horizon['z_dot'][index],
                float(self.horizon_dt_sequence[index]),
                h_step=float(self.horizon_dt_sequence[index]),
                terminal=False)
            self.acados_solver.set(index, 'p', params)
        terminal_params = self.acados_stage_parameter(
            horizon['z'][-1], horizon['z_dot'][-1], 1.0,
            h_step=1.0, terminal=True)
        self.acados_solver.set(self.horizon_steps, 'p', terminal_params)

    def set_acados_initial_guess(self, horizon, state_vector):
        if (
                self.last_acados_x is not None and
                self.last_acados_u is not None and
                len(self.last_acados_x) == self.horizon_steps + 1 and
                len(self.last_acados_u) == self.horizon_steps):
            x_guess = np.zeros_like(self.last_acados_x)
            u_guess = np.zeros_like(self.last_acados_u)
            x_guess[:-1] = self.last_acados_x[1:]
            x_guess[-1] = self.last_acados_x[-1]
            u_guess[:-1] = self.last_acados_u[1:]
            u_guess[-1] = self.last_acados_u[-1]
        else:
            x_guess = np.column_stack((
                np.asarray(horizon['z'], dtype=float),
                np.asarray(horizon['z_dot'], dtype=float)))
            u_guess = np.asarray(
                horizon['tau'][:self.horizon_steps], dtype=float).copy()
            slack_guess = max(
                float(np.max(np.abs(np.asarray(
                    horizon['tau'][:self.horizon_steps, 1], dtype=float)))),
                min(self.tau_v_slack_warm_start, self.tau_v_slack_max))
            slack_column = np.full(
                self.horizon_steps + 1,
                clamp(slack_guess, 0.0, self.tau_v_slack_max),
                dtype=float)
            prev_tau_u = np.empty(self.horizon_steps + 1, dtype=float)
            prev_tau_r = np.empty(self.horizon_steps + 1, dtype=float)
            prev_tau_u[0] = float(self.last_tau[0])
            prev_tau_r[0] = float(self.last_tau[2])
            prev_tau_u[1:] = u_guess[:, 0]
            prev_tau_r[1:] = u_guess[:, 2]
            x_guess = np.column_stack((
                x_guess, slack_column, prev_tau_u, prev_tau_r))
        x_guess[0, :6] = state_vector
        x_guess[0, 7] = float(self.last_tau[0])
        x_guess[0, 8] = float(self.last_tau[2])
        for index in range(self.horizon_steps + 1):
            x_guess[index, 2] = wrap_angle(float(x_guess[index, 2]))
            x_guess[index, 6] = clamp(
                float(x_guess[index, 6]), 0.0, self.tau_v_slack_max)
            self.acados_solver.set(index, 'x', x_guess[index])
        for index in range(self.horizon_steps):
            u_guess[index, 0] = clamp(
                float(u_guess[index, 0]),
                2.0 * self.model_params.min_thrust,
                2.0 * self.model_params.max_thrust)
            u_guess[index, 1] = clamp(
                float(u_guess[index, 1]),
                -self.tau_v_slack_max, self.tau_v_slack_max)
            u_guess[index, 2] = clamp(
                float(u_guess[index, 2]),
                -self.max_yaw_moment, self.max_yaw_moment)
            self.acados_solver.set(index, 'u', u_guess[index])

    def evaluate_prediction_cost(self, z, z_dot, tau, slack, horizon) -> float:
        cost = 0.0
        for index in range(self.horizon_steps):
            ref_z = horizon['z'][index]
            pos_error = z[index, :2] - ref_z[:2]
            yaw_error = wrap_angle(float(z[index, 2] - ref_z[2]))
            vel_error = z_dot[index, :2] - horizon['z_dot'][index, :2]
            yaw_rate_error = z_dot[index, 2] - horizon['z_dot'][index, 2]
            tau_step = tau[index]
            h = self.horizon_dt_sequence[index]
            cost += h * self.weights['position'] * float(pos_error @ pos_error)
            cost += h * self.weights['yaw'] * yaw_error * yaw_error
            cost += h * self.weights['velocity'] * float(vel_error @ vel_error)
            cost += h * self.weights['yaw_rate'] * yaw_rate_error * yaw_rate_error
            cost += h * self.weights['tau_u'] * tau_step[0] * tau_step[0]
            cost += h * self.weights['tau_r'] * tau_step[2] * tau_step[2]
            if index == 0:
                prev_tau_u = self.last_tau[0]
                prev_tau_r = self.last_tau[2]
            else:
                prev_tau_u = tau[index - 1, 0]
                prev_tau_r = tau[index - 1, 2]
            cost += h * self.weights['delta_tau'] * (
                (tau_step[0] - prev_tau_u)**2 +
                (tau_step[2] - prev_tau_r)**2)
            cost += h * self.weights['tau_v_slack'] * slack[index] * slack[index]

        terminal_z = horizon['z'][-1]
        terminal_pos_error = z[-1, :2] - terminal_z[:2]
        terminal_yaw_error = wrap_angle(float(z[-1, 2] - terminal_z[2]))
        terminal_vel_error = z_dot[-1, :2] - horizon['z_dot'][-1, :2]
        terminal_yaw_rate_error = z_dot[-1, 2] - horizon['z_dot'][-1, 2]
        cost += self.weights['terminal_position'] * float(
            terminal_pos_error @ terminal_pos_error)
        cost += self.weights['terminal_yaw'] * terminal_yaw_error * terminal_yaw_error
        cost += self.weights['terminal_velocity'] * float(
            terminal_vel_error @ terminal_vel_error)
        cost += (
            self.weights['terminal_yaw_rate'] *
            terminal_yaw_rate_error * terminal_yaw_rate_error)
        return float(cost)

    def decision_size(self):
        n = self.horizon_steps
        return 6 + 3 * (n + 1) + 1

    def pack_decision(self, z0, z_dot0, z_ddot, slack):
        return np.concatenate([
            np.asarray(z0, dtype=float).reshape(3),
            np.asarray(z_dot0, dtype=float).reshape(3),
            np.asarray(z_ddot, dtype=float).reshape(-1),
            np.asarray(slack, dtype=float).reshape(-1),
        ])

    def unpack_decision(self, decision):
        n = self.horizon_steps
        decision = np.asarray(decision, dtype=float)
        z0 = decision[0:3]
        z_dot0 = decision[3:6]
        z_ddot_start = 6
        z_ddot_stop = z_ddot_start + 3 * (n + 1)
        z_ddot = decision[z_ddot_start:z_ddot_stop].reshape(n + 1, 3)
        slack = float(decision[z_ddot_stop])
        return z0, z_dot0, z_ddot, slack

    def initial_guess(self, horizon, state):
        n = self.horizon_steps
        if (
                self.last_solution is not None and
                len(self.last_solution) == self.decision_size()):
            _, _, last_z_ddot, last_slack = self.unpack_decision(
                self.last_solution)
            z_ddot = np.zeros_like(last_z_ddot)
            z_ddot[:-1] = last_z_ddot[1:]
            z_ddot[-1] = last_z_ddot[-1]
            slack = float(last_slack)
        else:
            z_ddot = np.asarray(horizon['z_ddot'][:n + 1], dtype=float)
            slack = max(
                float(np.max(np.abs(np.asarray(
                    horizon['tau'][:n + 1, 1], dtype=float)))),
                min(self.tau_v_slack_warm_start, self.tau_v_slack_max))
        return self.pack_decision(
            state['z'], state['z_dot'], z_ddot, slack)

    def decision_bounds(self, state):
        bounds = []
        state_vector = np.concatenate([state['z'], state['z_dot']])
        for value in state_vector:
            value = float(value)
            bounds.append((value, value))
        for _ in range(self.horizon_steps + 1):
            bounds.extend([
                (-self.accel_bound_xy, self.accel_bound_xy),
                (-self.accel_bound_xy, self.accel_bound_xy),
                (-self.yaw_accel_bound, self.yaw_accel_bound),
            ])
        bounds.append((0.0, self.tau_v_slack_max))
        return bounds

    def rollout_prediction(self, decision, state):
        n = self.horizon_steps
        z = np.zeros((n + 1, 3), dtype=float)
        z_dot = np.zeros((n + 1, 3), dtype=float)
        z0, z_dot0, z_ddot, slack = self.unpack_decision(decision)
        tau = np.zeros((n + 1, 3), dtype=float)
        z[0] = z0
        z[0, 2] = wrap_angle(float(z[0, 2]))
        z_dot[0] = z_dot0
        slack_pred = np.full(n + 1, float(slack), dtype=float)
        for index in range(n):
            h = self.horizon_dt_sequence[index]
            z[index + 1] = (
                z[index] + h * z_dot[index] +
                h * h * (
                    (1.0 / 3.0) * z_ddot[index] +
                    (1.0 / 6.0) * z_ddot[index + 1]))
            z[index + 1, 2] = wrap_angle(z[index + 1, 2])
            z_dot[index + 1] = (
                z_dot[index] +
                0.5 * h * (z_ddot[index] + z_ddot[index + 1]))
        for index in range(n + 1):
            tau[index] = theta_tau_nominal(
                z[index], z_dot[index], z_ddot[index], self.model_params)
        return z, z_dot, z_ddot, tau, slack_pred

    def mpc_cost(self, decision, state, horizon) -> float:
        z, z_dot, _z_ddot, tau, slack = self.rollout_prediction(
            decision, state)
        cost = 0.0
        prev_tau_ur = self.last_tau[[0, 2]]
        for index in range(self.horizon_steps + 1):
            ref_z = horizon['z'][index]
            ref_nu = horizon['nu'][index]
            nu = flat_to_body_velocity(z[index], z_dot[index])
            pos_error = z[index, :2] - ref_z[:2]
            yaw_error = wrap_angle(float(z[index, 2] - ref_z[2]))
            vel_error = nu[:2] - ref_nu[:2]
            yaw_rate_error = nu[2] - ref_nu[2]
            tau_step = tau[index]
            tau_ur = tau_step[[0, 2]]
            node_cost = 0.0
            node_cost += self.weights['position'] * float(pos_error @ pos_error)
            node_cost += self.weights['yaw'] * yaw_error * yaw_error
            node_cost += self.weights['velocity'] * float(vel_error @ vel_error)
            node_cost += self.weights['yaw_rate'] * yaw_rate_error * yaw_rate_error
            node_cost += self.weights['tau_u'] * tau_step[0] * tau_step[0]
            node_cost += self.weights['tau_r'] * tau_step[2] * tau_step[2]
            node_cost += self.weights['tau_v_slack'] * slack[index] * slack[index]
            cost += self.horizon_node_weights[index] * node_cost
            delta_tau_ur = tau_ur - prev_tau_ur
            cost += (
                self.horizon_node_weights[index] *
                self.weights['delta_tau'] *
                float(delta_tau_ur @ delta_tau_ur))
            prev_tau_ur = tau_ur

        terminal_z = horizon['z'][-1]
        terminal_nu = horizon['nu'][-1]
        predicted_terminal_nu = flat_to_body_velocity(z[-1], z_dot[-1])
        terminal_pos_error = z[-1, :2] - terminal_z[:2]
        terminal_yaw_error = wrap_angle(float(z[-1, 2] - terminal_z[2]))
        terminal_vel_error = predicted_terminal_nu[:2] - terminal_nu[:2]
        terminal_yaw_rate_error = predicted_terminal_nu[2] - terminal_nu[2]
        cost += self.weights['terminal_position'] * float(
            terminal_pos_error @ terminal_pos_error)
        cost += self.weights['terminal_yaw'] * terminal_yaw_error * terminal_yaw_error
        cost += self.weights['terminal_velocity'] * float(
            terminal_vel_error @ terminal_vel_error)
        cost += (
            self.weights['terminal_yaw_rate'] *
            terminal_yaw_rate_error * terminal_yaw_rate_error)
        return float(cost)

    def initial_condition_residual(self, decision, state):
        z0, z_dot0, _z_ddot, _slack = self.unpack_decision(decision)
        residual = np.concatenate([z0 - state['z'], z_dot0 - state['z_dot']])
        residual[2] = wrap_angle(float(z0[2] - state['z'][2]))
        return residual

    def path_margin(self, decision, state):
        return np.concatenate([
            self.thrust_margin(decision, state),
            self.yaw_moment_margin(decision, state),
            self.lateral_margin(decision, state),
        ])

    def thrust_margin(self, decision, state):
        _, _, _, tau, _ = self.rollout_prediction(decision, state)
        margins = []
        for tau_step in tau:
            left, right = generalized_force_to_thrust(
                tau_step[0], tau_step[2], self.model_params)
            margins.extend([
                left - self.model_params.min_thrust,
                self.model_params.max_thrust - left,
                right - self.model_params.min_thrust,
                self.model_params.max_thrust - right,
            ])
        return np.asarray(margins, dtype=float)

    def yaw_moment_margin(self, decision, state):
        _, _, _, tau, _ = self.rollout_prediction(decision, state)
        margins = []
        for tau_step in tau:
            margins.extend([
                tau_step[2] + self.max_yaw_moment,
                self.max_yaw_moment - tau_step[2],
            ])
        return np.asarray(margins, dtype=float)

    def lateral_margin(self, decision, state):
        _, _, _, tau, slack = self.rollout_prediction(decision, state)
        margins = []
        for tau_step, slack_step in zip(tau, slack):
            margins.extend([
                slack_step - tau_step[1],
                slack_step + tau_step[1],
                slack_step,
            ])
        return np.asarray(margins, dtype=float)

    def apply_thrust_limits(self, left: float, right: float, dt: float):
        left = clamp(left, self.model_params.min_thrust, self.model_params.max_thrust)
        right = clamp(right, self.model_params.min_thrust, self.model_params.max_thrust)
        max_delta = self.model_params.thrust_rate_limit * max(dt, 0.0)
        left = clamp(left, self.prev_left_thrust - max_delta,
                     self.prev_left_thrust + max_delta)
        right = clamp(right, self.prev_right_thrust - max_delta,
                      self.prev_right_thrust + max_delta)
        self.prev_left_thrust = left
        self.prev_right_thrust = right
        return left, right

    def publish_zero(self, fail_reason: float, state=None, solve=None):
        self.prev_left_thrust = 0.0
        self.prev_right_thrust = 0.0
        self.publish_thrust(0.0, 0.0)
        self.publish_command(0.0, 0.0, np.zeros(3, dtype=float))
        if state is not None:
            elapsed = 0.0
            if self.reference_start_time is not None:
                elapsed = (
                    self.get_clock().now().nanoseconds * 1e-9 -
                    self.reference_start_time)
            if solve is None:
                solve = {
                    'success': False,
                    'status': 0.0,
                    'backend': (
                        SOLVER_BACKEND_ACADOS
                        if self.solver_backend == 'acados'
                        else SOLVER_BACKEND_SLSQP),
                    'objective': float('nan'),
                    'solve_time_ms': 0.0,
                    'tau': np.zeros(3, dtype=float),
                    'tau_pred': np.zeros((1, 3), dtype=float),
                    'slack_pred': np.zeros(1, dtype=float),
                    'reference': state['z'],
                    'reference_dot': np.zeros(3, dtype=float),
                    'terminal_reference': state['z'],
                    'terminal_pos_error': 0.0,
                    'terminal_yaw_error': 0.0,
                    'scenario_complete': 0.0,
                }
            self.publish_debug(
                state, elapsed, solve, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, fail_reason)

    def publish_thrust(self, left: float, right: float):
        left_msg = Float64()
        right_msg = Float64()
        left_msg.data = float(left)
        right_msg.data = float(right)
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    def publish_command(self, left: float, right: float, tau):
        msg = Float64MultiArray()
        msg.data = [
            float(left), float(right),
            float(tau[0]), float(tau[1]), float(tau[2]),
        ]
        self.command_pub.publish(msg)

    def publish_debug(self, state, elapsed: float, solve,
                      left_raw: float, right_raw: float,
                      left_cmd: float, right_cmd: float,
                      tau_u_c: float, tau_r_c: float,
                      fail_reason: float):
        tau = np.asarray(solve['tau'], dtype=float)
        tau_pred = np.asarray(solve.get('tau_pred', np.zeros((1, 3))), dtype=float)
        slack_pred = np.asarray(
            solve.get('slack_pred', np.zeros(len(tau_pred))), dtype=float)
        if (
                float(solve.get('backend', -1.0)) == SOLVER_BACKEND_ACADOS and
                len(tau_pred) > 1 and len(slack_pred) > 1):
            tau_limit_pred = tau_pred[:-1]
            slack_limit_pred = slack_pred[:-1]
        else:
            tau_limit_pred = tau_pred
            slack_limit_pred = slack_pred
        ref = np.asarray(solve['reference'], dtype=float)
        ref_dot = np.asarray(solve['reference_dot'], dtype=float)
        terminal_ref = np.asarray(solve['terminal_reference'], dtype=float)
        e = ref - state['z']
        e[2] = wrap_angle(float(e[2]))
        saturated = (
            abs(left_raw - left_cmd) > 1e-6 or
            abs(right_raw - right_cmd) > 1e-6)
        msg_dict = {
            'elapsed': float(elapsed),
            'reference_duration': self.reference.duration if self.reference else 0.0,
            'solver_success': 1.0 if solve['success'] else 0.0,
            'solver_backend': float(solve.get('backend', -1.0)),
            'solver_status': float(solve.get('status', 0.0)),
            'solve_time_ms': float(solve['solve_time_ms']),
            'objective': float(solve['objective']),
            'x_ref': float(ref[0]),
            'y_ref': float(ref[1]),
            'psi_ref': float(ref[2]),
            'x': float(state['z'][0]),
            'y': float(state['z'][1]),
            'psi': float(state['z'][2]),
            'e_x': float(e[0]),
            'e_y': float(e[1]),
            'e_psi': float(e[2]),
            'x_dot_ref': float(ref_dot[0]),
            'y_dot_ref': float(ref_dot[1]),
            'psi_dot_ref': float(ref_dot[2]),
            'x_dot': float(state['z_dot'][0]),
            'y_dot': float(state['z_dot'][1]),
            'psi_dot': float(state['z_dot'][2]),
            'tau_u': float(tau[0]),
            'tau_v': float(tau[1]),
            'tau_v_slack': float(slack_pred[0]) if len(slack_pred) else 0.0,
            'tau_r': float(tau[2]),
            'left_raw': float(left_raw),
            'right_raw': float(right_raw),
            'left_cmd': float(left_cmd),
            'right_cmd': float(right_cmd),
            'thrust_saturated': 1.0 if saturated else 0.0,
            'max_pred_tau_v': float(np.max(np.abs(tau_limit_pred[:, 1]))),
            'rms_pred_tau_v': float(np.sqrt(np.mean(tau_limit_pred[:, 1]**2))),
            'max_pred_tau_v_slack': (
                float(np.max(slack_limit_pred)) if len(slack_limit_pred) else 0.0),
            'rms_pred_tau_v_slack': (
                float(np.sqrt(np.mean(slack_limit_pred**2)))
                if len(slack_limit_pred) else 0.0),
            'max_pred_tau_v_violation': (
                float(np.max(np.maximum(
                    np.abs(tau_limit_pred[:, 1]) - slack_limit_pred, 0.0)))
                if len(slack_limit_pred) else 0.0),
            'max_pred_tau_u': float(np.max(np.abs(tau_limit_pred[:, 0]))),
            'max_pred_tau_r': float(np.max(np.abs(tau_limit_pred[:, 2]))),
            'max_pred_left_abs': self.max_predicted_thrust_abs(tau_limit_pred, 0),
            'max_pred_right_abs': self.max_predicted_thrust_abs(tau_limit_pred, 1),
            'terminal_pos_error': float(solve['terminal_pos_error']),
            'terminal_yaw_error': float(solve['terminal_yaw_error']),
            'terminal_x_ref': float(terminal_ref[0]),
            'terminal_y_ref': float(terminal_ref[1]),
            'terminal_psi_ref': float(terminal_ref[2]),
            'fail_reason': float(fail_reason),
            'allocation_tau_u_residual': float(tau[0] - tau_u_c),
            'allocation_tau_r_residual': float(tau[2] - tau_r_c),
            'scenario_complete': float(solve['scenario_complete']),
        }
        self.last_debug = msg_dict
        msg = Float64MultiArray()
        msg.data = [msg_dict[field] for field in DEBUG_FIELDS]
        self.debug_pub.publish(msg)

    def max_predicted_thrust_abs(self, tau_pred, side: int) -> float:
        values = []
        for tau_step in tau_pred:
            left, right = generalized_force_to_thrust(
                tau_step[0], tau_step[2], self.model_params)
            values.append(abs(left if side == 0 else right))
        return float(max(values)) if values else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = FlatnessMpcController()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.publish_thrust(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
