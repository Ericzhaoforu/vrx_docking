#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robotx_safe_docking_control.flatness_mpc_controller import DEBUG_FIELDS
from robotx_safe_docking_control.flatness_mpc_controller import stamp_to_sec
from robotx_safe_docking_control.flatness_mpc_controller import yaw_from_quaternion
from robotx_safe_docking_control.usv_lattice_frontend import parse_circular_obstacles
from robotx_safe_docking_control.usv_flatness import wrap_angle


class FlatnessMpcRecorder(Node):
    def __init__(
            self, output_dir: Path, duration: float, obstacles=(),
            global_goal=None):
        super().__init__('flatness_mpc_performance_recorder')
        self.output_dir = output_dir
        self.duration = duration
        self.obstacles = tuple(obstacles)
        self.global_goal = (
            np.asarray(global_goal, dtype=float).reshape(3)
            if global_goal is not None else None)
        self.start_time = None
        self.latest_relative_time = None
        self.latest_debug = {}
        self.latest_command = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.latest_estimate_xy = None
        self.latest_truth_xy = None
        self.truth_to_estimate_offset = None
        self.samples = []
        self.done = False

        self.create_subscription(
            Odometry, '/safe_docking/odometry', self.on_estimate, 10)
        self.create_subscription(
            Odometry, '/wamv/sensors/position/ground_truth_odometry',
            self.on_truth, 10)
        self.create_subscription(
            Float64MultiArray, '/safe_docking/flatness_mpc/command',
            self.on_command, 10)
        self.create_subscription(
            Float64MultiArray, '/safe_docking/flatness_mpc/debug',
            self.on_debug, 10)
        self.create_timer(0.2, self.on_timer)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def rel_time(self):
        now = self.now_sec()
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time
        self.latest_relative_time = t
        return t

    def rel_time_from_stamp(self, stamp_sec: float):
        stamp_sec = float(stamp_sec)
        if math.isfinite(stamp_sec) and stamp_sec > 0.0:
            if self.start_time is None:
                self.start_time = stamp_sec
            t = stamp_sec - self.start_time
            self.latest_relative_time = t
            return t
        return self.rel_time()

    def on_command(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) >= 5:
            self.latest_command = [float(value) for value in data[:5]]

    def on_debug(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < len(DEBUG_FIELDS):
            return
        self.latest_debug = {
            field: float(data[index])
            for index, field in enumerate(DEBUG_FIELDS)
        }
        t = self.rel_time_from_stamp(
            self.latest_debug.get('controller_time', float('nan')))
        self.samples.append(self.make_reference_sample(t))

    def on_estimate(self, msg: Odometry):
        t = self.rel_time_from_stamp(stamp_to_sec(msg.header.stamp))
        z, z_dot = self.extract_state(msg)
        self.latest_estimate_xy = z[:2]
        self.try_initialize_truth_alignment()
        self.samples.append(self.make_sample(
            t, 'estimate', z, z_dot))

    def on_truth(self, msg: Odometry):
        t = self.rel_time_from_stamp(stamp_to_sec(msg.header.stamp))
        z, z_dot = self.extract_state(msg)
        self.latest_truth_xy = z[:2]
        self.try_initialize_truth_alignment()
        if self.truth_to_estimate_offset is None:
            return
        z = z.copy()
        z[:2] += self.truth_to_estimate_offset
        self.samples.append(self.make_sample(
            t, 'truth', z, z_dot))

    def extract_state(self, msg: Odometry):
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(msg.pose.pose.orientation),
        ], dtype=float)
        z_dot = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z,
        ], dtype=float)
        return z, z_dot

    def make_sample(self, t: float, source: str, z, z_dot):
        debug = self.latest_debug
        x_ref = debug.get('active_x_ref', debug.get('x_ref', float('nan')))
        y_ref = debug.get('active_y_ref', debug.get('y_ref', float('nan')))
        psi_ref = debug.get('active_psi_ref', debug.get('psi_ref', float('nan')))
        terminal_x = debug.get('terminal_x_ref', float('nan'))
        terminal_y = debug.get('terminal_y_ref', float('nan'))
        terminal_psi = debug.get('terminal_psi_ref', float('nan'))
        ref_distance = math.hypot(x_ref - z[0], y_ref - z[1])
        terminal_distance = math.hypot(terminal_x - z[0], terminal_y - z[1])
        psi_ref_error = (
            wrap_angle(psi_ref - z[2]) if not math.isnan(psi_ref)
            else float('nan'))
        terminal_yaw_error = (
            wrap_angle(terminal_psi - z[2]) if not math.isnan(terminal_psi)
            else float('nan'))
        speed = math.hypot(z_dot[0], z_dot[1])
        return [
            t, source,
            z[0], z[1], z[2],
            z_dot[0], z_dot[1], z_dot[2],
            speed,
            ref_distance, psi_ref_error,
            terminal_distance, terminal_yaw_error,
            *self.latest_command,
            *[debug.get(field, float('nan')) for field in DEBUG_FIELDS],
        ]

    def make_reference_sample(self, t: float):
        debug = self.latest_debug
        z = np.array([
            debug.get('active_x_ref', debug.get('x_ref', float('nan'))),
            debug.get('active_y_ref', debug.get('y_ref', float('nan'))),
            debug.get('active_psi_ref', debug.get('psi_ref', float('nan'))),
        ], dtype=float)
        z_dot = np.array([
            debug.get('active_x_dot_ref',
                      debug.get('x_dot_ref', float('nan'))),
            debug.get('active_y_dot_ref',
                      debug.get('y_dot_ref', float('nan'))),
            debug.get('active_psi_dot_ref',
                      debug.get('psi_dot_ref', float('nan'))),
        ], dtype=float)
        speed = math.hypot(z_dot[0], z_dot[1])
        return [
            t, 'reference',
            z[0], z[1], z[2],
            z_dot[0], z_dot[1], z_dot[2],
            speed,
            float('nan'), float('nan'),
            float('nan'), float('nan'),
            *self.latest_command,
            *[debug.get(field, float('nan')) for field in DEBUG_FIELDS],
        ]

    def try_initialize_truth_alignment(self):
        if self.truth_to_estimate_offset is not None:
            return
        if self.latest_estimate_xy is None or self.latest_truth_xy is None:
            return
        self.truth_to_estimate_offset = (
            self.latest_estimate_xy - self.latest_truth_xy)
        self.get_logger().info(
            'Aligned ground truth into EKF local frame with offset: '
            f'x={self.truth_to_estimate_offset[0]:.3f} m, '
            f'y={self.truth_to_estimate_offset[1]:.3f} m')

    def on_timer(self):
        if self.latest_relative_time is None:
            return
        if self.latest_relative_time >= self.duration and not self.done:
            self.write_outputs()
            self.done = True

    def write_outputs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            self.get_logger().error('No MPC samples recorded.')
            return

        csv_path = self.output_dir / 'flatness_mpc_performance.csv'
        header = [
            't', 'source',
            'x', 'y', 'psi',
            'x_dot', 'y_dot', 'psi_dot',
            'speed',
            'ref_distance', 'psi_ref_error',
            'terminal_distance', 'terminal_yaw_error',
            'left_cmd', 'right_cmd', 'tau_u_cmd', 'tau_v_cmd', 'tau_r_cmd',
            *DEBUG_FIELDS,
        ]
        with csv_path.open('w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(self.samples)

        metrics = self.compute_metrics(header)
        metrics['simulated_obstacle_count'] = len(self.obstacles)
        metrics['simulated_obstacles'] = [
            {'x': float(item.x), 'y': float(item.y), 'radius': float(item.radius)}
            for item in self.obstacles
        ]
        with (self.output_dir / 'metrics.json').open('w') as metrics_file:
            json.dump(metrics, metrics_file, indent=2, sort_keys=True)
        self.plot(header)
        self.get_logger().info(f'MPC metrics: {json.dumps(metrics, sort_keys=True)}')

    def data_for(self, header, source: str):
        rows = [row for row in self.samples if row[1] == source]
        if not rows:
            return np.empty((0, len(header)), dtype=object)
        return np.asarray(rows, dtype=object)

    def column(self, data, header, name: str):
        if len(data) == 0:
            return np.asarray([], dtype=float)
        index = header.index(name)
        return np.asarray(data[:, index], dtype=float)

    def optional_column(self, data, header, name: str):
        if len(data) == 0:
            return np.asarray([], dtype=float)
        if name not in header:
            return np.full(len(data), np.nan, dtype=float)
        return self.column(data, header, name)

    def first_available_column(self, data, header, names):
        for name in names:
            if name in header:
                return self.column(data, header, name)
        return np.full(len(data), np.nan, dtype=float)

    def finite(self, values):
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]

    def compute_metrics(self, header):
        metrics = {}
        for source in ['estimate', 'truth']:
            data = self.data_for(header, source)
            if len(data) == 0:
                continue
            ref_dist = self.finite(self.column(data, header, 'ref_distance'))
            terminal_dist = self.finite(
                self.column(data, header, 'terminal_distance'))
            ref_yaw = self.finite(self.column(data, header, 'psi_ref_error'))
            terminal_yaw = self.finite(
                self.column(data, header, 'terminal_yaw_error'))
            speed = self.finite(self.column(data, header, 'speed'))
            metrics[f'{source}_samples'] = int(len(data))
            if len(ref_dist):
                metrics[f'{source}_rms_ref_distance_m'] = float(
                    math.sqrt(np.mean(ref_dist**2)))
            if len(ref_yaw):
                metrics[f'{source}_rms_ref_yaw_error_rad'] = float(
                    math.sqrt(np.mean(ref_yaw**2)))
            if len(terminal_dist):
                metrics[f'{source}_final_terminal_distance_m'] = float(
                    terminal_dist[-1])
            if len(terminal_yaw):
                metrics[f'{source}_final_terminal_yaw_error_rad'] = float(
                    abs(terminal_yaw[-1]))
            if len(speed):
                metrics[f'{source}_final_speed_mps'] = float(speed[-1])
            if self.global_goal is not None:
                x = self.finite(self.column(data, header, 'x'))
                y = self.finite(self.column(data, header, 'y'))
                psi = self.finite(self.column(data, header, 'psi'))
                if len(x) and len(y):
                    metrics[f'{source}_final_global_goal_distance_m'] = float(
                        math.hypot(
                            self.global_goal[0] - x[-1],
                            self.global_goal[1] - y[-1]))
                if len(psi):
                    metrics[f'{source}_final_global_goal_yaw_error_rad'] = float(
                        abs(wrap_angle(self.global_goal[2] - psi[-1])))

        debug_data = self.data_for(header, 'reference')
        if len(debug_data) == 0:
            debug_data = self.data_for(header, 'estimate')
        if len(debug_data):
            success = self.finite(
                self.column(debug_data, header, 'solver_success'))
            solve_ms = self.finite(
                self.column(debug_data, header, 'solve_time_ms'))
            tau_v = self.finite(self.column(debug_data, header, 'tau_v'))
            tau_v_slack = self.finite(
                self.column(debug_data, header, 'tau_v_slack'))
            max_pred_tau_v = self.finite(
                self.column(debug_data, header, 'max_pred_tau_v'))
            max_pred_tau_v_slack = self.finite(
                self.column(debug_data, header, 'max_pred_tau_v_slack'))
            max_pred_tau_v_violation = self.finite(
                self.column(debug_data, header, 'max_pred_tau_v_violation'))
            reference_candidates = self.finite(
                self.column(debug_data, header, 'reference_candidate_count'))
            reference_swaps = self.finite(
                self.column(debug_data, header, 'reference_swap_count'))
            reference_rejects = self.finite(
                self.column(debug_data, header, 'reference_reject_count'))
            swap_failures = self.finite(
                self.column(
                    debug_data, header,
                    'reference_solver_failure_after_swap_count'))
            swap_jump = self.finite(
                self.column(debug_data, header, 'reference_max_swap_thrust_jump'))
            accepted_tau_v = self.finite(
                self.column(debug_data, header, 'reference_max_abs_tau_v'))
            accepted_tau_v_rms = self.finite(
                self.column(debug_data, header, 'reference_rms_tau_v'))
            accepted_start_z = self.finite(
                self.column(debug_data, header, 'reference_start_z_error'))
            accepted_start_z_dot = self.finite(
                self.column(debug_data, header, 'reference_start_z_dot_error'))
            left = self.finite(self.column(debug_data, header, 'left_cmd'))
            right = self.finite(self.column(debug_data, header, 'right_cmd'))
            saturation = self.finite(
                self.column(debug_data, header, 'thrust_saturated'))
            if len(success):
                metrics['solver_success_rate'] = float(np.mean(success > 0.5))
            if len(solve_ms):
                metrics['solve_time_mean_ms'] = float(np.mean(solve_ms))
                metrics['solve_time_max_ms'] = float(np.max(solve_ms))
            if len(tau_v):
                metrics['rms_tau_v_n'] = float(math.sqrt(np.mean(tau_v**2)))
                metrics['max_abs_tau_v_n'] = float(np.max(np.abs(tau_v)))
            if len(tau_v_slack):
                metrics['rms_tau_v_slack_n'] = float(
                    math.sqrt(np.mean(tau_v_slack**2)))
                metrics['max_tau_v_slack_n'] = float(np.max(tau_v_slack))
            if len(max_pred_tau_v):
                metrics['max_pred_abs_tau_v_n'] = float(np.max(max_pred_tau_v))
            if len(max_pred_tau_v_slack):
                metrics['max_pred_tau_v_slack_n'] = float(
                    np.max(max_pred_tau_v_slack))
            if len(max_pred_tau_v_violation):
                metrics['max_pred_tau_v_violation_n'] = float(
                    np.max(max_pred_tau_v_violation))
            if len(reference_candidates):
                metrics['reference_candidate_count'] = int(
                    round(reference_candidates[-1]))
            if len(reference_swaps) and np.max(reference_swaps) > 0.5:
                metrics['reference_swap_count'] = int(round(reference_swaps[-1]))
            else:
                metrics['reference_swap_count'] = len(
                    self.detect_reference_elapsed_resets(header, debug_data))
            if len(reference_rejects):
                metrics['reference_rejected_candidate_count'] = int(
                    round(reference_rejects[-1]))
            if len(swap_failures):
                metrics['reference_solver_failure_after_swap_count'] = int(
                    round(swap_failures[-1]))
            if len(swap_jump):
                metrics['max_reference_swap_thrust_jump_n'] = float(
                    np.max(swap_jump))
            if len(accepted_tau_v):
                metrics['accepted_reference_max_abs_tau_v_n'] = float(
                    np.max(accepted_tau_v))
            if len(accepted_tau_v_rms):
                metrics['accepted_reference_max_rms_tau_v_n'] = float(
                    np.max(accepted_tau_v_rms))
            if len(accepted_start_z):
                metrics['accepted_reference_max_start_z_error'] = float(
                    np.max(accepted_start_z))
            if len(accepted_start_z_dot):
                metrics['accepted_reference_max_start_z_dot_error'] = float(
                    np.max(accepted_start_z_dot))
            if len(left) and len(right):
                metrics['max_abs_thrust_cmd_n'] = float(
                    max(np.max(np.abs(left)), np.max(np.abs(right))))
            if len(saturation):
                metrics['thrust_saturation_fraction'] = float(
                    np.mean(saturation > 0.5))
        return metrics

    def plot(self, header):
        estimate = self.data_for(header, 'estimate')
        truth = self.data_for(header, 'truth')
        reference = self.data_for(header, 'reference')
        if len(estimate) == 0:
            return
        accepted_replans, rejected_replans = self.replan_event_times(
            header, reference)

        def c(data, name):
            return self.column(data, header, name)

        fig, ax = plt.subplots(figsize=(9, 8))
        self.plot_xy_context(
            ax, header, estimate, truth, reference,
            include_global_goal=True)
        fig.tight_layout()
        fig.savefig(self.output_dir / '01_trajectory_xy.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 8))
        self.plot_xy_context(
            ax, header, estimate, truth, reference,
            include_global_goal=False)
        self.set_local_xy_limits(ax, header, estimate, truth, reference)
        fig.tight_layout()
        fig.savefig(self.output_dir / '01a_trajectory_xy_zoom.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for axis, name, ylabel in [
                (axes[0], 'x', 'x [m]'),
                (axes[1], 'y', 'y [m]'),
                (axes[2], 'psi', 'psi [rad]')]:
            axis.plot(c(estimate, 't'), c(estimate, name), label='EKF')
            if len(reference):
                axis.plot(
                    c(reference, 't'), c(reference, name), '--',
                    label='active reference')
            else:
                axis.plot(
                    c(estimate, 't'),
                    self.first_available_column(
                        estimate, header, [f'active_{name}_ref', f'{name}_ref']),
                    '--', label='active reference')
            if len(truth):
                axis.plot(c(truth, 't'), c(truth, name), label='truth')
            axis.set_ylabel(ylabel)
            axis.grid(True)
        self.draw_replan_event_lines(axes, accepted_replans)
        axes[-1].set_xlabel('time [s]')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / '02_states.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        t = c(estimate, 't')
        axes[0].plot(t, c(estimate, 'speed'), label='EKF speed')
        if len(reference):
            axes[0].plot(
                c(reference, 't'), c(reference, 'speed'), '--',
                label='active reference speed')
        axes[0].set_ylabel('speed [m/s]')
        axes[1].plot(t, c(estimate, 'psi_dot'), label='yaw rate')
        axes[1].plot(t, c(estimate, 'psi_dot_ref'), '--', label='yaw rate ref')
        axes[1].set_ylabel('yaw rate [rad/s]')
        self.draw_replan_event_lines(axes, accepted_replans, rejected_replans)
        for axis in axes:
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(self.output_dir / '03_velocities.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        u_est, v_est = self.body_velocity_from_local(
            c(estimate, 'x_dot'), c(estimate, 'y_dot'), c(estimate, 'psi'))
        axes[0].plot(t, u_est, label='EKF u')
        axes[1].plot(t, v_est, label='EKF v')
        axes[2].plot(t, c(estimate, 'x_dot'), label='EKF x_dot')
        axes[3].plot(t, c(estimate, 'y_dot'), label='EKF y_dot')
        if len(reference):
            t_ref = c(reference, 't')
            u_ref, v_ref = self.body_velocity_from_local(
                c(reference, 'x_dot'), c(reference, 'y_dot'),
                c(reference, 'psi'))
            axes[0].plot(t_ref, u_ref, '--', label='reference u')
            axes[1].plot(t_ref, v_ref, '--', label='reference v')
            axes[2].plot(
                t_ref, c(reference, 'x_dot'), '--',
                label='reference x_dot')
            axes[3].plot(
                t_ref, c(reference, 'y_dot'), '--',
                label='reference y_dot')
        if len(truth):
            t_truth = c(truth, 't')
            u_truth, v_truth = self.body_velocity_from_local(
                c(truth, 'x_dot'), c(truth, 'y_dot'), c(truth, 'psi'))
            axes[0].plot(t_truth, u_truth, ':', label='truth u')
            axes[1].plot(t_truth, v_truth, ':', label='truth v')
            axes[2].plot(
                t_truth, c(truth, 'x_dot'), ':', label='truth x_dot')
            axes[3].plot(
                t_truth, c(truth, 'y_dot'), ':', label='truth y_dot')
        for axis, ylabel in zip(
                axes,
                ['u [m/s]', 'v [m/s]', 'x_dot [m/s]', 'y_dot [m/s]']):
            axis.set_ylabel(ylabel)
            axis.grid(True)
            axis.legend()
        self.draw_replan_event_lines(axes, accepted_replans, rejected_replans)
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(
            self.output_dir / '03b_body_world_velocities.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for axis, name, ylabel in [
                (axes[0], 'x_dot', 'x_dot [m/s]'),
                (axes[1], 'y_dot', 'y_dot [m/s]'),
                (axes[2], 'psi', 'psi [rad]')]:
            axis.plot(t, c(estimate, name), label='EKF')
            if len(reference):
                axis.plot(
                    c(reference, 't'), c(reference, name), '--',
                    label='active reference')
            if len(truth):
                axis.plot(c(truth, 't'), c(truth, name), ':', label='truth')
            axis.set_ylabel(ylabel)
            axis.grid(True)
            axis.legend()
        self.draw_replan_event_lines(axes, accepted_replans, rejected_replans)
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(self.output_dir / '03c_xdot_ydot_psi.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, c(estimate, 'tau_u'), label='tau_u')
        axes[0].plot(t, c(estimate, 'tau_v'), label='tau_v implied')
        axes[0].plot(t, c(estimate, 'tau_v_slack'), '--', label='tau_v slack')
        axes[0].plot(t, c(estimate, 'tau_r'), label='tau_r')
        axes[0].set_ylabel('generalized force')
        axes[1].plot(t, c(estimate, 'left_cmd'), label='left')
        axes[1].plot(t, c(estimate, 'right_cmd'), label='right')
        axes[1].set_ylabel('thrust [N]')
        for axis in axes:
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(self.output_dir / '04_forces.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(t, c(estimate, 'ref_distance'), label='current ref distance')
        axes[0].plot(t, c(estimate, 'terminal_distance'), label='terminal distance')
        axes[0].set_ylabel('distance [m]')
        axes[1].plot(t, c(estimate, 'psi_ref_error'), label='current ref yaw')
        axes[1].plot(t, c(estimate, 'terminal_yaw_error'), label='terminal yaw')
        axes[1].set_ylabel('yaw error [rad]')
        axes[2].plot(t, c(estimate, 'max_pred_tau_v'), label='max |tau_v|')
        axes[2].plot(
            t, c(estimate, 'max_pred_tau_v_slack'), '--',
            label='max slack')
        axes[2].plot(
            t, c(estimate, 'max_pred_tau_v_violation'), ':',
            label='max violation')
        axes[2].set_ylabel('tau_v [N]')
        axes[3].plot(t, c(estimate, 'solver_success'), label='solver success')
        axes[3].plot(t, c(estimate, 'thrust_saturated'), label='thrust saturated')
        axes[3].set_ylabel('flag')
        self.draw_replan_event_lines(
            axes, accepted_replans, rejected_replans)
        for axis in axes:
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(self.output_dir / '05_errors_solver.png', dpi=160)
        plt.close(fig)

        if len(reference):
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for axis, name, horizon_name, ylabel in [
                    (axes[0], 'x', 'x_ref', 'x [m]'),
                    (axes[1], 'y', 'y_ref', 'y [m]'),
                    (axes[2], 'psi', 'psi_ref', 'psi [rad]')]:
                axis.plot(
                    c(reference, 't'), c(reference, name),
                    label='executed reference sample z_ref(t)')
                axis.plot(
                    c(reference, 't'), c(reference, horizon_name), '--',
                    label='first NMPC horizon sample z_ref(t+dt_1)')
                axis.set_ylabel(ylabel)
                axis.grid(True)
            self.draw_replan_event_lines(
                axes, accepted_replans, rejected_replans)
            axes[-1].set_xlabel('time [s]')
            axes[0].legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / '06_reference_timing.png', dpi=160)
            plt.close(fig)

    def replan_event_times(self, header, reference):
        if len(reference) == 0:
            return [], []
        accepted = []
        rejected = []
        t = self.column(reference, header, 't')
        if 'reference_last_swap' in header:
            swap = self.column(reference, header, 'reference_last_swap')
            accepted = [float(time) for time, flag in zip(t, swap) if flag > 0.5]
        if 'reference_last_reject' in header:
            reject = self.column(reference, header, 'reference_last_reject')
            rejected = [
                float(time) for time, flag in zip(t, reject) if flag > 0.5]
        if not accepted:
            accepted = self.detect_reference_elapsed_resets(header, reference)
        return accepted, rejected

    def detect_reference_elapsed_resets(self, header, data):
        if len(data) == 0 or 'active_reference_elapsed' not in header:
            return []
        t = self.column(data, header, 't')
        elapsed = self.column(data, header, 'active_reference_elapsed')
        duration = (
            self.column(data, header, 'reference_duration')
            if 'reference_duration' in header else
            np.full(len(data), np.nan, dtype=float))
        events = []
        previous_elapsed = float('nan')
        previous_duration = float('nan')
        active_seen = False
        for time_value, elapsed_value, duration_value in zip(t, elapsed, duration):
            if not np.isfinite(elapsed_value):
                continue
            if not active_seen:
                events.append(float(time_value))
                active_seen = True
            elif (
                    elapsed_value + 0.2 < previous_elapsed or
                    (
                        np.isfinite(duration_value) and
                        np.isfinite(previous_duration) and
                        abs(duration_value - previous_duration) > 0.5 and
                        elapsed_value < 0.3)):
                events.append(float(time_value))
            previous_elapsed = elapsed_value
            previous_duration = duration_value
        return events

    def draw_replan_event_lines(
            self, axes, accepted_replans, rejected_replans=None):
        axes = np.atleast_1d(axes)
        rejected_replans = rejected_replans or []
        for axis_index, axis in enumerate(axes):
            for event_index, event_time in enumerate(accepted_replans):
                axis.axvline(
                    event_time,
                    color='tab:green',
                    linestyle='-',
                    linewidth=0.9,
                    alpha=0.45,
                    label=(
                        'accepted reference swap'
                        if axis_index == 0 and event_index == 0 else None))
            for event_index, event_time in enumerate(rejected_replans):
                axis.axvline(
                    event_time,
                    color='tab:red',
                    linestyle=':',
                    linewidth=0.9,
                    alpha=0.35,
                    label=(
                        'rejected replan'
                        if axis_index == 0 and event_index == 0 else None))

    def body_velocity_from_local(self, x_dot, y_dot, psi):
        x_dot = np.asarray(x_dot, dtype=float)
        y_dot = np.asarray(y_dot, dtype=float)
        psi = np.asarray(psi, dtype=float)
        u = np.cos(psi) * x_dot + np.sin(psi) * y_dot
        v = -np.sin(psi) * x_dot + np.cos(psi) * y_dot
        return u, v

    def plot_xy_context(
            self, ax, header, estimate, truth, reference,
            include_global_goal):
        debug_source = reference if len(reference) else estimate
        if len(reference):
            x_ref = self.column(reference, header, 'x')
            y_ref = self.column(reference, header, 'y')
        else:
            x_ref = self.first_available_column(
                estimate, header, ['active_x_ref', 'x_ref'])
            y_ref = self.first_available_column(
                estimate, header, ['active_y_ref', 'y_ref'])
        terminal_x = self.optional_column(debug_source, header, 'terminal_x_ref')
        terminal_y = self.optional_column(debug_source, header, 'terminal_y_ref')
        local_x = self.optional_column(debug_source, header, 'frontend_terminal_x')
        local_y = self.optional_column(debug_source, header, 'frontend_terminal_y')
        self.plot_xy_line(
            ax, x_ref, y_ref, 'k:', label='executed reference sample z_ref(t)',
            linewidth=1.4, alpha=0.85)
        self.plot_xy_line(
            ax, terminal_x, terminal_y, color='0.45', linestyle='--',
            label='NMPC lookahead terminal z_ref(t+H)',
            linewidth=1.2, alpha=0.7)
        local_points = self.unique_xy(local_x, local_y)
        if len(local_points):
            ax.scatter(
                local_points[:, 0], local_points[:, 1],
                s=28, marker='o', color='tab:purple', alpha=0.55,
                label='selected local MINCO terminal')
        self.plot_xy_line(
            ax,
            self.column(estimate, header, 'x'),
            self.column(estimate, header, 'y'),
            color='tab:blue', linestyle='-', label='EKF estimate',
            linewidth=1.7, alpha=0.9)
        self.plot_start_end(
            ax,
            self.column(estimate, header, 'x'),
            self.column(estimate, header, 'y'),
            'EKF')
        if len(truth):
            self.plot_xy_line(
                ax,
                self.column(truth, header, 'x'),
                self.column(truth, header, 'y'),
                color='tab:orange', linestyle='-', label='truth',
                linewidth=1.4, alpha=0.85)
        self.draw_obstacles(ax)
        if include_global_goal:
            self.draw_global_goal(ax)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend()

    def plot_xy_line(
            self, ax, x, y, *args, label=None, color=None, linestyle=None,
            linewidth=None, alpha=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return
        kwargs = {}
        if label is not None:
            kwargs['label'] = label
        if color is not None:
            kwargs['color'] = color
        if linestyle is not None:
            kwargs['linestyle'] = linestyle
        if linewidth is not None:
            kwargs['linewidth'] = linewidth
        if alpha is not None:
            kwargs['alpha'] = alpha
        ax.plot(x[mask], y[mask], *args, **kwargs)

    def unique_xy(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return np.empty((0, 2), dtype=float)
        points = np.column_stack((x[mask], y[mask]))
        _, unique_indices = np.unique(
            np.round(points, decimals=3), axis=0, return_index=True)
        return points[np.sort(unique_indices)]

    def plot_start_end(self, ax, x, y, label_prefix):
        points = self.unique_xy(x, y)
        if len(points) == 0:
            return
        ax.scatter(
            points[0, 0], points[0, 1], marker='s', s=42,
            color='tab:blue', edgecolor='white', linewidth=0.8,
            label=f'{label_prefix} start')
        ax.scatter(
            points[-1, 0], points[-1, 1], marker='x', s=54,
            color='tab:blue', linewidth=1.4,
            label=f'{label_prefix} end')

    def draw_global_goal(self, ax):
        if self.global_goal is None:
            return
        x, y, psi = [float(value) for value in self.global_goal]
        ax.scatter(
            [x], [y], marker='*', s=130, color='tab:green',
            edgecolor='black', linewidth=0.6,
            label='global goal frontend_goal_z')
        ax.arrow(
            x, y, 0.65 * math.cos(psi), 0.65 * math.sin(psi),
            color='tab:green', width=0.035, length_includes_head=True,
            head_width=0.18, head_length=0.24)

    def set_local_xy_limits(self, ax, header, estimate, truth, reference):
        debug_source = reference if len(reference) else estimate
        arrays = [
            (self.column(estimate, header, 'x'), self.column(estimate, header, 'y')),
            (self.first_available_column(
                estimate, header, ['active_x_ref', 'x_ref']),
             self.first_available_column(
                 estimate, header, ['active_y_ref', 'y_ref'])),
            (self.optional_column(debug_source, header, 'terminal_x_ref'),
             self.optional_column(debug_source, header, 'terminal_y_ref')),
            (self.optional_column(debug_source, header, 'frontend_terminal_x'),
             self.optional_column(debug_source, header, 'frontend_terminal_y')),
        ]
        if len(reference):
            arrays.append((
                self.column(reference, header, 'x'),
                self.column(reference, header, 'y')))
        if len(truth):
            arrays.append((
                self.column(truth, header, 'x'),
                self.column(truth, header, 'y')))
        points = []
        for x, y in arrays:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                points.append(np.column_stack((x[mask], y[mask])))
        if not points:
            return
        points = np.vstack(points)
        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        margin = 0.15 * span + 0.25
        ax.set_xlim(min_xy[0] - margin[0], max_xy[0] + margin[0])
        ax.set_ylim(min_xy[1] - margin[1], max_xy[1] + margin[1])

    def draw_obstacles(self, ax):
        for index, obstacle in enumerate(self.obstacles):
            label = 'simulated obstacle' if index == 0 else None
            footprint = plt.Circle(
                (float(obstacle.x), float(obstacle.y)),
                float(obstacle.radius),
                fill=True,
                color='tab:red',
                alpha=0.18,
                label=label,
            )
            boundary = plt.Circle(
                (float(obstacle.x), float(obstacle.y)),
                float(obstacle.radius),
                fill=False,
                color='tab:red',
                linewidth=1.5,
            )
            ax.add_patch(footprint)
            ax.add_patch(boundary)


def parse_global_goal(text: str):
    if not text:
        return None
    values = [float(part.strip()) for part in text.split(',')]
    if len(values) != 3:
        raise ValueError('Global goal must be formatted as x,y,psi.')
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--duration', type=float, default=45.0)
    parser.add_argument(
        '--obstacles',
        default='',
        help=(
            'Optional simulated circular obstacles formatted as '
            '"x,y,r;x,y,r". Pass the same value as frontend_obstacles.'))
    parser.add_argument(
        '--global-goal',
        default='3.0,6.0,2.4',
        help='Configured frontend global goal formatted as x,y,psi.')
    args = parser.parse_args()

    rclpy.init()
    node = FlatnessMpcRecorder(
        Path(args.output_dir),
        args.duration,
        obstacles=parse_circular_obstacles(args.obstacles),
        global_goal=parse_global_goal(args.global_goal),
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
