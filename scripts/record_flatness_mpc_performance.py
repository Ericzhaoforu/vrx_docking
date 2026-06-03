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
from robotx_safe_docking_control.flatness_mpc_controller import yaw_from_quaternion
from robotx_safe_docking_control.usv_flatness import wrap_angle


class FlatnessMpcRecorder(Node):
    def __init__(self, output_dir: Path, duration: float):
        super().__init__('flatness_mpc_performance_recorder')
        self.output_dir = output_dir
        self.duration = duration
        self.start_time = None
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
        return now - self.start_time

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

    def on_estimate(self, msg: Odometry):
        t = self.rel_time()
        z, z_dot = self.extract_state(msg)
        self.latest_estimate_xy = z[:2]
        self.try_initialize_truth_alignment()
        self.samples.append(self.make_sample(
            t, 'estimate', z, z_dot))

    def on_truth(self, msg: Odometry):
        t = self.rel_time()
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
        x_ref = debug.get('x_ref', float('nan'))
        y_ref = debug.get('y_ref', float('nan'))
        psi_ref = debug.get('psi_ref', float('nan'))
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
        if self.start_time is None:
            return
        if self.rel_time() >= self.duration and not self.done:
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
        if len(estimate) == 0:
            return

        def c(data, name):
            return self.column(data, header, name)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(c(estimate, 'x_ref'), c(estimate, 'y_ref'), 'k--', label='reference')
        ax.plot(c(estimate, 'x'), c(estimate, 'y'), label='EKF')
        if len(truth):
            ax.plot(c(truth, 'x'), c(truth, 'y'), label='truth')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / '01_trajectory_xy.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for axis, name, ref_name, ylabel in [
                (axes[0], 'x', 'x_ref', 'x [m]'),
                (axes[1], 'y', 'y_ref', 'y [m]'),
                (axes[2], 'psi', 'psi_ref', 'psi [rad]')]:
            axis.plot(c(estimate, 't'), c(estimate, name), label='EKF')
            axis.plot(c(estimate, 't'), c(estimate, ref_name), '--', label='reference')
            if len(truth):
                axis.plot(c(truth, 't'), c(truth, name), label='truth')
            axis.set_ylabel(ylabel)
            axis.grid(True)
        axes[-1].set_xlabel('time [s]')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / '02_states.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        t = c(estimate, 't')
        axes[0].plot(t, c(estimate, 'speed'), label='speed')
        axes[0].set_ylabel('speed [m/s]')
        axes[1].plot(t, c(estimate, 'psi_dot'), label='yaw rate')
        axes[1].plot(t, c(estimate, 'psi_dot_ref'), '--', label='yaw rate ref')
        axes[1].set_ylabel('yaw rate [rad/s]')
        for axis in axes:
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(self.output_dir / '03_velocities.png', dpi=160)
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
        for axis in axes:
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(self.output_dir / '05_errors_solver.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--duration', type=float, default=45.0)
    args = parser.parse_args()

    rclpy.init()
    node = FlatnessMpcRecorder(Path(args.output_dir), args.duration)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
