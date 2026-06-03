#!/usr/bin/env python3
import argparse
import csv
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

from robotx_safe_docking_estimation.gps_imu_ekf_node import yaw_from_quaternion
from robotx_safe_docking_estimation.gps_imu_ekf_node import wrap_angle


class ControllerRecorder(Node):
    def __init__(self, output_dir: Path, duration: float,
                 target_x: float, target_y: float, target_yaw: float):
        super().__init__('controller_performance_recorder')
        self.output_dir = output_dir
        self.duration = duration
        self.target = np.array([target_x, target_y], dtype=float)
        self.target_yaw = target_yaw
        self.start_time = None
        self.latest_estimate_xy = None
        self.latest_truth_xy = None
        self.truth_to_estimate_offset = None
        self.latest_command = [0.0, 0.0]
        self.latest_debug = {}
        self.samples = []
        self.wrote_outputs = False
        self.done = False

        self.create_subscription(
            Odometry, '/safe_docking/odometry', self.on_estimate, 10)
        self.create_subscription(
            Odometry, '/wamv/sensors/position/ground_truth_odometry',
            self.on_truth, 10)
        self.create_subscription(
            Float64MultiArray, '/safe_docking/controller/command',
            self.on_command, 10)
        self.create_subscription(
            Float64MultiArray, '/safe_docking/controller/debug',
            self.on_debug, 10)
        self.create_timer(0.2, self.check_done)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def rel_time(self):
        now = self.now_sec()
        if self.start_time is None:
            self.start_time = now
        return now - self.start_time

    def on_command(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            self.latest_command = [float(msg.data[0]), float(msg.data[1])]

    def on_debug(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 40:
            return
        self.latest_debug = {
            'x_d': float(data[9]),
            'y_d': float(data[10]),
            'psi_d': float(data[11]),
            'u_d': float(data[12]),
            'r_d': float(data[13]),
            'u_cmd': float(data[14]),
            'r_cmd': float(data[15]),
            'e_x': float(data[16]),
            'e_y': float(data[17]),
            'e_psi': float(data[18]),
            's_ref': float(data[19]),
            'arc_ref': float(data[20]),
            'kappa_ref': float(data[21]),
            'trajectory_u_limit': float(data[22]),
            'trajectory_curvature_limit': float(data[23]),
            'trajectory_yaw_rate_limit': float(data[24]),
            'surge_speed': float(data[25]),
            'force': float(data[26]),
            'torque': float(data[27]),
            'surge_force_c': float(data[30]),
            'tau_c': float(data[31]),
            'allocation_cost': float(data[32]),
            'path_max_abs_kappa': float(data[33]),
            'path_curvature_feasible': float(data[34]),
            'trajectory_valid': float(data[35]),
            'waypoint_index': float(data[39]),
        }

    def on_estimate(self, msg: Odometry):
        t = self.rel_time()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        x_dot = msg.twist.twist.linear.x
        y_dot = msg.twist.twist.linear.y
        yaw_rate = msg.twist.twist.angular.z
        speed = math.hypot(
            x_dot, y_dot)
        self.latest_estimate_xy = np.array([x, y], dtype=float)
        self.try_initialize_truth_alignment()
        distance = math.hypot(self.target[0] - x, self.target[1] - y)
        yaw_error = wrap_angle(self.target_yaw - yaw)
        self.samples.append(self.make_sample([
            t, 'estimate', x, y, yaw, x_dot, y_dot, yaw_rate,
            speed, distance, yaw_error,
            self.latest_command[0], self.latest_command[1],
        ], yaw))

    def on_truth(self, msg: Odometry):
        t = self.rel_time()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.latest_truth_xy = np.array([x, y], dtype=float)
        self.try_initialize_truth_alignment()
        if self.truth_to_estimate_offset is None:
            return
        truth_local = self.latest_truth_xy + self.truth_to_estimate_offset
        x_local = truth_local[0]
        y_local = truth_local[1]
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        x_dot = msg.twist.twist.linear.x
        y_dot = msg.twist.twist.linear.y
        yaw_rate = msg.twist.twist.angular.z
        speed = math.hypot(
            x_dot, y_dot)
        distance = math.hypot(
            self.target[0] - x_local, self.target[1] - y_local)
        yaw_error = wrap_angle(self.target_yaw - yaw)
        self.samples.append(self.make_sample([
            t, 'truth', x_local, y_local, yaw, x_dot, y_dot, yaw_rate,
            speed, distance, yaw_error,
            self.latest_command[0], self.latest_command[1],
        ], yaw))

    def make_sample(self, base_sample, yaw: float):
        psi_d = self.latest_debug.get('psi_d', float('nan'))
        sample = list(base_sample)
        sample.extend([
            self.latest_debug.get('waypoint_index', float('nan')),
            self.latest_debug.get('x_d', float('nan')),
            self.latest_debug.get('y_d', float('nan')),
            psi_d,
            self._yaw_error_or_nan(psi_d, yaw),
            self.latest_debug.get('u_d', float('nan')),
            self.latest_debug.get('r_d', float('nan')),
            self.latest_debug.get('u_cmd', float('nan')),
            self.latest_debug.get('r_cmd', float('nan')),
            self.latest_debug.get('e_x', float('nan')),
            self.latest_debug.get('e_y', float('nan')),
            self.latest_debug.get('e_psi', float('nan')),
            self.latest_debug.get('s_ref', float('nan')),
            self.latest_debug.get('arc_ref', float('nan')),
            self.latest_debug.get('kappa_ref', float('nan')),
            self.latest_debug.get('trajectory_u_limit', float('nan')),
            self.latest_debug.get(
                'trajectory_curvature_limit', float('nan')),
            self.latest_debug.get(
                'trajectory_yaw_rate_limit', float('nan')),
            self.latest_debug.get('surge_speed', float('nan')),
            self.latest_debug.get('force', float('nan')),
            self.latest_debug.get('torque', float('nan')),
            self.latest_debug.get('surge_force_c', float('nan')),
            self.latest_debug.get('tau_c', float('nan')),
            self.latest_debug.get('allocation_cost', float('nan')),
            self.latest_debug.get('path_max_abs_kappa', float('nan')),
            self.latest_debug.get('path_curvature_feasible', float('nan')),
            self.latest_debug.get('trajectory_valid', float('nan')),
        ])
        return sample

    def _yaw_error_or_nan(self, reference: float, yaw: float):
        if math.isnan(reference):
            return float('nan')
        return wrap_angle(reference - yaw)

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

    def check_done(self):
        if self.start_time is None:
            return
        if self.now_sec() - self.start_time >= self.duration:
            self.write_outputs()
            self.wrote_outputs = True
            self.done = True

    def write_outputs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            self.get_logger().error('No controller samples recorded.')
            return

        csv_path = self.output_dir / 'controller_performance.csv'
        with csv_path.open('w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                't', 'source', 'x', 'y', 'yaw', 'x_dot', 'y_dot',
                'yaw_rate', 'speed', 'distance', 'yaw_error',
                'left_thrust', 'right_thrust',
                'waypoint_index', 'x_d', 'y_d', 'psi_d', 'psi_error',
                'u_d', 'r_d', 'u_cmd', 'r_cmd', 'e_x', 'e_y', 'e_psi',
                's_ref', 'arc_ref', 'kappa_ref', 'trajectory_u_limit',
                'trajectory_curvature_limit', 'trajectory_yaw_rate_limit',
                'surge_speed', 'force', 'torque', 'surge_force_c', 'tau_c',
                'allocation_cost', 'path_max_abs_kappa',
                'path_curvature_feasible', 'trajectory_valid',
            ])
            writer.writerows(self.samples)

        self.plot()
        self.summary()

    def data_for(self, source: str):
        rows = [row for row in self.samples if row[1] == source]
        return np.array([[row[0]] + row[2:] for row in rows], dtype=float)

    def plot(self):
        est = self.data_for('estimate')
        truth = self.data_for('truth')
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        if len(truth):
            ax.plot(truth[:, 1], truth[:, 2], label='ground truth')
        if len(est):
            ax.plot(est[:, 1], est[:, 2], label='EKF estimate')
        ax.scatter([self.target[0]], [self.target[1]], marker='x',
                   s=90, label='target')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        ax = axes[0, 1]
        if len(truth):
            ax.plot(truth[:, 0], truth[:, 8], label='truth distance')
        if len(est):
            ax.plot(est[:, 0], est[:, 8], label='estimate distance')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('distance to target (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        ax = axes[1, 0]
        if len(est):
            ax.plot(est[:, 0], est[:, 9], label='yaw error')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('yaw error (rad)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if len(est):
            ax.plot(est[:, 0], est[:, 10], label='left thrust')
            ax.plot(est[:, 0], est[:, 11], label='right thrust')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('thrust command (N)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        fig.suptitle('Safe Docking Trajectory Tracker Performance')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'controller_performance.png', dpi=160)
        plt.close(fig)

    def summary(self):
        est = self.data_for('estimate')
        truth = self.data_for('truth')
        summary_path = self.output_dir / 'controller_performance_summary.txt'
        with summary_path.open('w') as f:
            f.write('Trajectory tracker performance\n')
            f.write(f'target_x,{self.target[0]:.6f}\n')
            f.write(f'target_y,{self.target[1]:.6f}\n')
            f.write(f'target_yaw,{self.target_yaw:.6f}\n')
            if self.truth_to_estimate_offset is not None:
                f.write(
                    'truth_to_estimate_offset_x,'
                    f'{self.truth_to_estimate_offset[0]:.6f}\n')
                f.write(
                    'truth_to_estimate_offset_y,'
                    f'{self.truth_to_estimate_offset[1]:.6f}\n')
            for name, data in (('estimate', est), ('truth', truth)):
                if len(data) == 0:
                    continue
                dist = data[:, 8]
                yaw_err = np.abs(data[:, 9])
                f.write(f'\n{name}_samples,{len(data)}\n')
                f.write(f'{name}_final_distance,{dist[-1]:.6f}\n')
                f.write(f'{name}_min_distance,{np.min(dist):.6f}\n')
                f.write(f'{name}_final_yaw_error,{yaw_err[-1]:.6f}\n')
                f.write(f'{name}_max_abs_yaw_error,{np.max(yaw_err):.6f}\n')
                f.write(f'{name}_mean_distance,{np.mean(dist):.6f}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/output_controller')
    parser.add_argument('--duration', type=float, default=80.0)
    parser.add_argument('--target-x', type=float, default=6.0)
    parser.add_argument('--target-y', type=float, default=12.0)
    parser.add_argument('--target-yaw', type=float, default=2.4)
    args = parser.parse_args()

    rclpy.init()
    node = ControllerRecorder(
        Path(args.output_dir), args.duration,
        args.target_x, args.target_y, args.target_yaw)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if not node.wrote_outputs:
            node.write_outputs()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
