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

from robotx_safe_docking_estimation.gps_imu_ekf_node import yaw_from_quaternion
from robotx_safe_docking_estimation.gps_imu_ekf_node import wrap_angle


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class EstimatorRecorder(Node):
    def __init__(self, output_dir: Path, duration: float):
        super().__init__('estimator_performance_recorder')
        self.output_dir = output_dir
        self.duration = duration
        self.start_time = None
        self.latest_estimate_xy = None
        self.latest_truth_xy = None
        self.truth_to_estimate_offset = None
        self.last_truth_pose = None
        self.estimates = []
        self.truth = []
        self.wrote_outputs = False
        self.done = False

        self.create_subscription(
            Odometry, '/safe_docking/odometry', self.on_estimate, 10)
        self.create_subscription(
            Odometry, '/wamv/sensors/position/ground_truth_odometry',
            self.on_truth, 10)
        self.create_timer(0.2, self.check_done)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_estimate(self, msg: Odometry):
        now = self._now_sec()
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.latest_estimate_xy = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ], dtype=float)
        self.try_initialize_truth_alignment()
        self.estimates.append([
            t,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z,
        ])

    def on_truth(self, msg):
        now = self._now_sec()
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.latest_truth_xy = np.array([x, y], dtype=float)
        self.try_initialize_truth_alignment()
        if self.truth_to_estimate_offset is None:
            return

        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        truth_local = self.latest_truth_xy + self.truth_to_estimate_offset
        x_local = truth_local[0]
        y_local = truth_local[1]

        if self.last_truth_pose is None:
            x_dot = 0.0
            y_dot = 0.0
            psi_dot = msg.twist.twist.angular.z
        else:
            last_t, last_x, last_y, last_yaw = self.last_truth_pose
            dt = t - last_t
            if dt > 1e-6:
                x_dot = (x_local - last_x) / dt
                y_dot = (y_local - last_y) / dt
                psi_dot = wrap_angle(yaw - last_yaw) / dt
            else:
                x_dot = 0.0
                y_dot = 0.0
                psi_dot = 0.0
        self.last_truth_pose = (t, x_local, y_local, yaw)
        self.truth.append([t, x_local, y_local, yaw, x_dot, y_dot, psi_dot])

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
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        if elapsed >= self.duration and not self.wrote_outputs:
            self.write_outputs()
            self.wrote_outputs = True
            self.done = True

    def write_outputs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if len(self.estimates) < 2 or len(self.truth) < 2:
            self.get_logger().error('Not enough data to write plots.')
            return

        est = np.array(self.estimates, dtype=float)
        truth = np.array(self.truth, dtype=float)

        aligned = []
        truth_t = truth[:, 0]
        for row in est:
            idx = int(np.argmin(np.abs(truth_t - row[0])))
            if abs(truth_t[idx] - row[0]) > 0.1:
                continue
            err = row[1:7] - truth[idx, 1:7]
            err[2] = wrap_angle(err[2])
            aligned.append(list(row) + list(truth[idx, 1:7]) + list(err))

        if not aligned:
            self.get_logger().error('No aligned estimator / truth samples.')
            return

        aligned = np.array(aligned, dtype=float)
        csv_path = self.output_dir / 'estimator_state_comparison.csv'
        headers = [
            't',
            'est_x', 'est_y', 'est_psi',
            'est_x_dot', 'est_y_dot', 'est_psi_dot',
            'truth_x', 'truth_y', 'truth_psi',
            'truth_x_dot', 'truth_y_dot', 'truth_psi_dot',
            'err_x', 'err_y', 'err_psi',
            'err_x_dot', 'err_y_dot', 'err_psi_dot',
        ]
        with csv_path.open('w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(aligned.tolist())

        self._plot_states(aligned)
        self._plot_errors(aligned)
        self._write_summary(aligned)

    def _plot_states(self, data):
        labels = [
            ('x', 'm'), ('y', 'm'), ('psi', 'rad'),
            ('x_dot', 'm/s'), ('y_dot', 'm/s'), ('psi_dot', 'rad/s'),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        t = data[:, 0]
        for i, ax in enumerate(axes.flat):
            est_col = 1 + i
            truth_col = 7 + i
            ax.plot(t, data[:, truth_col], label='ground truth', linewidth=1.4)
            ax.plot(t, data[:, est_col], label='EKF estimate', linewidth=1.0)
            ax.set_ylabel(f'{labels[i][0]} ({labels[i][1]})')
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel('time (s)')
        axes[-1, 1].set_xlabel('time (s)')
        axes[0, 0].legend(loc='best')
        fig.suptitle('GPS + IMU EKF State Estimate vs Ground Truth')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'estimator_state_comparison.png',
                    dpi=160)
        plt.close(fig)

    def _plot_errors(self, data):
        labels = [
            ('x error', 'm'), ('y error', 'm'), ('psi error', 'rad'),
            ('x_dot error', 'm/s'), ('y_dot error', 'm/s'),
            ('psi_dot error', 'rad/s'),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        t = data[:, 0]
        for i, ax in enumerate(axes.flat):
            err_col = 13 + i
            ax.plot(t, data[:, err_col], linewidth=1.0)
            ax.axhline(0.0, color='black', linewidth=0.7)
            ax.set_ylabel(f'{labels[i][0]} ({labels[i][1]})')
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel('time (s)')
        axes[-1, 1].set_xlabel('time (s)')
        fig.suptitle('GPS + IMU EKF Estimation Error')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'estimator_errors.png', dpi=160)
        plt.close(fig)

    def _write_summary(self, data):
        err = data[:, 13:19]
        rmse = np.sqrt(np.mean(err * err, axis=0))
        mae = np.mean(np.abs(err), axis=0)
        max_abs = np.max(np.abs(err), axis=0)
        labels = ['x', 'y', 'psi', 'x_dot', 'y_dot', 'psi_dot']
        summary_path = self.output_dir / 'estimator_performance_summary.txt'
        with summary_path.open('w') as f:
            f.write('GPS + IMU EKF performance vs debug ground truth\n')
            f.write(f'samples: {len(data)}\n')
            f.write(f'duration_s: {data[-1, 0] - data[0, 0]:.3f}\n\n')
            f.write('metric,rmse,mae,max_abs\n')
            for label, r, a, m in zip(labels, rmse, mae, max_abs):
                f.write(f'{label},{r:.6f},{a:.6f},{m:.6f}\n')
        self.get_logger().info(f'Wrote plots and CSV to {self.output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/output')
    parser.add_argument('--duration', type=float, default=30.0)
    args = parser.parse_args()

    rclpy.init()
    node = EstimatorRecorder(Path(args.output_dir), args.duration)
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
