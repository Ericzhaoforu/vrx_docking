#!/usr/bin/env python3
import argparse
import csv
import math
from bisect import bisect_left
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_float_list(value: str):
    if not value:
        return []
    return [float(part.strip()) for part in value.split(',') if part.strip()]


def draw_waypoint_yaw_arrows(ax, waypoints, length=0.85):
    for index, (wx, wy, wyaw) in enumerate(waypoints):
        marker = 'D' if index == 0 else 'x'
        color = 'tab:orange' if index == 0 else 'tab:red'
        label = f'waypoint {index + 1}'
        ax.scatter([wx], [wy], marker=marker, s=90, c=color, label=label)
        ax.arrow(
            wx,
            wy,
            length * math.cos(wyaw),
            length * math.sin(wyaw),
            width=0.035,
            head_width=0.22,
            head_length=0.28,
            length_includes_head=True,
            color=color,
            alpha=0.9,
        )


def value_or_nan(row, key):
    value = row.get(key)
    if value is None or value == '':
        return float('nan')
    return float(value)


def load_rows(csv_path: Path, source: str):
    rows = []
    with csv_path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['source'] != source:
                continue
            x_dot = float(row.get('x_dot', 0.0))
            y_dot = float(row.get('y_dot', 0.0))
            yaw_rate = float(row.get('yaw_rate', 0.0))
            rows.append([
                float(row['t']),
                float(row['x']),
                float(row['y']),
                float(row['yaw']),
                x_dot,
                y_dot,
                yaw_rate,
                float(row['speed']),
                float(row['distance']),
                float(row['yaw_error']),
                float(row['left_thrust']),
                float(row['right_thrust']),
                value_or_nan(row, 'waypoint_index'),
                value_or_nan(row, 'desired_yaw'),
                value_or_nan(row, 'desired_yaw_rate'),
                value_or_nan(row, 's_ref'),
                value_or_nan(row, 'x_ref'),
                value_or_nan(row, 'y_ref'),
                value_or_nan(row, 'psi_ref'),
                value_or_nan(row, 'path_yaw_error'),
                value_or_nan(row, 'command_yaw_error'),
                value_or_nan(row, 'along_track_error'),
                value_or_nan(row, 'cross_track_error'),
                value_or_nan(row, 'u_ref'),
                value_or_nan(row, 'surge_speed'),
                value_or_nan(row, 'surge_force_c'),
                value_or_nan(row, 'tau_c'),
                value_or_nan(row, 'path_max_abs_kappa'),
                value_or_nan(row, 'path_smoothing_attempts'),
                value_or_nan(row, 'path_curvature_feasible'),
            ])
    return np.array(rows, dtype=float)


def nearest_pairs(est, truth):
    truth_t = truth[:, 0]
    pairs = []
    for row in est:
        i = bisect_left(truth_t, row[0])
        candidates = [
            j for j in (i - 1, i, i + 1)
            if 0 <= j < len(truth)
        ]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(truth_t[k] - row[0]))
        if abs(truth_t[j] - row[0]) <= 0.1:
            pairs.append((row, truth[j]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--target-x', type=float, default=6.0)
    parser.add_argument('--target-y', type=float, default=12.0)
    parser.add_argument('--target-yaw', type=float, default=1.0)
    parser.add_argument('--thruster-half-spacing', type=float, default=1.027135)
    parser.add_argument('--waypoint-xs', default='4.0,6.0')
    parser.add_argument('--waypoint-ys', default='8.0,12.0')
    parser.add_argument('--waypoint-yaws', default='1.0,2.4')
    parser.add_argument('--kappa-max', type=float, default=0.7)
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = args.output_dir or csv_path.parent / 'detailed_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    est = load_rows(csv_path, 'estimate')
    truth = load_rows(csv_path, 'truth')
    target = np.array([args.target_x, args.target_y])
    waypoint_xs = parse_float_list(args.waypoint_xs)
    waypoint_ys = parse_float_list(args.waypoint_ys)
    waypoint_yaws = parse_float_list(args.waypoint_yaws)
    waypoints = [
        (x, y, yaw)
        for x, y, yaw in zip(waypoint_xs, waypoint_ys, waypoint_yaws)
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(truth):
        ax.plot(truth[:, 1], truth[:, 2],
                label='ground truth aligned', lw=1.6)
    ax.plot(est[:, 1], est[:, 2], label='EKF estimate', lw=1.1)
    ax.scatter([target[0]], [target[1]], marker='x', s=120,
               c='tab:red', label='target')
    draw_waypoint_yaw_arrows(ax, waypoints)
    ax.scatter([est[0, 1]], [est[0, 2]], marker='o', s=55,
               c='tab:green', label='start')
    ax.scatter([est[-1, 1]], [est[-1, 2]], marker='s', s=55,
               c='tab:purple', label='final estimate')
    ax.set_title('Safe Docking Trajectory in EKF Local Frame')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / '01_trajectory_xy.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = [('x', 'm', 1), ('y', 'm', 2), ('yaw', 'rad', 3)]
    for ax, (label, unit, col) in zip(axes, labels):
        if len(truth):
            ax.plot(truth[:, 0], truth[:, col],
                    label='ground truth aligned', lw=1.3)
        ax.plot(est[:, 0], est[:, col], label='EKF estimate', lw=1.0)
        if label == 'yaw' and est.shape[1] > 18:
            if not np.all(np.isnan(est[:, 18])):
                ax.plot(est[:, 0], est[:, 18], label='path psi_ref',
                        lw=0.9, ls=':')
            if not np.all(np.isnan(est[:, 13])):
                ax.plot(est[:, 0], est[:, 13], label='command psi_cmd',
                        lw=0.9, ls='-.')
        target_value = (
            args.target_x if label == 'x'
            else args.target_y if label == 'y'
            else args.target_yaw
        )
        ax.axhline(target_value, color='tab:red', ls='--',
                   lw=1.0, label='target')
        ax.set_ylabel(f'{label} ({unit})')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    axes[0].legend(loc='best')
    fig.suptitle('Estimated States Compared With Aligned Ground Truth')
    fig.tight_layout()
    fig.savefig(out_dir / '02_states_x_y_yaw.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    rate_labels = [
        ('x_dot', 'm/s', 4),
        ('y_dot', 'm/s', 5),
        ('yaw_rate', 'rad/s', 6),
    ]
    for ax, (label, unit, col) in zip(axes, rate_labels):
        if len(truth):
            ax.plot(truth[:, 0], truth[:, col],
                    label='ground truth aligned', lw=1.3)
        ax.plot(est[:, 0], est[:, col], label='EKF estimate', lw=1.0)
        ax.axhline(0.0, color='black', lw=0.7)
        ax.set_ylabel(f'{label} ({unit})')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    axes[0].legend(loc='best')
    fig.suptitle('Estimated Velocity States')
    fig.tight_layout()
    fig.savefig(out_dir / '03_velocity_states.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    if len(truth):
        axes[0].plot(truth[:, 0], truth[:, 8],
                     label='truth distance', lw=1.3)
    axes[0].plot(est[:, 0], est[:, 8], label='estimate distance', lw=1.0)
    axes[0].axhline(0.4, color='tab:gray', ls='--',
                    lw=1.0, label='0.4 m tolerance')
    axes[0].set_ylabel('distance to target (m)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    if len(truth):
        axes[1].plot(truth[:, 0], truth[:, 9],
                     label='truth final yaw error', lw=1.3)
    axes[1].plot(est[:, 0], est[:, 9],
                 label='estimate final yaw error', lw=1.0)
    if est.shape[1] > 20:
        if not np.all(np.isnan(est[:, 19])):
            axes[1].plot(est[:, 0], est[:, 19],
                         label='path yaw error psi_ref - psi',
                         lw=1.0, ls=':')
        if not np.all(np.isnan(est[:, 20])):
            axes[1].plot(est[:, 0], est[:, 20],
                         label='command yaw error psi_cmd - psi',
                         lw=1.0, ls='-.')
    axes[1].axhline(0.0, color='black', lw=0.7)
    axes[1].axhline(0.15, color='tab:gray', ls='--',
                    lw=1.0, label='+-0.15 rad tolerance')
    axes[1].axhline(-0.15, color='tab:gray', ls='--', lw=1.0)
    axes[1].set_ylabel('yaw error (rad)')
    axes[1].set_xlabel('time (s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')
    fig.suptitle('Target Tracking Error')
    fig.tight_layout()
    fig.savefig(out_dir / '04_target_errors.png', dpi=170)
    plt.close(fig)

    pairs = nearest_pairs(est, truth)
    if pairs:
        times = np.array([p[0][0] for p in pairs])
        err_x = np.array([p[0][1] - p[1][1] for p in pairs])
        err_y = np.array([p[0][2] - p[1][2] for p in pairs])
        err_yaw = np.array([
            wrap_angle(p[0][3] - p[1][3]) for p in pairs])
        err_xy = np.hypot(err_x, err_y)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for ax, data, ylabel in [
            (axes[0], err_x, 'x error (m)'),
            (axes[1], err_y, 'y error (m)'),
            (axes[2], err_yaw, 'yaw error (rad)'),
            (axes[3], err_xy, 'position error norm (m)'),
        ]:
            ax.plot(times, data, lw=1.0)
            ax.axhline(0.0, color='black', lw=0.7)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('time (s)')
        fig.suptitle('EKF Estimate Error vs Aligned Ground Truth')
        fig.tight_layout()
        fig.savefig(out_dir / '05_estimation_errors.png', dpi=170)
        plt.close(fig)

    left = est[:, 10]
    right = est[:, 11]
    surge = left + right
    torque = args.thruster_half_spacing * (right - left)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(est[:, 0], left, label='left thrust', lw=1.0)
    axes[0].plot(est[:, 0], right, label='right thrust', lw=1.0)
    axes[0].axhline(500, color='tab:gray', ls='--', lw=0.8)
    axes[0].axhline(-500, color='tab:gray', ls='--', lw=0.8)
    axes[0].set_ylabel('thruster force (N)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(est[:, 0], surge, color='tab:green', lw=1.0)
    axes[1].set_ylabel('T_L + T_R (N)')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(est[:, 0], torque, color='tab:purple', lw=1.0)
    axes[2].set_ylabel('yaw torque (N*m)')
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(est[:, 0], est[:, 8], label='distance', lw=1.0)
    axes[3].set_ylabel('distance (m)')
    axes[3].set_xlabel('time (s)')
    axes[3].grid(True, alpha=0.3)
    fig.suptitle('Thruster Commands and Allocated Force/Torque')
    fig.tight_layout()
    fig.savefig(out_dir / '06_thruster_forces.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(est[:, 1], est[:, 2], label='EKF', lw=1.0)
    if len(truth):
        axes[0, 0].plot(truth[:, 1], truth[:, 2], label='truth', lw=1.2)
    axes[0, 0].scatter([target[0]], [target[1]], marker='x',
                       s=90, c='tab:red')
    draw_waypoint_yaw_arrows(axes[0, 0], waypoints, length=0.75)
    axes[0, 0].axis('equal')
    axes[0, 0].set_title('trajectory')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='best')
    axes[0, 1].plot(est[:, 0], est[:, 8], label='estimate', lw=1.0)
    if len(truth):
        axes[0, 1].plot(truth[:, 0], truth[:, 8], label='truth', lw=1.2)
    axes[0, 1].set_title('distance to target')
    axes[0, 1].set_xlabel('time (s)')
    axes[0, 1].set_ylabel('m')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='best')
    axes[1, 0].plot(est[:, 0], est[:, 3], label='yaw', lw=1.0)
    if est.shape[1] > 18 and not np.all(np.isnan(est[:, 18])):
        axes[1, 0].plot(est[:, 0], est[:, 18], label='path psi_ref',
                        lw=0.9, ls=':')
    if est.shape[1] > 13 and not np.all(np.isnan(est[:, 13])):
        axes[1, 0].plot(est[:, 0], est[:, 13], label='command psi_cmd',
                        lw=0.9, ls='-.')
    axes[1, 0].axhline(args.target_yaw, color='tab:red',
                       ls='--', lw=1.0, label='target')
    axes[1, 0].set_title('yaw')
    axes[1, 0].set_xlabel('time (s)')
    axes[1, 0].set_ylabel('rad')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='best')
    axes[1, 1].plot(est[:, 0], left, label='left', lw=1.0)
    axes[1, 1].plot(est[:, 0], right, label='right', lw=1.0)
    axes[1, 1].set_title('thruster commands')
    axes[1, 1].set_xlabel('time (s)')
    axes[1, 1].set_ylabel('N')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='best')
    fig.suptitle('Controller Performance Dashboard')
    fig.tight_layout()
    fig.savefig(out_dir / '00_controller_dashboard.png', dpi=170)
    plt.close(fig)

    if est.shape[1] > 24 and not np.all(np.isnan(est[:, 19])):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(est[:, 0], est[:, 3], label='estimated yaw', lw=1.0)
        if not np.all(np.isnan(est[:, 18])):
            axes[0].plot(est[:, 0], est[:, 18], label='path psi_ref',
                         lw=1.0, ls=':')
        if not np.all(np.isnan(est[:, 13])):
            axes[0].plot(est[:, 0], est[:, 13], label='command psi_cmd',
                         lw=1.0, ls='-.')
        axes[0].axhline(args.target_yaw, color='tab:red', ls='--',
                        lw=0.9, label='final target yaw')
        axes[0].set_ylabel('yaw (rad)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')

        axes[1].plot(est[:, 0], est[:, 9],
                     label='final target yaw error', lw=1.0)
        axes[1].plot(est[:, 0], est[:, 19],
                     label='path yaw error', lw=1.0, ls=':')
        if not np.all(np.isnan(est[:, 20])):
            axes[1].plot(est[:, 0], est[:, 20],
                         label='command yaw error', lw=1.0, ls='-.')
        axes[1].axhline(0.0, color='black', lw=0.7)
        axes[1].set_ylabel('yaw error (rad)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        axes[2].plot(est[:, 0], est[:, 21],
                     label='along-track error e_s', lw=1.0)
        axes[2].plot(est[:, 0], est[:, 22],
                     label='cross-track error e_y', lw=1.0)
        axes[2].axhline(0.0, color='black', lw=0.7)
        axes[2].set_ylabel('path error (m)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best')

        axes[3].plot(est[:, 0], est[:, 23], label='u_ref', lw=1.0)
        axes[3].plot(est[:, 0], est[:, 24], label='measured surge u',
                     lw=1.0)
        axes[3].set_ylabel('surge speed (m/s)')
        axes[3].set_xlabel('time (s)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='best')

        fig.suptitle('Path Reference Tracking Diagnostics')
        fig.tight_layout()
        fig.savefig(out_dir / '07_path_reference_tracking.png', dpi=170)
        plt.close(fig)

    if est.shape[1] > 29 and not np.all(np.isnan(est[:, 27])):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(est[:, 0], est[:, 27],
                     label='max sampled |kappa|', lw=1.0)
        axes[0].axhline(args.kappa_max, color='tab:red', ls='--',
                        lw=0.9, label='configured kappa_max')
        axes[0].set_ylabel('1/m')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        axes[1].plot(est[:, 0], est[:, 28],
                     label='smoothing attempts', lw=1.0)
        axes[1].set_ylabel('attempts')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')
        axes[2].plot(est[:, 0], est[:, 29],
                     label='curvature feasible flag', lw=1.0)
        axes[2].set_ylabel('0/1')
        axes[2].set_xlabel('time (s)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best')
        fig.suptitle('Path Curvature Feasibility Diagnostics')
        fig.tight_layout()
        fig.savefig(out_dir / '08_path_curvature_feasibility.png', dpi=170)
        plt.close(fig)

    print(f'Wrote detailed controller plots to {out_dir}')


if __name__ == '__main__':
    main()
