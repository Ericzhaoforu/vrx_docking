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


def parse_bool_list(value: str):
    if not value:
        return []
    parsed = []
    for part in value.split(','):
        token = part.strip().lower()
        if not token:
            continue
        parsed.append(token in ('1', 'true', 'yes', 'y'))
    return parsed


def value_or_nan(row, key):
    value = row.get(key)
    if value is None or value == '':
        return float('nan')
    return float(value)


def rows_to_arrays(csv_path: Path):
    groups = {'estimate': [], 'truth': []}
    with csv_path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        numeric_names = [name for name in fieldnames if name != 'source']
        for row in reader:
            source = row.get('source')
            if source not in groups:
                continue
            groups[source].append([
                value_or_nan(row, name) for name in numeric_names
            ])
    return {
        source: np.asarray(values, dtype=float)
        for source, values in groups.items()
    }, numeric_names


def col(names, name):
    return names.index(name)


def draw_waypoint_yaw_arrows(ax, waypoints, length=0.85):
    for index, (wx, wy, wyaw, yaw_specified) in enumerate(waypoints):
        marker = 'D' if index == 0 else 'x'
        color = 'tab:orange' if index == 0 else 'tab:red'
        label = f'waypoint {index + 1}'
        ax.scatter([wx], [wy], marker=marker, s=90, c=color, label=label)
        if not yaw_specified:
            continue
        ax.arrow(
            wx, wy,
            length * math.cos(wyaw),
            length * math.sin(wyaw),
            width=0.035,
            head_width=0.22,
            head_length=0.28,
            length_includes_head=True,
            color=color,
            alpha=0.9,
        )


def valid_xy(data, names, x_name, y_name):
    if len(data) == 0:
        return np.empty((0, 2), dtype=float)
    xy = data[:, [col(names, x_name), col(names, y_name)]]
    return xy[~np.isnan(xy).any(axis=1)]


def nearest_pairs(est, truth, names):
    if len(est) == 0 or len(truth) == 0:
        return []
    t_col = col(names, 't')
    truth_t = truth[:, t_col]
    pairs = []
    for row in est:
        i = bisect_left(truth_t, row[t_col])
        candidates = [j for j in (i - 1, i, i + 1) if 0 <= j < len(truth)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(truth_t[k] - row[t_col]))
        if abs(truth_t[j] - row[t_col]) <= 0.1:
            pairs.append((row, truth[j]))
    return pairs


def plot_dashboard(est, truth, names, out_dir, target, target_yaw,
                   waypoints, half_spacing):
    t = col(names, 't')
    x = col(names, 'x')
    y = col(names, 'y')
    yaw = col(names, 'yaw')
    distance = col(names, 'distance')
    left = col(names, 'left_thrust')
    right = col(names, 'right_thrust')
    x_d = col(names, 'x_d')
    y_d = col(names, 'y_d')
    psi_d = col(names, 'psi_d')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ref_xy = valid_xy(est, names, 'x_d', 'y_d')
    if len(ref_xy):
        axes[0, 0].plot(ref_xy[:, 0], ref_xy[:, 1],
                        label='time reference', lw=1.4, ls='--')
    if len(truth):
        axes[0, 0].plot(truth[:, x], truth[:, y],
                        label='ground truth aligned', lw=1.4)
    if len(est):
        axes[0, 0].plot(est[:, x], est[:, y], label='EKF estimate', lw=1.1)
    axes[0, 0].scatter([target[0]], [target[1]], marker='x', s=120,
                       c='tab:red', label='target')
    draw_waypoint_yaw_arrows(axes[0, 0], waypoints)
    axes[0, 0].axis('equal')
    axes[0, 0].set_title('trajectory')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='best')

    if len(truth):
        axes[0, 1].plot(truth[:, t], truth[:, distance],
                        label='truth distance', lw=1.3)
    if len(est):
        axes[0, 1].plot(est[:, t], est[:, distance],
                        label='estimate distance', lw=1.1)
    axes[0, 1].axhline(0.20, color='tab:gray', ls='--', lw=1.0,
                       label='0.20 m target')
    axes[0, 1].set_title('distance to target')
    axes[0, 1].set_ylabel('m')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='best')

    if len(est):
        axes[1, 0].plot(est[:, t], est[:, yaw], label='yaw', lw=1.0)
        axes[1, 0].plot(est[:, t], est[:, psi_d],
                        label='psi_d', lw=1.0, ls=':')
    axes[1, 0].axhline(target_yaw, color='tab:red', ls='--',
                       lw=1.0, label='target yaw')
    axes[1, 0].set_title('yaw reference tracking')
    axes[1, 0].set_ylabel('rad')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='best')

    if len(est):
        axes[1, 1].plot(est[:, t], est[:, left], label='left', lw=1.0)
        axes[1, 1].plot(est[:, t], est[:, right], label='right', lw=1.0)
        axes[1, 1].plot(
            est[:, t],
            half_spacing * (est[:, right] - est[:, left]),
            label='allocated yaw moment', lw=0.9, ls=':')
    axes[1, 1].set_title('actuator commands')
    axes[1, 1].set_xlabel('time (s)')
    axes[1, 1].set_ylabel('N / N*m')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='best')

    fig.suptitle('Trajectory Tracker Dashboard')
    fig.tight_layout()
    fig.savefig(out_dir / '00_controller_dashboard.png', dpi=170)
    plt.close(fig)

    if len(est):
        fig, ax = plt.subplots(figsize=(8, 8))
        if len(ref_xy):
            ax.plot(ref_xy[:, 0], ref_xy[:, 1],
                    label='time-parameterized reference', lw=1.4, ls='--')
        ax.plot(est[:, x], est[:, y], label='EKF estimate', lw=1.1)
        if len(truth):
            ax.plot(truth[:, x], truth[:, y],
                    label='ground truth aligned', lw=1.3)
        ax.scatter([target[0]], [target[1]], marker='x', s=120,
                   c='tab:red', label='target')
        draw_waypoint_yaw_arrows(ax, waypoints)
        if len(est):
            ax.scatter([est[0, x]], [est[0, y]], marker='o', s=55,
                       c='tab:green', label='start')
            ax.scatter([est[-1, x]], [est[-1, y]], marker='s', s=55,
                       c='tab:purple', label='final estimate')
        ax.axis('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(out_dir / '01_trajectory_xy.png', dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, name, target_value in zip(
            axes, ('x', 'y', 'yaw'), (target[0], target[1], target_yaw)):
        c = col(names, name)
        if len(truth):
            axis.plot(truth[:, t], truth[:, c],
                      label='ground truth aligned', lw=1.3)
        if len(est):
            axis.plot(est[:, t], est[:, c], label='EKF estimate', lw=1.0)
            ref_name = f'{name}_d' if name in ('x', 'y') else 'psi_d'
            axis.plot(est[:, t], est[:, col(names, ref_name)],
                      label=f'{ref_name}', lw=1.0, ls=':')
        axis.axhline(target_value, color='tab:red', ls='--', lw=0.9)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    axes[0].legend(loc='best')
    fig.suptitle('State Tracking')
    fig.tight_layout()
    fig.savefig(out_dir / '02_states_x_y_yaw.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    if len(est):
        axes[0].plot(est[:, t], est[:, col(names, 'u_d')],
                     label='u_d', lw=1.0)
        axes[0].plot(est[:, t], est[:, col(names, 'u_cmd')],
                     label='u_cmd', lw=1.0)
        axes[0].plot(est[:, t], est[:, col(names, 'surge_speed')],
                     label='measured surge u', lw=1.0)
        axes[1].plot(est[:, t], est[:, col(names, 'r_d')],
                     label='r_d', lw=1.0)
        axes[1].plot(est[:, t], est[:, col(names, 'r_cmd')],
                     label='r_cmd', lw=1.0)
        axes[1].plot(est[:, t], est[:, col(names, 'yaw_rate')],
                     label='measured yaw rate', lw=1.0)
        axes[2].plot(est[:, t], est[:, col(names, 'trajectory_u_limit')],
                     label='combined u limit', lw=1.0)
        axes[2].plot(est[:, t],
                     est[:, col(names, 'trajectory_curvature_limit')],
                     label='curvature speed limit', lw=1.0)
        axes[2].plot(est[:, t],
                     est[:, col(names, 'trajectory_yaw_rate_limit')],
                     label='yaw-rate speed limit', lw=1.0)
    axes[0].set_ylabel('m/s')
    axes[1].set_ylabel('rad/s')
    axes[2].set_ylabel('m/s')
    axes[2].set_xlabel('time (s)')
    for axis in axes:
        axis.axhline(0.0, color='black', lw=0.6)
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best')
    fig.suptitle('Trajectory Speed and Rate Tracking')
    fig.tight_layout()
    fig.savefig(out_dir / '03_speed_rate_tracking.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axis, name, ylabel in zip(
            axes, ('e_x', 'e_y', 'e_psi'),
            ('body x error (m)', 'body y error (m)', 'yaw error (rad)')):
        if len(est):
            axis.plot(est[:, t], est[:, col(names, name)], lw=1.0)
        axis.axhline(0.0, color='black', lw=0.7)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle('Body-Frame Tracking Errors')
    fig.tight_layout()
    fig.savefig(out_dir / '04_body_frame_errors.png', dpi=170)
    plt.close(fig)

    pairs = nearest_pairs(est, truth, names)
    if pairs:
        times = np.array([pair[0][t] for pair in pairs])
        err_x = np.array([pair[0][x] - pair[1][x] for pair in pairs])
        err_y = np.array([pair[0][y] - pair[1][y] for pair in pairs])
        err_yaw = np.array([
            wrap_angle(pair[0][yaw] - pair[1][yaw]) for pair in pairs
        ])
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for axis, data, ylabel in [
                (axes[0], err_x, 'x error (m)'),
                (axes[1], err_y, 'y error (m)'),
                (axes[2], err_yaw, 'yaw error (rad)'),
                (axes[3], np.hypot(err_x, err_y), 'position error norm (m)')]:
            axis.plot(times, data, lw=1.0)
            axis.axhline(0.0, color='black', lw=0.7)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        axes[-1].set_xlabel('time (s)')
        fig.suptitle('EKF Estimate Error vs Aligned Ground Truth')
        fig.tight_layout()
        fig.savefig(out_dir / '05_estimation_errors.png', dpi=170)
        plt.close(fig)

    if len(est):
        left_force = est[:, left]
        right_force = est[:, right]
        allocated_force = left_force + right_force
        allocated_torque = half_spacing * (right_force - left_force)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(est[:, t], left_force, label='left thrust', lw=1.0)
        axes[0].plot(est[:, t], right_force, label='right thrust', lw=1.0)
        axes[0].axhline(500, color='tab:gray', ls='--', lw=0.8)
        axes[0].axhline(-500, color='tab:gray', ls='--', lw=0.8)
        axes[0].set_ylabel('N')
        axes[0].legend(loc='best')
        axes[1].plot(est[:, t], est[:, col(names, 'force')],
                     label='desired F_x', lw=1.0)
        axes[1].plot(est[:, t], allocated_force,
                     label='allocated F_x', lw=1.0, ls=':')
        axes[1].set_ylabel('N')
        axes[1].legend(loc='best')
        axes[2].plot(est[:, t], est[:, col(names, 'torque')],
                     label='desired tau_z', lw=1.0)
        axes[2].plot(est[:, t], allocated_torque,
                     label='allocated tau_z', lw=1.0, ls=':')
        axes[2].set_ylabel('N*m')
        axes[2].legend(loc='best')
        axes[3].plot(est[:, t], est[:, col(names, 'allocation_cost')],
                     label='allocation residual cost', lw=1.0)
        axes[3].set_ylabel('cost')
        axes[3].set_xlabel('time (s)')
        axes[3].legend(loc='best')
        for axis in axes:
            axis.grid(True, alpha=0.3)
        fig.suptitle('Thruster Forces and Allocation Residuals')
        fig.tight_layout()
        fig.savefig(out_dir / '06_thruster_allocation.png', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--target-x', type=float, default=6.0)
    parser.add_argument('--target-y', type=float, default=12.0)
    parser.add_argument('--target-yaw', type=float, default=2.4)
    parser.add_argument('--thruster-half-spacing', type=float, default=1.027135)
    parser.add_argument('--waypoint-xs', default='4.0,6.0')
    parser.add_argument('--waypoint-ys', default='8.0,12.0')
    parser.add_argument('--waypoint-yaws', default='0.0,2.4')
    parser.add_argument('--waypoint-yaw-specified', default='false,true')
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = args.output_dir or csv_path.parent / 'detailed_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, names = rows_to_arrays(csv_path)
    est = groups['estimate']
    truth = groups['truth']
    if len(est) == 0:
        raise RuntimeError('No estimate rows found in controller CSV.')

    waypoint_xs = parse_float_list(args.waypoint_xs)
    waypoint_ys = parse_float_list(args.waypoint_ys)
    waypoint_yaws = parse_float_list(args.waypoint_yaws)
    waypoint_yaw_specified = parse_bool_list(args.waypoint_yaw_specified)
    if len(waypoint_yaw_specified) != len(waypoint_xs):
        waypoint_yaw_specified = [True] * len(waypoint_xs)
    waypoints = [
        (x, y, yaw, yaw_specified)
        for x, y, yaw, yaw_specified
        in zip(waypoint_xs, waypoint_ys, waypoint_yaws,
               waypoint_yaw_specified)
    ]

    plot_dashboard(
        est,
        truth,
        names,
        out_dir,
        np.array([args.target_x, args.target_y], dtype=float),
        args.target_yaw,
        waypoints,
        args.thruster_half_spacing,
    )
    print(f'Wrote detailed controller plots to {out_dir}')


if __name__ == '__main__':
    main()
