#!/usr/bin/env python3
"""Spawn visible Gazebo entities for the NMPC reference and WAM-V trail."""

import argparse
import math
import subprocess
from collections import deque

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robotx_safe_docking_control.feasible_references import (
    make_synthetic_reference,
)
from robotx_safe_docking_control.flatness_mpc_controller import DEBUG_FIELDS
from robotx_safe_docking_control.flatness_mpc_controller import yaw_from_quaternion
from robotx_safe_docking_control.usv_flatness import UsvModelParams


def escape_proto_string(text):
    return text.replace('\\', '\\\\').replace('"', '\\"')


class GazeboEntityTrailVisualizer(Node):
    def __init__(self, args):
        super().__init__('gazebo_nmpc_entity_trail_visualizer')
        self.args = args
        self.world_offset = np.array(
            [args.world_offset_x, args.world_offset_y, args.world_offset_z],
            dtype=float)
        self.static_reference_spawned = bool(args.skip_static_reference)
        self.spawned_count = 0
        self.actual_count = 0
        self.active_ref_count = 0
        self.last_actual_xy = None
        self.last_active_ref_xy = None
        self.pending_static_initial_z = None
        self.spawn_log = deque(maxlen=20)

        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        self.create_subscription(
            Float64MultiArray, args.debug_topic, self.on_debug, 10)

        self.get_logger().info(
            'Gazebo entity trail visualizer active. '
            f'world={args.world_name}, service={self.create_service_name}')

    @property
    def create_service_name(self):
        return f'/world/{self.args.world_name}/create'

    def on_odom(self, msg):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        psi = yaw_from_quaternion(msg.pose.pose.orientation)

        if not self.static_reference_spawned:
            self.static_reference_spawned = True
            self.spawn_static_reference(np.array([x, y, psi], dtype=float))

        if self.last_actual_xy is not None:
            if math.hypot(x - self.last_actual_xy[0],
                          y - self.last_actual_xy[1]) < self.args.min_step_m:
                return
        self.last_actual_xy = (x, y)
        self.actual_count += 1
        self.spawn_sphere(
            f'nmpc_actual_trail_{self.actual_count:04d}',
            x,
            y,
            self.args.actual_z_offset,
            self.args.actual_radius,
            self.args.actual_color)

    def on_debug(self, msg):
        if len(msg.data) < len(DEBUG_FIELDS):
            return
        values = {
            field: float(msg.data[index])
            for index, field in enumerate(DEBUG_FIELDS)
        }
        x = values.get('active_x_ref', float('nan'))
        y = values.get('active_y_ref', float('nan'))
        if not all(math.isfinite(v) for v in (x, y)):
            return
        if self.last_active_ref_xy is not None:
            if math.hypot(x - self.last_active_ref_xy[0],
                          y - self.last_active_ref_xy[1]) < self.args.min_step_m:
                return
        self.last_active_ref_xy = (x, y)
        self.active_ref_count += 1
        self.spawn_sphere(
            f'nmpc_active_reference_{self.active_ref_count:04d}',
            x,
            y,
            self.args.active_reference_z_offset,
            self.args.active_reference_radius,
            self.args.active_reference_color)

    def spawn_static_reference(self, initial_z):
        params = UsvModelParams(
            mass=self.args.mass,
            iz=self.args.iz,
            du=self.args.du,
            duu=self.args.duu,
            dv=self.args.dv,
            dvv=self.args.dvv,
            dr=self.args.dr,
            drr=self.args.drr,
            thruster_half_spacing=self.args.thruster_half_spacing,
            min_thrust=self.args.min_thrust,
            max_thrust=self.args.max_thrust,
        )
        reference = make_synthetic_reference(
            self.args.reference_name,
            initial_z,
            params,
            dt=self.args.reference_dt,
            initial_nu=np.zeros(3, dtype=float))
        points = reference.z[::self.args.reference_stride]
        self.get_logger().info(
            f'Spawning {len(points)} static reference points for '
            f'[{self.args.reference_name}], duration={reference.duration:.2f}s')
        for i, row in enumerate(points):
            self.spawn_sphere(
                f'nmpc_full_reference_{i:04d}',
                float(row[0]),
                float(row[1]),
                self.args.reference_z_offset,
                self.args.reference_radius,
                self.args.reference_color)

    def spawn_sphere(self, name, local_x, local_y, z_offset, radius, color):
        world_x = self.world_offset[0] + float(local_x)
        world_y = self.world_offset[1] + float(local_y)
        world_z = self.world_offset[2] + float(z_offset)
        sdf = self.sphere_sdf(name, radius, color)
        request = (
            f'sdf: "{escape_proto_string(sdf)}" '
            f'pose: {{ position: {{ x: {world_x:.6f} y: {world_y:.6f} '
            f'z: {world_z:.6f} }} }} '
            f'name: "{name}" allow_renaming: true')
        cmd = [
            self.args.gz_service_bin,
            '-s', self.create_service_name,
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', str(self.args.service_timeout_ms),
            '--req', request,
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(self.args.service_timeout_ms / 1000.0 + 1.0, 1.0),
                check=False)
        except subprocess.TimeoutExpired:
            self.get_logger().warning(f'Timed out spawning {name}')
            return
        if result.returncode != 0:
            self.spawn_log.append(result.stdout.strip())
            self.get_logger().warning(
                f'Failed to spawn {name}, rc={result.returncode}: '
                f'{result.stdout.strip()[:180]}')
            return
        self.spawned_count += 1

    @staticmethod
    def sphere_sdf(name, radius, color):
        r, g, b, a = color
        rgba = f'{r:.3f} {g:.3f} {b:.3f} {a:.3f}'
        return (
            '<?xml version="1.0" ?>'
            '<sdf version="1.7">'
            f'<model name="{name}">'
            '<static>true</static>'
            '<link name="link">'
            '<visual name="visual">'
            '<cast_shadows>false</cast_shadows>'
            '<geometry>'
            f'<sphere><radius>{radius:.4f}</radius></sphere>'
            '</geometry>'
            '<material>'
            f'<ambient>{rgba}</ambient>'
            f'<diffuse>{rgba}</diffuse>'
            f'<emissive>{rgba}</emissive>'
            '</material>'
            '</visual>'
            '</link>'
            '</model>'
            '</sdf>')


def color_arg(value):
    parts = [float(part) for part in value.split(',')]
    if len(parts) == 3:
        parts.append(1.0)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('color must be r,g,b[,a]')
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--odom-topic', default='/safe_docking/odometry')
    parser.add_argument(
        '--debug-topic', default='/safe_docking/flatness_mpc/debug')
    parser.add_argument('--world-name', default='safe_docking_task_recording')
    parser.add_argument(
        '--gz-service-bin',
        default='/usr/libexec/gz/transport12/gz-transport-service')
    parser.add_argument('--service-timeout-ms', type=int, default=1000)
    parser.add_argument('--reference-name', default='figure8')
    parser.add_argument('--skip-static-reference', action='store_true')
    parser.add_argument('--reference-dt', type=float, default=0.2)
    parser.add_argument('--reference-stride', type=int, default=8)
    parser.add_argument('--min-step-m', type=float, default=0.25)
    parser.add_argument('--world-offset-x', type=float, default=-532.0)
    parser.add_argument('--world-offset-y', type=float, default=162.0)
    parser.add_argument('--world-offset-z', type=float, default=0.35)
    parser.add_argument('--reference-z-offset', type=float, default=1.55)
    parser.add_argument('--active-reference-z-offset', type=float, default=1.85)
    parser.add_argument('--actual-z-offset', type=float, default=2.15)
    parser.add_argument('--reference-radius', type=float, default=0.22)
    parser.add_argument('--active-reference-radius', type=float, default=0.16)
    parser.add_argument('--actual-radius', type=float, default=0.18)
    parser.add_argument(
        '--reference-color', type=color_arg, default=(1.0, 0.35, 0.0, 1.0))
    parser.add_argument(
        '--active-reference-color', type=color_arg,
        default=(1.0, 0.95, 0.05, 1.0))
    parser.add_argument(
        '--actual-color', type=color_arg, default=(0.0, 0.95, 1.0, 1.0))

    parser.add_argument('--mass', type=float, default=180.0)
    parser.add_argument('--iz', type=float, default=446.0)
    parser.add_argument('--du', type=float, default=100.0)
    parser.add_argument('--duu', type=float, default=150.0)
    parser.add_argument('--dv', type=float, default=100.0)
    parser.add_argument('--dvv', type=float, default=100.0)
    parser.add_argument('--dr', type=float, default=800.0)
    parser.add_argument('--drr', type=float, default=800.0)
    parser.add_argument('--thruster-half-spacing', type=float, default=1.027135)
    parser.add_argument('--min-thrust', type=float, default=-100.0)
    parser.add_argument('--max-thrust', type=float, default=100.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = GazeboEntityTrailVisualizer(args)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
