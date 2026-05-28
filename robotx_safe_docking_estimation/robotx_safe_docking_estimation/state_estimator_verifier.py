import math
from typing import Optional

import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robotx_safe_docking_estimation.gps_imu_ekf_node import yaw_from_quaternion
from robotx_safe_docking_estimation.gps_imu_ekf_node import wrap_angle


class StateEstimatorVerifier(Node):
    """Ground-truth comparison node for development only.

    This node intentionally lives outside the estimator. It subscribes to
    Gazebo pose output for RMSE reporting, so it must not be used by autonomy.
    """

    def __init__(self):
        super().__init__('state_estimator_verifier')
        self.declare_parameter('estimate_topic', '/safe_docking/state')
        self.declare_parameter(
            'ground_truth_topic',
            '/wamv/sensors/position/ground_truth_odometry')
        self.declare_parameter('report_period_sec', 5.0)

        self.estimate_topic = self.get_parameter(
            'estimate_topic').get_parameter_value().string_value
        self.ground_truth_topic = self.get_parameter(
            'ground_truth_topic').get_parameter_value().string_value
        report_period = self.get_parameter(
            'report_period_sec').get_parameter_value().double_value

        self.latest_estimate: Optional[np.ndarray] = None
        self.latest_truth: Optional[np.ndarray] = None
        self.latest_truth_raw: Optional[np.ndarray] = None
        self.truth_to_estimate_offset: Optional[np.ndarray] = None
        self.errors = []

        self.create_subscription(
            Float64MultiArray, self.estimate_topic, self.on_estimate, 10)
        self.create_subscription(
            Odometry, self.ground_truth_topic, self.on_ground_truth, 10)
        self.create_timer(report_period, self.report)

        self.get_logger().warn(
            'Verifier subscribes to Gazebo ground truth for evaluation only. '
            'Do not use it inside the autonomy stack.')

    def on_estimate(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return
        self.latest_estimate = np.array(msg.data[:6], dtype=float)
        self._try_initialize_truth_alignment()
        self._sample_error()

    def on_ground_truth(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        truth = np.array([x, y, yaw], dtype=float)
        self.latest_truth_raw = truth
        self._try_initialize_truth_alignment()
        if self.truth_to_estimate_offset is None:
            return
        truth_local = truth.copy()
        truth_local[:2] += self.truth_to_estimate_offset
        truth_local[2] = wrap_angle(truth_local[2])
        self.latest_truth = truth_local
        self._sample_error()

    def _try_initialize_truth_alignment(self):
        if self.truth_to_estimate_offset is not None:
            return
        if self.latest_estimate is None or self.latest_truth_raw is None:
            return
        self.truth_to_estimate_offset = (
            self.latest_estimate[:2] - self.latest_truth_raw[:2])
        self.get_logger().info(
            'Aligned ground truth into EKF local frame with offset: '
            f'x={self.truth_to_estimate_offset[0]:.3f} m, '
            f'y={self.truth_to_estimate_offset[1]:.3f} m')

    def _sample_error(self):
        if self.latest_estimate is None or self.latest_truth is None:
            return
        error = np.array([
            self.latest_estimate[0] - self.latest_truth[0],
            self.latest_estimate[1] - self.latest_truth[1],
            wrap_angle(self.latest_estimate[2] - self.latest_truth[2]),
        ])
        self.errors.append(error)

    def report(self):
        if not self.errors:
            self.get_logger().info('Waiting for estimate and ground truth...')
            return
        errors = np.array(self.errors)
        rmse = np.sqrt(np.mean(errors * errors, axis=0))
        latest = errors[-1]
        self.get_logger().info(
            'Estimator RMSE vs debug ground truth: '
            f'x={rmse[0]:.3f} m, y={rmse[1]:.3f} m, '
            f'yaw={rmse[2]:.3f} rad; latest error '
            f'x={latest[0]:.3f}, y={latest[1]:.3f}, yaw={latest[2]:.3f}; '
            f'samples={len(self.errors)}')

    def destroy_node(self):
        self.report()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorVerifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
