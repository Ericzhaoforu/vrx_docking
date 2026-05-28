import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import NavSatStatus
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


WGS84_A = 6378137.0


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    # REP 103 yaw, assuming ENU convention.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class LocalCartesian:
    def __init__(self, latitude_deg: float, longitude_deg: float,
                 altitude_m: float):
        self.lat0 = math.radians(latitude_deg)
        self.lon0 = math.radians(longitude_deg)
        self.alt0 = altitude_m
        self.cos_lat0 = math.cos(self.lat0)

    def forward(self, latitude_deg: float, longitude_deg: float,
                altitude_m: float):
        lat = math.radians(latitude_deg)
        lon = math.radians(longitude_deg)
        x = (lon - self.lon0) * self.cos_lat0 * WGS84_A
        y = (lat - self.lat0) * WGS84_A
        z = altitude_m - self.alt0
        return x, y, z


class GpsImuEkfNode(Node):
    """Planar inertial EKF for local-frame WAM-V state.

    The estimator uses GPS only to define and measure local ENU position. It
    uses IMU acceleration and yaw rate for prediction, estimates accelerometer
    and gyro bias, then optionally updates yaw from IMU orientation as a
    compass / AHRS placeholder. It does not subscribe to Gazebo ground truth.
    """

    IDX_X = 0
    IDX_Y = 1
    IDX_PSI = 2
    IDX_X_DOT = 3
    IDX_Y_DOT = 4
    IDX_B_AX = 5
    IDX_B_AY = 6
    IDX_B_GZ = 7
    STATE_SIZE = 8

    def __init__(self):
        super().__init__('gps_imu_ekf_node')

        self.declare_parameter(
            'gps_topic', '/wamv/sensors/gps/gps/fix')
        self.declare_parameter(
            'imu_topic', '/wamv/sensors/imu/imu/data')
        self.declare_parameter('state_topic', '/safe_docking/state')
        self.declare_parameter('odom_topic', '/safe_docking/odometry')
        self.declare_parameter('origin_frame', 'safe_docking/odom')
        self.declare_parameter('base_frame', 'wamv/wamv/base_link')
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('gps_body_x', -0.85)
        self.declare_parameter('gps_body_y', 0.0)
        self.declare_parameter('use_imu_orientation', True)
        self.declare_parameter('use_gps_velocity_measurement', False)
        self.declare_parameter('origin_init_duration_sec', 5.0)
        self.declare_parameter('origin_min_samples', 20)
        self.declare_parameter('origin_max_std_m', 0.5)

        self.declare_parameter('gps_position_variance', 0.25)
        self.declare_parameter('gps_velocity_variance', 0.5)
        self.declare_parameter('imu_yaw_variance', 0.02)
        self.declare_parameter('acceleration_noise_variance', 0.25)
        self.declare_parameter('gyro_noise_variance', 0.0004)
        self.declare_parameter('yaw_process_variance', 0.01)
        self.declare_parameter('accel_bias_random_walk_variance', 0.0004)
        self.declare_parameter('gyro_bias_random_walk_variance', 0.000001)
        self.declare_parameter('unmodeled_velocity_process_variance', 0.02)

        self.gps_topic = self.get_parameter(
            'gps_topic').get_parameter_value().string_value
        self.imu_topic = self.get_parameter(
            'imu_topic').get_parameter_value().string_value
        self.state_topic = self.get_parameter(
            'state_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter(
            'odom_topic').get_parameter_value().string_value
        self.origin_frame = self.get_parameter(
            'origin_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter(
            'base_frame').get_parameter_value().string_value
        self.publish_tf = self.get_parameter(
            'publish_tf').get_parameter_value().bool_value
        self.gps_body_x = self.get_parameter(
            'gps_body_x').get_parameter_value().double_value
        self.gps_body_y = self.get_parameter(
            'gps_body_y').get_parameter_value().double_value
        self.use_imu_orientation = self.get_parameter(
            'use_imu_orientation').get_parameter_value().bool_value
        self.use_gps_velocity_measurement = self.get_parameter(
            'use_gps_velocity_measurement').get_parameter_value().bool_value
        self.origin_init_duration_sec = self.get_parameter(
            'origin_init_duration_sec').get_parameter_value().double_value
        self.origin_min_samples = self.get_parameter(
            'origin_min_samples').get_parameter_value().integer_value
        self.origin_max_std_m = self.get_parameter(
            'origin_max_std_m').get_parameter_value().double_value

        self.gps_position_variance = self.get_parameter(
            'gps_position_variance').get_parameter_value().double_value
        self.gps_velocity_variance = self.get_parameter(
            'gps_velocity_variance').get_parameter_value().double_value
        self.imu_yaw_variance = self.get_parameter(
            'imu_yaw_variance').get_parameter_value().double_value
        self.acceleration_noise_variance = self.get_parameter(
            'acceleration_noise_variance').get_parameter_value().double_value
        self.gyro_noise_variance = self.get_parameter(
            'gyro_noise_variance').get_parameter_value().double_value
        self.yaw_process_variance = self.get_parameter(
            'yaw_process_variance').get_parameter_value().double_value
        self.accel_bias_random_walk_variance = self.get_parameter(
            'accel_bias_random_walk_variance'
        ).get_parameter_value().double_value
        self.gyro_bias_random_walk_variance = self.get_parameter(
            'gyro_bias_random_walk_variance'
        ).get_parameter_value().double_value
        self.unmodeled_velocity_process_variance = self.get_parameter(
            'unmodeled_velocity_process_variance'
        ).get_parameter_value().double_value

        self.local_cartesian: Optional[LocalCartesian] = None
        self.origin_fix: Optional[NavSatFix] = None
        self.origin_samples = []
        self.origin_start_time: Optional[float] = None
        self.last_origin_wait_log_time = -float('inf')
        self.initial_yaw: Optional[float] = None
        self.last_predict_time: Optional[float] = None
        self.last_gps_time: Optional[float] = None
        self.last_base_xy_meas: Optional[np.ndarray] = None
        self.last_yaw_rate = 0.0

        self.x = np.zeros(self.STATE_SIZE)
        self.p = np.diag([4.0, 4.0, 0.5, 2.0, 2.0, 0.25, 0.25, 0.05])

        self.state_pub = self.create_publisher(
            Float64MultiArray, self.state_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(
            self) if self.publish_tf else None

        self.create_subscription(
            NavSatFix, self.gps_topic, self.on_gps, qos_profile_sensor_data)
        self.create_subscription(
            Imu, self.imu_topic, self.on_imu, qos_profile_sensor_data)

        self.get_logger().info(
            f'GPS+IMU EKF collecting GPS origin samples on '
            f'{self.gps_topic}; IMU topic: {self.imu_topic}')

    def on_gps(self, msg: NavSatFix):
        if not self._valid_fix(msg):
            return

        stamp = stamp_to_sec(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        if self.local_cartesian is None:
            self._collect_origin_sample(msg, stamp)
            return

        gps_x, gps_y, _ = self.local_cartesian.forward(
            msg.latitude, msg.longitude, msg.altitude)
        gps_delta = np.array([gps_x, gps_y])
        if self.initial_yaw is None:
            self.last_gps_time = stamp
            return

        z = self._base_position_measurement(gps_delta, self.x[self.IDX_PSI])
        h = np.zeros((2, self.STATE_SIZE))
        h[0, self.IDX_X] = 1.0
        h[1, self.IDX_Y] = 1.0
        r = np.eye(2) * self._gps_position_variance(msg)
        self._update(z, h, r)

        if (self.use_gps_velocity_measurement and
                self.last_gps_time is not None and
                self.last_base_xy_meas is not None):
            dt = stamp - self.last_gps_time
            if 0.02 <= dt <= 2.0:
                vel = (z - self.last_base_xy_meas) / dt
                h_vel = np.zeros((2, self.STATE_SIZE))
                h_vel[0, self.IDX_X_DOT] = 1.0
                h_vel[1, self.IDX_Y_DOT] = 1.0
                r_vel = np.eye(2) * self.gps_velocity_variance
                self._update(vel, h_vel, r_vel)

        self.last_gps_time = stamp
        self.last_base_xy_meas = z
        self.publish_state(msg.header.stamp)

    def on_imu(self, msg: Imu):
        if self.local_cartesian is None:
            return

        stamp = stamp_to_sec(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        if self.last_predict_time is None:
            self.last_predict_time = stamp
            if self.use_imu_orientation and self._imu_has_orientation(msg):
                self.x[self.IDX_PSI] = yaw_from_quaternion(msg.orientation)
                self._set_initial_yaw(self.x[self.IDX_PSI])
            self.last_yaw_rate = msg.angular_velocity.z - self.x[self.IDX_B_GZ]
            self.publish_state(msg.header.stamp)
            return

        dt = stamp - self.last_predict_time
        if dt <= 0.0:
            return
        if dt > 0.5:
            self.get_logger().warn(
                f'Large IMU gap {dt:.3f}s; limiting EKF prediction step.')
            dt = 0.5
        self.last_predict_time = stamp

        self._predict(msg, dt)

        if self.use_imu_orientation and self._imu_has_orientation(msg):
            yaw = yaw_from_quaternion(msg.orientation)
            self._set_initial_yaw(yaw)
            h_yaw = np.zeros((1, self.STATE_SIZE))
            h_yaw[0, self.IDX_PSI] = 1.0
            self._update_angle(
                np.array([yaw]), h_yaw,
                np.array([[self._imu_yaw_variance(msg)]]), angle_index=0)

        self.x[self.IDX_PSI] = wrap_angle(self.x[self.IDX_PSI])
        self.last_yaw_rate = msg.angular_velocity.z - self.x[self.IDX_B_GZ]
        self.publish_state(msg.header.stamp)

    def _predict(self, msg: Imu, dt: float):
        psi = self.x[self.IDX_PSI]
        c = math.cos(psi)
        s = math.sin(psi)

        ax_body = msg.linear_acceleration.x - self.x[self.IDX_B_AX]
        ay_body = msg.linear_acceleration.y - self.x[self.IDX_B_AY]
        yaw_rate = msg.angular_velocity.z - self.x[self.IDX_B_GZ]
        ax = c * ax_body - s * ay_body
        ay = s * ax_body + c * ay_body

        d_ax_d_psi = -s * ax_body - c * ay_body
        d_ay_d_psi = c * ax_body - s * ay_body

        self.x[self.IDX_X] += (
            self.x[self.IDX_X_DOT] * dt + 0.5 * ax * dt * dt)
        self.x[self.IDX_Y] += (
            self.x[self.IDX_Y_DOT] * dt + 0.5 * ay * dt * dt)
        self.x[self.IDX_PSI] = wrap_angle(
            self.x[self.IDX_PSI] + yaw_rate * dt)
        self.x[self.IDX_X_DOT] += ax * dt
        self.x[self.IDX_Y_DOT] += ay * dt

        f = np.eye(self.STATE_SIZE)
        f[self.IDX_X, self.IDX_PSI] = 0.5 * d_ax_d_psi * dt * dt
        f[self.IDX_X, self.IDX_X_DOT] = dt
        f[self.IDX_X, self.IDX_B_AX] = -0.5 * c * dt * dt
        f[self.IDX_X, self.IDX_B_AY] = 0.5 * s * dt * dt
        f[self.IDX_Y, self.IDX_PSI] = 0.5 * d_ay_d_psi * dt * dt
        f[self.IDX_Y, self.IDX_Y_DOT] = dt
        f[self.IDX_Y, self.IDX_B_AX] = -0.5 * s * dt * dt
        f[self.IDX_Y, self.IDX_B_AY] = -0.5 * c * dt * dt
        f[self.IDX_PSI, self.IDX_B_GZ] = -dt
        f[self.IDX_X_DOT, self.IDX_PSI] = d_ax_d_psi * dt
        f[self.IDX_X_DOT, self.IDX_B_AX] = -c * dt
        f[self.IDX_X_DOT, self.IDX_B_AY] = s * dt
        f[self.IDX_Y_DOT, self.IDX_PSI] = d_ay_d_psi * dt
        f[self.IDX_Y_DOT, self.IDX_B_AX] = -s * dt
        f[self.IDX_Y_DOT, self.IDX_B_AY] = -c * dt

        q = np.zeros((self.STATE_SIZE, self.STATE_SIZE))
        accel_q = self.acceleration_noise_variance
        q[self.IDX_X, self.IDX_X] = 0.25 * accel_q * dt**4
        q[self.IDX_Y, self.IDX_Y] = 0.25 * accel_q * dt**4
        q[self.IDX_X_DOT, self.IDX_X_DOT] = (
            accel_q * dt**2 + self.unmodeled_velocity_process_variance * dt)
        q[self.IDX_Y_DOT, self.IDX_Y_DOT] = (
            accel_q * dt**2 + self.unmodeled_velocity_process_variance * dt)
        q[self.IDX_PSI, self.IDX_PSI] = (
            self.gyro_noise_variance * dt**2 + self.yaw_process_variance * dt)
        q[self.IDX_B_AX, self.IDX_B_AX] = (
            self.accel_bias_random_walk_variance * dt)
        q[self.IDX_B_AY, self.IDX_B_AY] = (
            self.accel_bias_random_walk_variance * dt)
        q[self.IDX_B_GZ, self.IDX_B_GZ] = (
            self.gyro_bias_random_walk_variance * dt)

        self.p = f @ self.p @ f.T + q
        self.p = 0.5 * (self.p + self.p.T)

    def _update(self, z: np.ndarray, h: np.ndarray, r: np.ndarray):
        innovation = z - h @ self.x
        self._apply_update(innovation, h, r)

    def _update_angle(self, z: np.ndarray, h: np.ndarray, r: np.ndarray,
                      angle_index: int):
        innovation = z - h @ self.x
        innovation[angle_index] = wrap_angle(innovation[angle_index])
        self._apply_update(innovation, h, r)

    def _apply_update(self, innovation: np.ndarray, h: np.ndarray,
                      r: np.ndarray):
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.x[self.IDX_PSI] = wrap_angle(self.x[self.IDX_PSI])
        i = np.eye(self.STATE_SIZE)
        # Joseph form keeps covariance symmetric / positive semidefinite.
        self.p = (i - k @ h) @ self.p @ (i - k @ h).T + k @ r @ k.T
        self.p = 0.5 * (self.p + self.p.T)

    def publish_state(self, stamp):
        primary_state = self._primary_state()
        msg = Float64MultiArray()
        msg.data = [float(v) for v in primary_state] + [
            float(self.x[self.IDX_B_AX]),
            float(self.x[self.IDX_B_AY]),
            float(self.x[self.IDX_B_GZ]),
        ]
        self.state_pub.publish(msg)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.origin_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(primary_state[0])
        odom.pose.pose.position.y = float(primary_state[1])
        odom.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(primary_state[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(primary_state[3])
        odom.twist.twist.linear.y = float(primary_state[4])
        odom.twist.twist.angular.z = float(primary_state[5])

        odom.pose.covariance[0] = float(self.p[self.IDX_X, self.IDX_X])
        odom.pose.covariance[7] = float(self.p[self.IDX_Y, self.IDX_Y])
        odom.pose.covariance[35] = float(
            self.p[self.IDX_PSI, self.IDX_PSI])
        odom.twist.covariance[0] = float(
            self.p[self.IDX_X_DOT, self.IDX_X_DOT])
        odom.twist.covariance[7] = float(
            self.p[self.IDX_Y_DOT, self.IDX_Y_DOT])
        odom.twist.covariance[35] = float(
            self.p[self.IDX_B_GZ, self.IDX_B_GZ] +
            self.gyro_noise_variance)
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header = odom.header
            tf_msg.child_frame_id = self.base_frame
            tf_msg.transform.translation.x = odom.pose.pose.position.x
            tf_msg.transform.translation.y = odom.pose.pose.position.y
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

    def _set_initial_yaw(self, yaw: float):
        if self.initial_yaw is None:
            self.initial_yaw = yaw

    def _collect_origin_sample(self, msg: NavSatFix, stamp: float):
        if self.origin_start_time is None:
            self.origin_start_time = stamp
            self.get_logger().info(
                'Collecting GPS fixes to initialize averaged local ENU '
                f'origin for {self.origin_init_duration_sec:.1f}s.')

        self.origin_samples.append((
            stamp, msg.latitude, msg.longitude, msg.altitude))

        elapsed = stamp - self.origin_start_time
        sample_count = len(self.origin_samples)
        enough_time = elapsed >= self.origin_init_duration_sec
        enough_samples = sample_count >= max(1, self.origin_min_samples)
        if not (enough_time and enough_samples):
            if stamp - self.last_origin_wait_log_time >= 2.0:
                self.last_origin_wait_log_time = stamp
                self.get_logger().info(
                    'Waiting for GPS origin initialization: '
                    f'samples={sample_count}, elapsed={elapsed:.2f}s.')
            return

        latitudes = np.array([sample[1] for sample in self.origin_samples])
        longitudes = np.array([sample[2] for sample in self.origin_samples])
        altitudes = np.array([sample[3] for sample in self.origin_samples])

        reference = LocalCartesian(
            self.origin_samples[0][1],
            self.origin_samples[0][2],
            self.origin_samples[0][3])
        offsets = np.array([
            reference.forward(sample[1], sample[2], sample[3])[:2]
            for sample in self.origin_samples
        ])
        std_xy = float(np.sqrt(np.mean(np.var(offsets, axis=0))))
        if std_xy > self.origin_max_std_m:
            if stamp - self.last_origin_wait_log_time >= 2.0:
                self.last_origin_wait_log_time = stamp
                self.get_logger().warn(
                    'GPS origin samples are still moving/noisy: '
                    f'std_xy={std_xy:.3f} m > '
                    f'{self.origin_max_std_m:.3f} m; continuing to collect.')
            return

        latitude = float(np.mean(latitudes))
        longitude = float(np.mean(longitudes))
        altitude = float(np.mean(altitudes))
        self.local_cartesian = LocalCartesian(latitude, longitude, altitude)
        self.origin_fix = msg
        self.x[0] = 0.0
        self.x[1] = 0.0
        self.last_gps_time = stamp
        self.last_base_xy_meas = None
        self.get_logger().info(
            'Initialized averaged local ENU origin from GPS fixes: '
            f'samples={sample_count}, elapsed={elapsed:.2f}s, '
            f'std_xy={std_xy:.3f} m, '
            f'lat={latitude:.9f}, lon={longitude:.9f}, alt={altitude:.3f}')
        self.publish_state(msg.header.stamp)

    def _base_position_measurement(self, gps_delta: np.ndarray,
                                   yaw: float) -> np.ndarray:
        sensor_offset = self._rotate_body_vector(
            yaw, self.gps_body_x, self.gps_body_y)
        initial_sensor_offset = self._rotate_body_vector(
            self.initial_yaw, self.gps_body_x, self.gps_body_y)
        return gps_delta - sensor_offset + initial_sensor_offset

    @staticmethod
    def _rotate_body_vector(yaw: float, x_body: float,
                            y_body: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([
            c * x_body - s * y_body,
            s * x_body + c * y_body,
        ])

    def _primary_state(self) -> np.ndarray:
        return np.array([
            self.x[self.IDX_X],
            self.x[self.IDX_Y],
            self.x[self.IDX_PSI],
            self.x[self.IDX_X_DOT],
            self.x[self.IDX_Y_DOT],
            self.last_yaw_rate,
        ])

    def _valid_fix(self, msg: NavSatFix) -> bool:
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return False
        return all(math.isfinite(v) for v in (
            msg.latitude, msg.longitude, msg.altitude))

    def _gps_position_variance(self, msg: NavSatFix) -> float:
        cov = msg.position_covariance
        if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            variances = [cov[0], cov[4]]
            if all(v > 0.0 and math.isfinite(v) for v in variances):
                return max(variances)
        return self.gps_position_variance

    def _imu_has_orientation(self, msg: Imu) -> bool:
        q = msg.orientation
        norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if msg.orientation_covariance[0] < 0.0:
            return False
        return norm > 0.5 and math.isfinite(norm)

    def _imu_yaw_variance(self, msg: Imu) -> float:
        cov = msg.orientation_covariance
        if cov[8] > 0.0 and math.isfinite(cov[8]):
            return cov[8]
        return self.imu_yaw_variance


def main(args=None):
    rclpy.init(args=args)
    node = GpsImuEkfNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
