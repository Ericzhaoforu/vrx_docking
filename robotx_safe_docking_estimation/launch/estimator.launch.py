import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robotx_safe_docking_estimation')
    default_params = os.path.join(pkg_share, 'config', 'gps_imu_ekf.yaml')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    verify = LaunchConfiguration('verify')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Parameter file for the GPS/IMU EKF.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock.'),
        DeclareLaunchArgument(
            'verify',
            default_value='False',
            description='Launch the ground-truth verifier for development.'),
        Node(
            package='robotx_safe_docking_estimation',
            executable='gps_imu_ekf_node',
            name='gps_imu_ekf_node',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}]),
        Node(
            package='robotx_safe_docking_estimation',
            executable='state_estimator_verifier',
            name='state_estimator_verifier',
            output='screen',
            condition=IfCondition(verify),
            parameters=[params_file, {'use_sim_time': use_sim_time}]),
    ])
