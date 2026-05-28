import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robotx_safe_docking_control')
    default_params = os.path.join(
        pkg_share, 'config', 'cascaded_pid_controller.yaml')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Parameter file for the cascaded PID controller.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock.'),
        Node(
            package='robotx_safe_docking_control',
            executable='cascaded_pid_controller',
            name='cascaded_pid_controller',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}]),
    ])
