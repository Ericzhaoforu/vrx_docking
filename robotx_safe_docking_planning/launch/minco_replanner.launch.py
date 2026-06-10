import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robotx_safe_docking_planning')
    default_params = os.path.join(
        pkg_share, 'config', 'minco_replanner.yaml')

    params_file = LaunchConfiguration('params_file')
    execution_status_topic = LaunchConfiguration('execution_status_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Parameter file for the C++ MINCO replanner.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock.'),
        DeclareLaunchArgument(
            'execution_status_topic',
            default_value='/safe_docking/minco_execution_status',
            description='Controller ACK / execution feedback topic.'),
        Node(
            package='robotx_safe_docking_planning',
            executable='minco_replanner_node',
            name='minco_replanner_node',
            output='screen',
            parameters=[
                params_file,
                {
                    'use_sim_time': use_sim_time,
                    'execution_status_topic': execution_status_topic,
                },
            ]),
    ])
