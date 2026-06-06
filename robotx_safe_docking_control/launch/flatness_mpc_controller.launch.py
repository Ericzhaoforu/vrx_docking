import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robotx_safe_docking_control')
    default_params = os.path.join(
        pkg_share, 'config', 'flatness_mpc_controller.yaml')

    params_file = LaunchConfiguration('params_file')
    reference_source = LaunchConfiguration('reference_source')
    reference_name = LaunchConfiguration('reference_name')
    use_sim_time = LaunchConfiguration('use_sim_time')
    acados_source_dir = LaunchConfiguration('acados_source_dir')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Parameter file for the flatness-MPC tracker.'),
        DeclareLaunchArgument(
            'reference_source',
            default_value='synthetic',
            description='Reference source: synthetic or minco.'),
        DeclareLaunchArgument(
            'reference_name',
            default_value='arc',
            description=(
                'Reference scenario/name. Synthetic supports hold, straight, '
                'arc, stop, figure8, spiral, yaw. MINCO validation should use '
                'local_offset for the short local terminal-offset maneuver.')),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock.'),
        DeclareLaunchArgument(
            'acados_source_dir',
            default_value='/home/zjy/acados',
            description='Local acados source/install directory.'),
        SetEnvironmentVariable(
            name='ACADOS_SOURCE_DIR',
            value=acados_source_dir),
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=[
                acados_source_dir,
                '/lib:',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ]),
        Node(
            package='robotx_safe_docking_control',
            executable='flatness_mpc_controller',
            name='flatness_mpc_controller',
            output='screen',
            parameters=[
                params_file,
                {
                    'reference_source': reference_source,
                    'reference_name': reference_name,
                    'use_sim_time': use_sim_time,
                    'acados_source_dir': acados_source_dir,
                },
            ]),
    ])
