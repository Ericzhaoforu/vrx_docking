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
    minco_trajectory_topic = LaunchConfiguration('minco_trajectory_topic')
    minco_execution_status_topic = LaunchConfiguration(
        'minco_execution_status_topic')
    minco_replan_enabled = LaunchConfiguration('minco_replan_enabled')
    minco_replan_check_rate_hz = LaunchConfiguration(
        'minco_replan_check_rate_hz')
    minco_handover_duration = LaunchConfiguration('minco_handover_duration')
    minco_tau_v_bar = LaunchConfiguration('minco_tau_v_bar')
    minco_accept_tau_v_bound = LaunchConfiguration(
        'minco_accept_tau_v_bound')
    frontend_tau_v_bar = LaunchConfiguration('frontend_tau_v_bar')
    frontend_control_discretization = LaunchConfiguration(
        'frontend_control_discretization')
    frontend_sample_thrust_limit = LaunchConfiguration(
        'frontend_sample_thrust_limit')
    frontend_minco_piece_count_min = LaunchConfiguration(
        'frontend_minco_piece_count_min')
    frontend_minco_piece_count_max = LaunchConfiguration(
        'frontend_minco_piece_count_max')
    frontend_max_local_goal_distance = LaunchConfiguration(
        'frontend_max_local_goal_distance')
    frontend_obstacles = LaunchConfiguration('frontend_obstacles')
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
                'arc, stop, figure8, spiral, yaw. MINCO supports local_offset '
                'and goal_lattice for USV-aware local front-end replanning.')),
        DeclareLaunchArgument(
            'minco_trajectory_topic',
            default_value='/safe_docking/minco_trajectory',
            description='External MINCO coefficient trajectory topic.'),
        DeclareLaunchArgument(
            'minco_execution_status_topic',
            default_value='/safe_docking/minco_execution_status',
            description='Controller ACK / execution feedback topic.'),
        DeclareLaunchArgument(
            'minco_replan_enabled',
            default_value='False',
            description='Enable online MINCO periodic replanning.'),
        DeclareLaunchArgument(
            'minco_replan_check_rate_hz',
            default_value='1.0',
            description='Rate for evaluating online MINCO replan eligibility.'),
        DeclareLaunchArgument(
            'minco_handover_duration',
            default_value='0.0',
            description='Deprecated old/new reference blend duration.'),
        DeclareLaunchArgument(
            'minco_tau_v_bar',
            default_value='12.0',
            description='MINCO lateral generalized-force diagnostic bound.'),
        DeclareLaunchArgument(
            'minco_accept_tau_v_bound',
            default_value='12.0',
            description='Accepted-reference lateral force diagnostic bound.'),
        DeclareLaunchArgument(
            'frontend_tau_v_bar',
            default_value='12.0',
            description='Lattice primitive lateral force feasibility bound.'),
        DeclareLaunchArgument(
            'frontend_control_discretization',
            default_value='1',
            description='Input-grid half resolution r for lattice primitives.'),
        DeclareLaunchArgument(
            'frontend_sample_thrust_limit',
            default_value='0.0',
            description=(
                'Symmetric per-thruster sampling limit for lattice primitives. '
                'Use 0 to sample the physical actuator bounds.')),
        DeclareLaunchArgument(
            'frontend_minco_piece_count_min',
            default_value='3',
            description='Minimum MINCO pieces for lattice-selected paths.'),
        DeclareLaunchArgument(
            'frontend_minco_piece_count_max',
            default_value='8',
            description='Maximum MINCO pieces for lattice-selected paths.'),
        DeclareLaunchArgument(
            'frontend_max_local_goal_distance',
            default_value='5.0',
            description='Maximum local kinodynamic front-end planning horizon.'),
        DeclareLaunchArgument(
            'frontend_obstacles',
            default_value='',
            description=(
                'Simulated local circular obstacles for the lattice front end, '
                'formatted as "x,y,r;x,y,r".')),
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
                    'minco_trajectory_topic': minco_trajectory_topic,
                    'minco_execution_status_topic':
                    minco_execution_status_topic,
                    'minco_replan_enabled': minco_replan_enabled,
                    'minco_replan_check_rate_hz': minco_replan_check_rate_hz,
                    'minco_handover_duration': minco_handover_duration,
                    'minco_tau_v_bar': minco_tau_v_bar,
                    'minco_accept_tau_v_bound': minco_accept_tau_v_bound,
                    'frontend_tau_v_bar': frontend_tau_v_bar,
                    'frontend_control_discretization':
                    frontend_control_discretization,
                    'frontend_sample_thrust_limit':
                    frontend_sample_thrust_limit,
                    'frontend_minco_piece_count_min':
                    frontend_minco_piece_count_min,
                    'frontend_minco_piece_count_max':
                    frontend_minco_piece_count_max,
                    'frontend_max_local_goal_distance':
                    frontend_max_local_goal_distance,
                    'frontend_obstacles': frontend_obstacles,
                    'use_sim_time': use_sim_time,
                    'acados_source_dir': acados_source_dir,
                },
            ]),
    ])
