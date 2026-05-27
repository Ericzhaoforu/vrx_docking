# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vrx_gz_share_dir = get_package_share_directory('vrx_gz')
    vrx_gz_launch_dir = os.path.join(
        vrx_gz_share_dir, 'launch')
    default_config_file = os.path.join(
        vrx_gz_share_dir, 'config', 'safe_docking_wamv.yaml')
    default_rviz_config = os.path.join(
        vrx_gz_share_dir, 'config', 'safe_docking.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='safe_docking_task',
            description='Safe docking world to launch.'),
        DeclareLaunchArgument(
            'sim_mode',
            default_value='full',
            description='Simulation mode: "full", "sim", or "bridge".'),
        DeclareLaunchArgument(
            'bridge_competition_topics',
            default_value='True',
            description='True to bridge common VRX task topics.'),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='YAML configuration file to spawn models.'),
        DeclareLaunchArgument(
            'robot',
            default_value='',
            description='Optional robot name to spawn from config_file.'),
        DeclareLaunchArgument(
            'headless',
            default_value='False',
            description='True to run simulation headless.'),
        DeclareLaunchArgument(
            'urdf',
            default_value='',
            description='Optional URDF file of the WAM-V model.'),
        DeclareLaunchArgument(
            'paused',
            default_value='False',
            description='True to start the simulation paused.'),
        DeclareLaunchArgument(
            'competition_mode',
            default_value='False',
            description='True to disable debug competition topics.'),
        DeclareLaunchArgument(
            'extra_gz_args',
            default_value='',
            description='Additional arguments to pass to gz sim.'),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='False',
            description='True to launch RViz with the safe docking config.'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz_config,
            description='RViz configuration file to load.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(vrx_gz_launch_dir, 'competition.launch.py')),
            launch_arguments={
                'world': LaunchConfiguration('world'),
                'sim_mode': LaunchConfiguration('sim_mode'),
                'bridge_competition_topics':
                    LaunchConfiguration('bridge_competition_topics'),
                'config_file': LaunchConfiguration('config_file'),
                'robot': LaunchConfiguration('robot'),
                'headless': LaunchConfiguration('headless'),
                'urdf': LaunchConfiguration('urdf'),
                'paused': LaunchConfiguration('paused'),
                'competition_mode': LaunchConfiguration('competition_mode'),
                'extra_gz_args': LaunchConfiguration('extra_gz_args'),
            }.items()),
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='safe_docking_rviz',
                    arguments=['-d', LaunchConfiguration('rviz_config')],
                    parameters=[{'use_sim_time': True}],
                    condition=IfCondition(LaunchConfiguration('launch_rviz')),
                    output='screen'),
            ]),
    ])
