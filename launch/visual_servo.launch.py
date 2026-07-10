#!/usr/bin/env python3
"""
当前版本为能追踪快速物体完整版本V1.0
visual_servo.launch.py

功能：
    一次启动 Franka 单目视觉伺服系统。

启动内容：
    1. Franka 关节速度底层控制器。
    2. cartesian_servo_node：
       将6维末端速度转换为7维关节速度。
    3. visual_servo_law：
       读取USB相机、检测绿色小球并计算视觉伺服速度。

输入：
    robot_ip:
        Franka机器人IP地址。

    mode:
        底层速度控制器的指令模式。

    dry_run:
        是否只计算而不向机器人发布关节速度。

    visual_params_file:
        视觉伺服参数文件路径。

输出：
    /visual_servo_velocity
        6维末端速度。

    /velocity_command_node/target_velocities
        7维关节速度。
"""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import (
    ParameterValue,
)


def generate_launch_description():
    """
    创建视觉伺服系统的LaunchDescription。
    """

    # -----------------------------
    # Launch参数
    # -----------------------------

    robot_ip = LaunchConfiguration("robot_ip")
    mode = LaunchConfiguration("mode")
    dry_run = LaunchConfiguration("dry_run")
    visual_params_file = LaunchConfiguration(
        "visual_params_file"
    )

    # -----------------------------
    # 默认参数文件
    # -----------------------------

    franka_python_share = get_package_share_directory(
        "franka_python"
    )

    default_visual_params_file = os.path.join(
        franka_python_share,
        "config",
        "cartesian_servo.yaml",
    )

    # -----------------------------
    # Franka底层控制Launch
    # -----------------------------

    franka_velocity_ctrl_share = (
        get_package_share_directory(
            "franka_velocity_ctrl"
        )
    )

    franka_velocity_launch_file = os.path.join(
        franka_velocity_ctrl_share,
        "launch",
        "fr3_velocity.launch.py",
    )

    franka_velocity_controller = (
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                franka_velocity_launch_file
            ),
            launch_arguments={
                "robot_ip": robot_ip,
                "mode": mode,
            }.items(),
        )
    )

    # -----------------------------
    # 笛卡尔速度控制节点
    # -----------------------------

    cartesian_servo_node = Node(
        package="franka_python",
        executable="cartesian_servo_node",
        name="cartesian_servo_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            visual_params_file,
            {
                "dry_run": ParameterValue(
                    dry_run,
                    value_type=bool,
                )
            },
        ],
    )

    # -----------------------------
    # 视觉伺服律节点
    # -----------------------------

    visual_servo_law_node = Node(
        package="franka_python",
        executable="visual_servo_law",
        name="visual_servo_law",
        output="screen",
        emulate_tty=True,
        parameters=[
            visual_params_file,
        ],
    )

    # -----------------------------
    # 启动顺序
    # -----------------------------
    #
    # 0秒：启动Franka底层控制器
    # 5秒：启动笛卡尔速度控制节点
    # 7秒：启动视觉伺服节点
    #
    # 这样可以尽量避免视觉节点已经发布速度，
    # 但底层控制器还没有准备好的情况。
    # -----------------------------

    delayed_cartesian_servo_node = TimerAction(
        period=5.0,
        actions=[
            cartesian_servo_node,
        ],
    )

    delayed_visual_servo_law_node = TimerAction(
        period=7.0,
        actions=[
            visual_servo_law_node,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value="172.16.0.2",
                description="Franka robot IP address.",
            ),
            DeclareLaunchArgument(
                "mode",
                default_value="topic",
                description=(
                    "Velocity controller command mode."
                ),
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="true",
                description=(
                    "Calculate velocities without "
                    "sending joint commands."
                ),
            ),
            DeclareLaunchArgument(
                "visual_params_file",
                default_value=(
                    default_visual_params_file
                ),
                description=(
                    "Visual servo parameter YAML file."
                ),
            ),
            franka_velocity_controller,
            delayed_cartesian_servo_node,
            delayed_visual_servo_law_node,
        ]
    )