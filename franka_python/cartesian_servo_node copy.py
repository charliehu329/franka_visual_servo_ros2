#!/usr/bin/env python3
"""
cartesian_servo_node.py

功能：
    测试“给定末端执行器速度 V_e，转换为关节速度 q_dot，并发送给 Franka”这一控制链路。
    本节点会读取关节状态 topic 中的当前关节角 q，调用 robot_kinematics.py 计算 J(q)，
    再调用 velocity_mapper.py 计算 q_dot，最后发布到速度控制 topic。

说明：
    本节点不在代码中设置默认参数。
    所有参数默认值请放在 YAML 文件中，例如：
        config/cartesian_servo.yaml
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

from franka_python.robot_kinematics import FrankaKinematics
from franka_python.velocity_mapper import cartesian_velocity_to_joint_velocity


class CartesianServoNode(Node):
    def __init__(self):
        super().__init__("cartesian_servo_node")

        # ===== 只声明参数名，具体默认值放在 YAML 文件里 =====

        # 运动学模型相关参数
        self.declare_parameter("urdf_path")
        self.declare_parameter("end_effector_frame")
        self.declare_parameter(
                "T_end_effector_camera"
            )

        # 控制参数
        self.declare_parameter("damping")
        self.declare_parameter("publish_rate_hz")
        self.declare_parameter("duration_sec")
        self.declare_parameter("dry_run")

        # ROS topic 和安全限幅参数
        self.declare_parameter("joint_state_topic")
        self.declare_parameter("command_topic")
        self.declare_parameter("max_joint_velocity")

        # ===== 读取必要参数 =====
        self.urdf_path = self.get_required_parameter("urdf_path")
        self.end_effector_frame = self.get_required_parameter("end_effector_frame")
        
        T_end_effector_camera = np.array(
            self.get_required_parameter(
                "T_end_effector_camera"
            ),
            dtype=float
        )

        if T_end_effector_camera.size != 16:
            raise ValueError(
                "T_end_effector_camera must contain 16 elements."
            )

        self.T_end_effector_camera = (
            T_end_effector_camera.reshape(4, 4)
        )



        self.damping = float(self.get_required_parameter("damping"))
        self.publish_rate_hz = float(self.get_required_parameter("publish_rate_hz"))
        self.duration_sec = float(self.get_required_parameter("duration_sec"))
        self.dry_run = bool(self.get_required_parameter("dry_run"))

        self.joint_state_topic = self.get_required_parameter("joint_state_topic")
        self.command_topic = self.get_required_parameter("command_topic")
        self.max_joint_velocity = float(
            self.get_required_parameter("max_joint_velocity")
        )

        # 当前末端速度，由视觉伺服节点通过topic更新
        # V_e = [vx, vy, vz, wx, wy, wz]
        self.V_e = np.zeros(
            6,
            dtype=float
        )

        # ===== 初始化运动学模型 =====
        self.kinematics = FrankaKinematics(
            urdf_path=self.urdf_path,
            end_effector_frame=self.end_effector_frame
        )

        self.current_q = None

        # ===== 订阅当前关节角 =====
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10
        )

        # ===== 订阅视觉伺服输出的末端速度 =====
        self.visual_velocity_sub = self.create_subscription(
            Float64MultiArray,
            "/visual_servo_velocity",
            self.visual_velocity_callback,
            10
        )

        # ===== 发布关节速度指令 =====
        self.publisher = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10
        )

        self.start_time = self.get_clock().now()

        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("Cartesian servo node started.")
        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(f"End-effector frame: {self.end_effector_frame}")
        self.get_logger().info(f"Joint state topic: {self.joint_state_topic}")
        self.get_logger().info(f"Command topic: {self.command_topic}")
        self.get_logger().info("Waiting for visual servo velocity.")
        self.get_logger().info(f"Damping: {self.damping}")
        self.get_logger().info(f"Publish rate: {self.publish_rate_hz} Hz")
        self.get_logger().info(f"Duration: {self.duration_sec} s")
        self.get_logger().info(f"Max joint velocity: {self.max_joint_velocity}")
        self.get_logger().info(f"Dry run: {self.dry_run}")

    def get_required_parameter(self, name):
        """
        读取必须由 YAML 或命令行提供的参数。
        如果参数没有被设置，则直接报错，避免使用不明确的默认值。
        """
        param = self.get_parameter(name)

        if param.type_ == rclpy.Parameter.Type.NOT_SET:
            raise ValueError(
                f"Required parameter '{name}' is not set. "
                f"Please provide it in YAML or command line."
            )

        return param.value

    def joint_state_callback(self, msg):
        joint_map = dict(zip(msg.name, msg.position))

        try:
            self.current_q = np.array([
                joint_map["fr3_joint1"],
                joint_map["fr3_joint2"],
                joint_map["fr3_joint3"],
                joint_map["fr3_joint4"],
                joint_map["fr3_joint5"],
                joint_map["fr3_joint6"],
                joint_map["fr3_joint7"],
            ], dtype=float)
        except KeyError as e:
            self.get_logger().warn(f"Missing joint in {self.joint_state_topic}: {e}")
            self.current_q = None


    def visual_velocity_callback(self, msg):
        """
        接收视觉伺服输出的相机速度V_c

        输入：
            msg.data:
                V_c = [vx, vy, vz, wx, wy, wz]
                速度在相机坐标系下表达。
        """

        V_c = np.array(
            msg.data,
            dtype=float
        )

        if V_c.shape != (6,):
            self.get_logger().warn(
                "Visual velocity must have 6 elements."
            )
            return

        self.V_e = (
            self.camera_velocity_to_end_effector_velocity(
                V_c
            )
        )



    @staticmethod
    def skew(vector):
        """
        将三维向量转换为反对称矩阵。
        """

        x, y, z = vector

        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0]
            ],
            dtype=float
        )


    def camera_velocity_to_end_effector_velocity(
        self,
        V_c
    ):
        """
        将相机坐标系下的速度V_c转换为
        末端执行器坐标系下的速度V_e。

        输入：
            V_c:
                [vx, vy, vz, wx, wy, wz]

        输出：
            V_e:
                [vx, vy, vz, wx, wy, wz]
        """

        R_e_c = self.T_end_effector_camera[
            :3,
            :3
        ]

        t_e_c = self.T_end_effector_camera[
            :3,
            3
        ]

        adjoint_e_c = np.zeros(
            (6, 6),
            dtype=float
        )

        # 当前速度排列方式：
        # [线速度, 角速度]
        adjoint_e_c[:3, :3] = R_e_c

        adjoint_e_c[:3, 3:] = (
            self.skew(t_e_c) @ R_e_c
        )

        adjoint_e_c[3:, 3:] = R_e_c

        V_e = adjoint_e_c @ V_c

        return V_e

    def timer_callback(self):
        if self.current_q is None:
            self.get_logger().warn(f"Waiting for {self.joint_state_topic}...")
            return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if self.duration_sec > 0.0 and elapsed >= self.duration_sec:
            self.publish_zero_velocity()
            self.get_logger().info("Duration reached. Stop.")
            rclpy.shutdown()
            return

        # 根据当前关节角 q 计算雅可比矩阵 J(q)
        J = self.kinematics.compute_jacobian(self.current_q)

        # 根据末端速度 V_e 和雅可比矩阵 J(q) 计算关节速度 q_dot
        q_dot = cartesian_velocity_to_joint_velocity(
            V_e=self.V_e,
            J=J,
            damping=self.damping
        )

        # 关节速度限幅，防止速度过大
        q_dot = self.limit_joint_velocity(
            q_dot,
            max_abs=self.max_joint_velocity
        )

        self.get_logger().info(
            f"q: {np.round(self.current_q, 4).tolist()} | "
            f"q_dot: {np.round(q_dot, 5).tolist()}"
        )

        # dry_run=True 时只打印，不真正发给机器人
        if not self.dry_run:
            msg = Float64MultiArray()
            msg.data = q_dot.tolist()
            self.publisher.publish(msg)

    def limit_joint_velocity(self, q_dot, max_abs):
        return np.clip(q_dot, -max_abs, max_abs)

    def publish_zero_velocity(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 7
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianServoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Send zero velocity.")
        node.publish_zero_velocity()
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()


"""
使用示例：

1. 重新编译并加载环境：

    cd ~/franka_ros2_ws
    colcon build --packages-select franka_python
    source install/setup.bash

2. 打开一个终端，运行底层控制节点：

    ros2 launch franka_velocity_ctrl fr3_velocity.launch.py \
    robot_ip:=172.16.0.2 \
    mode:=topic
概括：
    创建ROS 2眼在手上手眼标定节点。
    节点打开USB摄像头，检测固定棋盘格，并自动从TF读取机器人末端位姿。
    用户每次按S保存一组同步数据；达到设定数量后，自动计算
    T_end_effector_camera，并保存为ROS 2 YAML文件。

3. 使用 YAML 参数文件运行：

    ros2 run franka_python cartesian_servo_node --ros-args \
      --params-file ~/franka_ros2_ws/src/franka_python/config/cartesian_servo.yaml


4. 使用 YAML 文件，并临时覆盖某些参数：

    ros2 run franka_python cartesian_servo_node --ros-args \
      --params-file ~/franka_ros2_ws/src/franka_python/config/cartesian_servo.yaml \
      -p dry_run:=false \
      -p target_cartesian_velocity:="[0.0, 0.0, 0.005, 0.0, 0.0, 0.0]" \
      -p duration_sec:=3.0


常用可传入参数：

    urdf_path:
        FR3 的 URDF 文件路径，例如 "/tmp/fr3.urdf"

    end_effector_frame:
        末端坐标系，例如 "fr3_hand_tcp"

    target_cartesian_velocity:
        末端速度 V_e = [vx, vy, vz, wx, wy, wz]
        vx, vy, vz 单位 m/s
        wx, wy, wz 单位 rad/s

    damping:
        阻尼最小二乘伪逆的阻尼因子，例如 0.05

    publish_rate_hz:
        关节速度发布频率，例如 50.0

    duration_sec:
        控制持续时间，单位秒，例如 1.0

    dry_run:
        true  表示只打印，不发送给机器人
        false 表示真实发送速度指令

    joint_state_topic:
        关节状态 topic，例如 "/franka/joint_states"

    command_topic:
        关节速度命令 topic，例如 "/velocity_command_node/target_velocities"

    max_joint_velocity:
        单个关节速度限幅，单位 rad/s，例如 0.05


注意：
    现在代码里没有默认参数。
    所以运行 cartesian_servo_node 时，必须使用 YAML 文件或者命令行传入所有参数。
    推荐用 YAML 作为默认配置，再用命令行 -p 临时覆盖某几个参数。
"""