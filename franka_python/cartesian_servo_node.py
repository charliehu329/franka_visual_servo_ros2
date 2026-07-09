#!/usr/bin/env python3
"""
cartesian_servo_node.py

概括：
    创建 CartesianServoNode 节点。
    接收视觉伺服节点发布的相机速度 V_c，将其转换为末端速度 V_e，
    再根据当前关节角和雅可比矩阵计算关节速度 q_dot，
    最后发送给 Franka 速度控制器。

功能：
    1. 订阅当前关节状态。
    2. 订阅视觉伺服速度 V_c。
    3. 将相机速度转换为末端执行器速度。
    4. 根据雅可比矩阵将末端速度转换为关节速度。
    5. 对关节速度进行有限值检查和限幅。
    6. 发布关节速度给 Franka。
    7. 视觉速度超时后强制发布零速度。
    8. 关节状态超时后强制发布零速度。
    9. 收到错误、无效或非有限速度时强制发布零速度。
    10. 节点退出前发布零速度。

接口：
    joint_state_callback(msg)
    visual_velocity_callback(msg)
    timer_callback()
    publish_zero_velocity()

输入：
    JointState:
        当前七个关节角。

    /visual_servo_velocity:
        V_c = [vx, vy, vz, wx, wy, wz]
        在相机坐标系下表达。

输出：
    command_topic:
        q_dot = [dq1, dq2, dq3, dq4, dq5, dq6, dq7]

安全机制：
    1. visual_velocity_timeout_sec：
       超过该时间未收到新的视觉速度，立即发送零关节速度。

    2. joint_state_timeout_sec：
       超过该时间未收到新的关节状态，立即发送零关节速度。

    3. visual_servo_law 没有识别到目标时应持续发布零相机速度。

说明：
    本节点不在代码中设置运动控制参数默认值。
    所有参数必须从 YAML 文件或命令行提供。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from franka_python.robot_kinematics import FrankaKinematics
from franka_python.safety import (
    check_finite_vector,
    limit_joint_velocity,
)
from franka_python.velocity_mapper import (
    cartesian_velocity_to_joint_velocity,
)


class CartesianServoNode(Node):
    """
    Franka 笛卡尔速度到关节速度转换节点。
    """

    def __init__(self):
        super().__init__("cartesian_servo_node")

        # =====================================================
        # 声明参数
        # =====================================================

        # 运动学模型
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

        # Topic 和速度限制
        self.declare_parameter("joint_state_topic")
        self.declare_parameter("command_topic")
        self.declare_parameter("max_joint_velocity")

        # 安全超时参数
        self.declare_parameter(
            "visual_velocity_timeout_sec"
        )
        self.declare_parameter(
            "joint_state_timeout_sec"
        )

        # =====================================================
        # 读取参数
        # =====================================================

        self.urdf_path = (
            self.get_required_parameter(
                "urdf_path"
            )
        )

        self.end_effector_frame = (
            self.get_required_parameter(
                "end_effector_frame"
            )
        )

        transform_data = np.asarray(
            self.get_required_parameter(
                "T_end_effector_camera"
            ),
            dtype=float
        )

        if transform_data.size != 16:
            raise ValueError(
                "T_end_effector_camera must "
                "contain 16 elements."
            )

        if not np.all(
            np.isfinite(transform_data)
        ):
            raise ValueError(
                "T_end_effector_camera contains "
                "nan or inf."
            )

        self.T_end_effector_camera = (
            transform_data.reshape(4, 4)
        )

        self.damping = float(
            self.get_required_parameter(
                "damping"
            )
        )

        self.publish_rate_hz = float(
            self.get_required_parameter(
                "publish_rate_hz"
            )
        )

        self.duration_sec = float(
            self.get_required_parameter(
                "duration_sec"
            )
        )

        self.dry_run = bool(
            self.get_required_parameter(
                "dry_run"
            )
        )

        self.joint_state_topic = str(
            self.get_required_parameter(
                "joint_state_topic"
            )
        )

        self.command_topic = str(
            self.get_required_parameter(
                "command_topic"
            )
        )

        self.max_joint_velocity = float(
            self.get_required_parameter(
                "max_joint_velocity"
            )
        )

        self.visual_velocity_timeout_sec = float(
            self.get_required_parameter(
                "visual_velocity_timeout_sec"
            )
        )

        self.joint_state_timeout_sec = float(
            self.get_required_parameter(
                "joint_state_timeout_sec"
            )
        )

        self.validate_parameters()

        # =====================================================
        # 当前控制状态
        # =====================================================

        # 当前关节角
        self.current_q = None

        # 当前末端速度
        self.V_e = np.zeros(
            6,
            dtype=float
        )

        # 最近一次收到消息的时间
        self.last_visual_velocity_time = None
        self.last_joint_state_time = None

        # 当前安全停止原因
        # 用来避免在每个定时器周期重复打印相同警告
        self.safe_stop_reason = None

        # =====================================================
        # 初始化运动学模型
        # =====================================================

        self.kinematics = FrankaKinematics(
            urdf_path=self.urdf_path,
            end_effector_frame=(
                self.end_effector_frame
            )
        )

        # =====================================================
        # ROS 通信
        # =====================================================

        self.joint_state_sub = (
            self.create_subscription(
                JointState,
                self.joint_state_topic,
                self.joint_state_callback,
                10
            )
        )

        self.visual_velocity_sub = (
            self.create_subscription(
                Float64MultiArray,
                "/visual_servo_velocity",
                self.visual_velocity_callback,
                10
            )
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10
        )

        self.start_time = self.get_clock().now()

        timer_period = (
            1.0 /
            self.publish_rate_hz
        )

        self.timer = self.create_timer(
            timer_period,
            self.timer_callback
        )

        # =====================================================
        # 启动信息
        # =====================================================

        self.get_logger().info(
            "Cartesian servo node started."
        )

        self.get_logger().info(
            f"URDF: {self.urdf_path}"
        )

        self.get_logger().info(
            "End-effector frame: "
            f"{self.end_effector_frame}"
        )

        self.get_logger().info(
            "Joint state topic: "
            f"{self.joint_state_topic}"
        )

        self.get_logger().info(
            "Visual velocity topic: "
            "/visual_servo_velocity"
        )

        self.get_logger().info(
            "Command topic: "
            f"{self.command_topic}"
        )

        self.get_logger().info(
            f"Damping: {self.damping}"
        )

        self.get_logger().info(
            "Publish rate: "
            f"{self.publish_rate_hz} Hz"
        )

        self.get_logger().info(
            f"Duration: {self.duration_sec} s"
        )

        self.get_logger().info(
            "Max joint velocity: "
            f"{self.max_joint_velocity} rad/s"
        )

        self.get_logger().info(
            "Visual velocity timeout: "
            f"{self.visual_velocity_timeout_sec} s"
        )

        self.get_logger().info(
            "Joint state timeout: "
            f"{self.joint_state_timeout_sec} s"
        )

        self.get_logger().info(
            f"Dry run: {self.dry_run}"
        )

        self.get_logger().info(
            "Waiting for joint state and "
            "visual servo velocity."
        )

    def validate_parameters(self):
        """
        检查参数是否合法。
        """

        if self.damping < 0.0:
            raise ValueError(
                "damping must be non-negative."
            )

        if self.publish_rate_hz <= 0.0:
            raise ValueError(
                "publish_rate_hz must be positive."
            )

        if self.duration_sec < 0.0:
            raise ValueError(
                "duration_sec must be non-negative."
            )

        if self.max_joint_velocity <= 0.0:
            raise ValueError(
                "max_joint_velocity must be positive."
            )

        if (
            self.visual_velocity_timeout_sec
            <= 0.0
        ):
            raise ValueError(
                "visual_velocity_timeout_sec "
                "must be positive."
            )

        if self.joint_state_timeout_sec <= 0.0:
            raise ValueError(
                "joint_state_timeout_sec "
                "must be positive."
            )

    def get_required_parameter(
        self,
        name
    ):
        """
        读取必须由 YAML 或命令行提供的参数。

        如果参数未设置，直接报错。
        """

        parameter = self.get_parameter(name)

        if (
            parameter.type_ ==
            rclpy.Parameter.Type.NOT_SET
        ):
            raise ValueError(
                f"Required parameter '{name}' "
                "is not set. Please provide it "
                "in YAML or command line."
            )

        return parameter.value

    def joint_state_callback(
        self,
        msg
    ):
        """
        接收 Franka 当前关节角。
        """

        joint_map = dict(
            zip(
                msg.name,
                msg.position
            )
        )

        try:
            current_q = np.asarray(
                [
                    joint_map["fr3_joint1"],
                    joint_map["fr3_joint2"],
                    joint_map["fr3_joint3"],
                    joint_map["fr3_joint4"],
                    joint_map["fr3_joint5"],
                    joint_map["fr3_joint6"],
                    joint_map["fr3_joint7"],
                ],
                dtype=float
            )

            current_q = check_finite_vector(
                current_q,
                "current_q"
            )

        except KeyError as error:
            self.current_q = None

            self.enter_safe_stop(
                "Joint state message is missing "
                f"a required joint: {error}"
            )

            return

        except ValueError as error:
            self.current_q = None

            self.enter_safe_stop(
                f"Invalid joint state: {error}"
            )

            return

        self.current_q = current_q

        self.last_joint_state_time = (
            self.get_clock().now()
        )

    def visual_velocity_callback(
        self,
        msg
    ):
        """
        接收视觉伺服节点输出的相机速度 V_c。

        输入：
            msg.data:
                V_c = [vx, vy, vz, wx, wy, wz]

        说明：
            视觉节点没有检测到目标时，
            应发布 [0, 0, 0, 0, 0, 0]。
        """

        V_c = np.asarray(
            msg.data,
            dtype=float
        ).reshape(-1)

        if V_c.shape != (6,):
            self.enter_safe_stop(
                "Visual velocity must contain "
                "exactly 6 elements."
            )
            return

        try:
            V_c = check_finite_vector(
                V_c,
                "V_c"
            )

            V_e = (
                self.camera_velocity_to_end_effector_velocity(
                    V_c
                )
            )

            V_e = check_finite_vector(
                V_e,
                "V_e"
            )

        except ValueError as error:
            self.enter_safe_stop(
                f"Invalid visual velocity: {error}"
            )
            return

        # 只有数据完全合法时才更新当前速度
        self.V_e = V_e

        self.last_visual_velocity_time = (
            self.get_clock().now()
        )

    @staticmethod
    def skew(
        vector
    ):
        """
        将三维向量转换为反对称矩阵。
        """

        x, y, z = vector

        return np.asarray(
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
        将相机坐标系速度 V_c 转换为
        末端执行器坐标系速度 V_e。

        输入：
            V_c:
                [vx, vy, vz, wx, wy, wz]

        输出：
            V_e:
                [vx, vy, vz, wx, wy, wz]
        """

        R_e_c = (
            self.T_end_effector_camera[
                :3,
                :3
            ]
        )

        t_e_c = (
            self.T_end_effector_camera[
                :3,
                3
            ]
        )

        adjoint_e_c = np.zeros(
            (6, 6),
            dtype=float
        )

        # 当前速度排列：
        # [线速度, 角速度]
        adjoint_e_c[:3, :3] = R_e_c

        adjoint_e_c[:3, 3:] = (
            self.skew(t_e_c) @
            R_e_c
        )

        adjoint_e_c[3:, 3:] = R_e_c

        V_e = adjoint_e_c @ V_c

        return V_e

    def message_is_fresh(
        self,
        last_message_time,
        timeout_sec,
        now
    ):
        """
        判断某一类消息是否在允许时间内更新。

        输出：
            is_fresh:
                是否仍然有效。

            age_sec:
                距离最后一次消息的时间。
                从未收到时返回 None。
        """

        if last_message_time is None:
            return False, None

        age_sec = (
            now -
            last_message_time
        ).nanoseconds * 1e-9

        # 防止 ROS 时间发生小幅回跳
        age_sec = max(
            0.0,
            float(age_sec)
        )

        is_fresh = (
            age_sec <= timeout_sec
        )

        return is_fresh, age_sec

    def timer_callback(self):
        """
        周期计算并发送关节速度。

        任何输入失效时，立即进入零速度状态。
        """

        now = self.get_clock().now()

        elapsed = (
            now -
            self.start_time
        ).nanoseconds * 1e-9

        # =====================================================
        # 控制持续时间检查
        # =====================================================

        if (
            self.duration_sec > 0.0
            and elapsed >= self.duration_sec
        ):
            self.enter_safe_stop(
                "Control duration reached."
            )

            self.get_logger().info(
                "Duration reached. Stop."
            )

            rclpy.shutdown()
            return

        # =====================================================
        # 关节状态看门狗
        # =====================================================

        (
            joint_state_is_fresh,
            joint_state_age
        ) = self.message_is_fresh(
            self.last_joint_state_time,
            self.joint_state_timeout_sec,
            now
        )

        if (
            self.current_q is None
            or not joint_state_is_fresh
        ):
            if joint_state_age is None:
                reason = (
                    "Waiting for the first "
                    "joint state message."
                )
            else:
                reason = (
                    "Joint state timeout: "
                )

            self.enter_safe_stop(reason)
            return

        # =====================================================
        # 视觉速度看门狗
        # =====================================================

        (
            visual_velocity_is_fresh,
            visual_velocity_age
        ) = self.message_is_fresh(
            self.last_visual_velocity_time,
            self.visual_velocity_timeout_sec,
            now
        )

        if not visual_velocity_is_fresh:
            if visual_velocity_age is None:
                reason = (
                    "Waiting for the first "
                    "visual velocity message."
                )
            else:
                reason = (
                    "Visual velocity timeout: "
                )

            self.enter_safe_stop(reason)
            return

        # 两类数据都恢复后，清除安全停止状态
        if self.safe_stop_reason is not None:
            self.get_logger().info(
                "Joint state and visual velocity "
                "are valid again."
            )

            self.safe_stop_reason = None

        # =====================================================
        # 计算关节速度
        # =====================================================

        try:
            J = self.kinematics.compute_jacobian(
                self.current_q
            )

            J = np.asarray(
                J,
                dtype=float
            )

            if J.shape != (6, 7):
                raise ValueError(
                    "Jacobian must have shape "
                    f"(6, 7), but got {J.shape}."
                )

            if not np.all(np.isfinite(J)):
                raise ValueError(
                    "Jacobian contains nan or inf."
                )

            q_dot = (
                cartesian_velocity_to_joint_velocity(
                    V_e=self.V_e,
                    J=J,
                    damping=self.damping
                )
            )

            q_dot = limit_joint_velocity(
                q_dot,
                max_abs=(
                    self.max_joint_velocity
                )
            )

        except Exception as error:
            self.enter_safe_stop(
                "Failed to compute safe joint "
                f"velocity: {error}"
            )
            return

        self.get_logger().info(
            f"q: "
            f"{np.round(self.current_q, 4).tolist()} | "
            f"q_dot: "
            f"{np.round(q_dot, 5).tolist()}",
            throttle_duration_sec=10.0
        )

        if not self.dry_run:
            message = Float64MultiArray()
            message.data = q_dot.tolist()

            self.publisher.publish(message)

    def enter_safe_stop(
        self,
        reason
    ):
        """
        进入安全停止状态。

        操作：
            1. 清除内部末端速度。
            2. 发布七维零关节速度。
            3. 仅在停止原因变化时打印警告。
        """

        self.V_e = np.zeros(
            6,
            dtype=float
        )

        if reason != self.safe_stop_reason:
            self.get_logger().warning(
                f"Safety stop: {reason}"
            )

            self.safe_stop_reason = reason

        # 在每个控制周期持续发送零速度，
        # 避免底层保留之前的非零命令。
        self.publish_zero_velocity()

    def publish_zero_velocity(self):
        """
        发布七维零关节速度。

        dry_run=True 时只清除内部速度，
        不向实际控制话题发送消息。
        """

        self.V_e = np.zeros(
            6,
            dtype=float
        )

        if self.dry_run:
            return

        message = Float64MultiArray()
        message.data = [0.0] * 7

        self.publisher.publish(message)


def main(args=None):
    """
    ROS 2 节点入口。
    """

    rclpy.init(args=args)

    node = None

    try:
        node = CartesianServoNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info(
                "Keyboard interrupt. "
                "Send zero velocity."
            )

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f"Unexpected error: {error}"
            )

        raise

    finally:
        # 只要 ROS 上下文仍有效，
        # 在退出节点前再发送一次零速度。
        if node is not None:
            if rclpy.ok():
                node.publish_zero_velocity()

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()