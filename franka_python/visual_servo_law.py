#!/usr/bin/env python3
"""
visual_servo_law.py

概括：创建VisualServoLaw节点，在节点中创建GreenBallDetector实例。
输入RGB图像后，检测器输出视觉特征 [u, v, r, Z]，
视觉伺服律根据当前特征和期望特征计算相机速度 V_c，
并通过ROS 2 topic发布。

功能：
    1. 从ROS 2参数文件读取相机参数、目标参数和期望视觉特征。
    2. 调用GreenBallDetector检测绿色小球。
    3. 根据球心特征建立图像雅可比矩阵。
    4. 使用IBVS控制律计算6维相机速度。
    5. 发布 /visual_servo_velocity。

接口：
    VisualServoLaw.process_image(image)

    VisualServoLaw.compute_velocity(
        current_feature,
        desired_feature
    )

输入：
    image:
        RGB相机图像，numpy.ndarray，格式为(H, W, 3)。

    current_feature:
        检测器输出的当前视觉特征：

        [
            u,
            v,
            r,
            Z
        ]

        u、v：
            球心像素坐标。

        r：
            球半径，单位pixel。

        Z：
            目标深度估计，单位m。

    desired_feature:
        期望视觉特征：

        [
            u_star,
            v_star,
            r_star
        ]

输出：
    V_c:
        6维相机速度：

        [
            vx,
            vy,
            vz,
            wx,
            wy,
            wz
        ]

方法：
    1. 调用绿色小球检测器，获得 [u, v, r, Z]。
    2. 将u、v像素误差转换为归一化图像坐标误差。
    3. 使用检测器提供的实时深度Z。
    4. 建立点特征图像雅可比矩阵Ls。
    5. 使用IBVS控制律：

        Vc = -lambda * pinv(Ls) * e

说明：
    当前版本只使用u、v误差控制球心位置。
    desired_radius已经从YAML读取，但暂未参与控制。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from franka_python.vision.camera import USBCamera
from franka_python.vision.green_ball_detector import GreenBallDetector

class VisualServoLaw(Node):
    """
    单目IBVS视觉伺服节点。
    """

    def __init__(self):
        """
        初始化视觉伺服节点。

        参数由ROS 2 YAML文件提供：
            fx
            fy
            cx
            cy
            sphere_radius
            initial_depth
            desired_u
            desired_v
            desired_radius
            lambda_gain
            visual_velocity_topic
        """

        super().__init__("visual_servo_law")

        # 相机内参
        self.declare_parameter("fx", 600.0)
        self.declare_parameter("fy", 600.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)

        # 目标球参数
        self.declare_parameter("sphere_radius", 0.02)
        self.declare_parameter("initial_depth", 0.5)

        # 期望视觉特征
        self.declare_parameter("desired_u", 320.0)
        self.declare_parameter("desired_v", 240.0)
        self.declare_parameter("desired_radius", 30.0)

        # 打开USB摄像头
        # camera_index=3对应/dev/video3
        self.camera = USBCamera(
            camera_index=3,
            width=640,
            height=480,
            fps=30
)

        # IBVS参数
        self.declare_parameter("lambda_gain", 0.001)

        # ROS topic
        self.declare_parameter(
            "visual_velocity_topic",
            "/visual_servo_velocity"
        )

        self.fx = float(
            self.get_parameter("fx").value
        )

        self.fy = float(
            self.get_parameter("fy").value
        )

        self.cx = float(
            self.get_parameter("cx").value
        )

        self.cy = float(
            self.get_parameter("cy").value
        )

        sphere_radius = float(
            self.get_parameter(
                "sphere_radius"
            ).value
        )

        initial_depth = float(
            self.get_parameter(
                "initial_depth"
            ).value
        )

        self.lambda_gain = float(
            self.get_parameter(
                "lambda_gain"
            ).value
        )

        self.desired_feature = np.array(
            [
                float(
                    self.get_parameter(
                        "desired_u"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "desired_v"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "desired_radius"
                    ).value
                ),
            ],
            dtype=float
        )

        visual_velocity_topic = str(
            self.get_parameter(
                "visual_velocity_topic"
            ).value
        )

        # detector.detect(image)统一输出：
        # [u, v, r, Z]
        self.detector = GreenBallDetector(
            fx=self.fx,
            fy=self.fy,
            sphere_radius=sphere_radius,
            initial_depth=initial_depth,
            hsv_lower=[30, 20, 20],
            hsv_upper=[90, 255, 255]  # H：色相，决定是什么颜色,S：饱和度，决定颜色有多鲜艳,V：亮度，决定颜色有多亮
        )
        """
        H = 0       红色
        H = 15～30  黄色、黄绿色
        H = 30～90  绿色区域
        H = 90～130 蓝色区域
        H = 150以上 紫红、红色

        S 接近 0：白色、灰色，颜色很淡
        S 较小：浅绿色、灰绿色
        S 较大：鲜绿色、深绿色
        S = 255：颜色非常鲜艳

        V 接近 0：非常黑
        V 较小：暗绿色
        V 较大：亮绿色
        V = 255：非常亮
        
        """
        

        # 发布视觉伺服计算得到的相机速度
        # V_c = [vx, vy, vz, wx, wy, wz]
        self.velocity_publisher = self.create_publisher(
            Float64MultiArray,
            visual_velocity_topic,
            10
        )

        self.get_logger().info(
            "VisualServoLaw started. "
            f"Desired feature: "
            f"{self.desired_feature.tolist()}"
        )


        # 每秒读取约30帧图像
        self.camera_timer = self.create_timer(
            1.0 / 30.0,
            self.camera_callback
        )

    def camera_callback(self):
        """
        读取一帧USB相机图像并执行视觉伺服。
        """

        image = self.camera.read()

        if image is None:
            self.publish_zero_velocity()
            return

        self.process_image(image)


    def process_image(
        self,
        image
    ):
        """
        从图像中检测绿色小球，并计算视觉伺服速度。

        输入：
            image:
                RGB相机图像。

        输出：
            V_c:
                检测成功时返回6维相机速度。

            None:
                未检测到绿色小球时返回None。
        """

        current_feature = self.detector.detect(
            image
        )

        # 未检测到目标时发布零速度
        if current_feature is None:
            self.publish_zero_velocity()
            return None

        return self.compute_velocity(
            current_feature,
            self.desired_feature
        )

    def compute_error(
        self,
        current_feature,
        desired_feature
    ):
        """
        计算视觉误差。

        current_feature:
            [u, v, r, Z]

        desired_feature:
            [u_star, v_star, r_star]

        图像雅可比矩阵使用归一化图像坐标，
        因此将u、v像素误差分别除以fx、fy。
        """

        current_feature = np.asarray(
            current_feature,
            dtype=float
        )

        desired_feature = np.asarray(
            desired_feature,
            dtype=float
        )

        if current_feature.shape != (4,):
            raise ValueError(
                "current_feature必须为[u, v, r, Z]。"
            )

        if desired_feature.shape != (3,):
            raise ValueError(
                "desired_feature必须为"
                "[u_star, v_star, r_star]。"
            )

        error = (
            current_feature[:3] -
            desired_feature
        )

        error[0] = error[0] / self.fx
        error[1] = error[1] / self.fy

        return error

    def interaction_matrix(
        self,
        feature,
        Z
    ):
        """
        计算球心点特征的图像雅可比矩阵。

        输入：
            feature:
                [u, v, r, Z]

            Z:
                检测器估计的目标深度，单位m。

        输出：
            L_uv:
                2x6图像雅可比矩阵。
        """

        u = feature[0]
        v = feature[1]

        # 像素坐标转换为归一化图像坐标
        x = (
            u - self.cx
        ) / self.fx

        y = (
            v - self.cy
        ) / self.fy

        L_uv = np.array(
            [
                [
                    -1.0 / Z,
                    0.0,
                    x / Z,
                    x * y,
                    -(1.0 + x * x),
                    y
                ],
                [
                    0.0,
                    -1.0 / Z,
                    y / Z,
                    1.0 + y * y,
                    -x * y,
                    -x
                ]
            ],
            dtype=float
        )

        return L_uv

    def compute_velocity(
        self,
        current_feature,
        desired_feature
    ):
        """
        根据当前视觉特征计算IBVS相机速度。
        """

        current_feature = np.asarray(
            current_feature,
            dtype=float
        )

        if current_feature.shape != (4,):
            self.get_logger().error(
                "Detector output must be [u, v, r, Z]."
            )
            self.publish_zero_velocity()
            return np.zeros(6, dtype=float)

        Z = current_feature[3]

        if (
            not np.all(
                np.isfinite(current_feature)
            )
            or Z <= 0.0
        ):
            self.get_logger().warn(
                "Invalid visual feature. "
                "Publishing zero velocity."
            )
            self.publish_zero_velocity()
            return np.zeros(6, dtype=float)

        error = self.compute_error(
            current_feature,
            desired_feature
        )

        Ls = self.interaction_matrix(
            current_feature,
            Z
        )

        # 当前版本只使用u、v误差
        e_uv = error[:2]

        V_c = (
            -self.lambda_gain *
            np.linalg.pinv(Ls) @
            e_uv
        )

        if not np.all(np.isfinite(V_c)):
            self.get_logger().warn(
                "Invalid IBVS velocity. "
                "Publishing zero velocity."
            )
            self.publish_zero_velocity()
            return np.zeros(6, dtype=float)

        msg = Float64MultiArray()
        msg.data = V_c.tolist()

        self.velocity_publisher.publish(msg)

        return V_c

    def publish_zero_velocity(self):
        """
        发布6维零速度。
        """

        msg = Float64MultiArray()
        msg.data = [0.0] * 6

        self.velocity_publisher.publish(msg)


def main(args=None):
    """
    ROS 2节点运行入口。
    """

    rclpy.init(args=args)

    node = VisualServoLaw()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.publish_zero_velocity()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()