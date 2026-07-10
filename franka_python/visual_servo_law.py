#!/usr/bin/env python3
"""
visual_servo_law.py

概括：
    创建 VisualServoLaw 节点，在节点中创建 GreenBallDetector 实例。
    检测器输出视觉特征 [u, v, r, Z]，控制器使用球心和半径三个误差，
    计算相机坐标系下的三维平移速度，并发布 6 维相机速度 V_c。

功能：
    1. 从 ROS 2 参数文件读取相机参数、目标参数和期望视觉特征。
    2. 调用 GreenBallDetector 检测绿色小球。
    3. 使用 u、v、r 三个视觉特征控制相机平移。
    4. 使用逆深度低通滤波，减小半径波动引起的深度抖动。
    5. 使用误差相关增益：误差大时回归更快，接近目标时更平稳。
    6. 对相机平移速度进行限幅。
    7. 发布 /visual_servo_velocity。
    8. 图像或目标丢失时连续发布零速度。

接口：
    VisualServoLaw.process_image(image)

    VisualServoLaw.compute_velocity(
        current_feature,
        desired_feature
    )

输入：
    image:
        RGB 相机图像，numpy.ndarray，格式为 (H, W, 3)。

    current_feature:
        检测器输出的当前视觉特征 [u, v, r, Z]。

        u、v：球心像素坐标。
        r：球半径，单位 pixel。
        Z：由球半径估计的目标深度，单位 m。

    desired_feature:
        期望视觉特征 [u_star, v_star, r_star]。

输出：
    V_c:
        6 维相机速度 [vx, vy, vz, 0, 0, 0]。

方法：
    1. 定义归一化球心误差：

        e_x = (u - u_star) / fx
        e_y = (v - v_star) / fy

    2. 定义尺度误差：

        e_r = log(r / r_star)

    3. 对逆深度 rho = 1 / Z 进行低通滤波，得到 Z_hat。

    4. 只控制三个平移自由度：

        vz = -k_r * Z_hat * e_r
        vx = x * vz + k_xy * Z_hat * e_x
        vy = y * vz + k_xy * Z_hat * e_y

    5. 角速度固定为零，避免一个球形目标产生不必要的姿态旋转。

说明：
    lambda_gain 参数仅为兼容旧 YAML 保留，当前三平移控制器不再使用。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import cv2

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
            xy_gain_min
            xy_gain_max
            xy_gain_beta
            radius_gain
            depth_filter_alpha
            min_depth
            max_depth
            pixel_deadband
            radius_deadband
            max_xy_speed
            max_z_speed
            enable_depth_control
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

        # 绿色目标HSV阈值
        # 具体数值必须从YAML或命令行传入
        self.declare_parameter("hsv_lower")
        self.declare_parameter("hsv_upper")

        # 摄像头设备编号，通常 0 对应 /dev/video0，1 对应 /dev/video1，由yaml里得到
        self.declare_parameter("camera_index", 2)

        # 打开USB摄像头
        # camera_index=2对应/dev/video2
        self.camera = USBCamera(
            camera_index=self.get_parameter("camera_index").value,
            width=640,
            height=480,
            fps=30
)

        # 旧版参数，仅为兼容现有 YAML 保留。
        # 当前三平移控制器不再使用 lambda_gain。
        self.declare_parameter("lambda_gain", 0.001)

        # 三平移视觉伺服参数
        # xy_gain 根据球心误差在最小值和最大值之间自适应变化。
        self.declare_parameter("xy_gain_min", 0.15)
        self.declare_parameter("xy_gain_max", 0.60)
        self.declare_parameter("xy_gain_beta", 6.0)

        # 半径误差控制 Z 方向速度的增益。
        self.declare_parameter("radius_gain", 0.25)

        # 逆深度低通滤波系数，范围 (0, 1]。
        # 越小越平滑，越大响应越快。
        self.declare_parameter("depth_filter_alpha", 0.20)

        # 深度有效范围，超过范围时先限幅再进入滤波。
        self.declare_parameter("min_depth", 0.20)
        self.declare_parameter("max_depth", 1.20)

        # 小误差死区，用于减小目标附近抖动。
        self.declare_parameter("pixel_deadband", 2.0)
        self.declare_parameter("radius_deadband", 1.0)

        # 相机平移速度限制，单位 m/s。
        self.declare_parameter("max_xy_speed", 0.020)
        self.declare_parameter("max_z_speed", 0.010)

        # 是否启用半径误差控制前后距离。
        self.declare_parameter("enable_depth_control", False)

        # 未检测到目标时的零速度连发参数
        # 默认立即发送1次，再以0.01 s间隔补发，共发送5次
        self.declare_parameter(
            "zero_velocity_repeat_count",
            5
        )
        self.declare_parameter(
            "zero_velocity_repeat_interval_sec",
            0.01
        )

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

        self.hsv_lower = list(
            self.get_required_parameter(
                "hsv_lower"
            )
        )

        self.hsv_upper = list(
            self.get_required_parameter(
                "hsv_upper"
            )
        )

        if (
            len(self.hsv_lower) != 3
            or len(self.hsv_upper) != 3
        ):
            raise ValueError(
                "hsv_lower and hsv_upper must contain "
                "three values: [H, S, V]."
            )

        # 读取旧版参数但不再参与当前控制律，
        # 避免现有 YAML 中保留 lambda_gain 时启动失败。
        self.lambda_gain = float(
            self.get_parameter(
                "lambda_gain"
            ).value
        )

        self.xy_gain_min = float(
            self.get_parameter(
                "xy_gain_min"
            ).value
        )

        self.xy_gain_max = float(
            self.get_parameter(
                "xy_gain_max"
            ).value
        )

        self.xy_gain_beta = float(
            self.get_parameter(
                "xy_gain_beta"
            ).value
        )

        self.radius_gain = float(
            self.get_parameter(
                "radius_gain"
            ).value
        )

        self.depth_filter_alpha = float(
            self.get_parameter(
                "depth_filter_alpha"
            ).value
        )

        self.min_depth = float(
            self.get_parameter(
                "min_depth"
            ).value
        )

        self.max_depth = float(
            self.get_parameter(
                "max_depth"
            ).value
        )

        self.pixel_deadband = float(
            self.get_parameter(
                "pixel_deadband"
            ).value
        )

        self.radius_deadband = float(
            self.get_parameter(
                "radius_deadband"
            ).value
        )

        self.max_xy_speed = float(
            self.get_parameter(
                "max_xy_speed"
            ).value
        )

        self.max_z_speed = float(
            self.get_parameter(
                "max_z_speed"
            ).value
        )

        self.enable_depth_control = bool(
            self.get_parameter(
                "enable_depth_control"
            ).value
        )

        self.zero_velocity_repeat_count = int(
            self.get_parameter(
                "zero_velocity_repeat_count"
            ).value
        )

        self.zero_velocity_repeat_interval_sec = float(
            self.get_parameter(
                "zero_velocity_repeat_interval_sec"
            ).value
        )

        if self.zero_velocity_repeat_count <= 0:
            raise ValueError(
                "zero_velocity_repeat_count must be positive."
            )

        if self.zero_velocity_repeat_interval_sec <= 0.0:
            raise ValueError(
                "zero_velocity_repeat_interval_sec "
                "must be positive."
            )

        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(
                "fx and fy must be positive."
            )

        if self.xy_gain_min <= 0.0:
            raise ValueError(
                "xy_gain_min must be positive."
            )

        if self.xy_gain_max < self.xy_gain_min:
            raise ValueError(
                "xy_gain_max must be greater than or "
                "equal to xy_gain_min."
            )

        if self.xy_gain_beta <= 0.0:
            raise ValueError(
                "xy_gain_beta must be positive."
            )

        if self.radius_gain <= 0.0:
            raise ValueError(
                "radius_gain must be positive."
            )

        if not 0.0 < self.depth_filter_alpha <= 1.0:
            raise ValueError(
                "depth_filter_alpha must be in (0, 1]."
            )

        if self.min_depth <= 0.0:
            raise ValueError(
                "min_depth must be positive."
            )

        if self.max_depth <= self.min_depth:
            raise ValueError(
                "max_depth must be greater than min_depth."
            )

        if self.pixel_deadband < 0.0:
            raise ValueError(
                "pixel_deadband must be non-negative."
            )

        if self.radius_deadband < 0.0:
            raise ValueError(
                "radius_deadband must be non-negative."
            )

        if self.max_xy_speed <= 0.0:
            raise ValueError(
                "max_xy_speed must be positive."
            )

        if self.max_z_speed <= 0.0:
            raise ValueError(
                "max_z_speed must be positive."
            )

        # 保存滤波后的逆深度 rho_hat = 1 / Z_hat。
        # 第一次有效检测时用测量值初始化。
        self.filtered_inverse_depth = None
        self.last_filtered_depth = None

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

        if self.desired_feature[2] <= 0.0:
            raise ValueError(
                "desired_radius must be positive."
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
            hsv_lower=self.hsv_lower,
            hsv_upper=self.hsv_upper
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

        # 零速度补发状态。
        # remaining表示还需要由定时器补发多少次零速度。
        self.zero_velocity_remaining = 0

        # 当前安全停止原因，用于避免同一故障反复打印警告。
        self.safe_stop_reason = None

        # 独立零速度补发定时器。
        # 定时器一直存在，但remaining为0时不会发布消息。
        self.zero_velocity_timer = self.create_timer(
            self.zero_velocity_repeat_interval_sec,
            self.zero_velocity_timer_callback
        )

        self.get_logger().info(
            "VisualServoLaw started. "
            f"Desired feature: "
            f"{self.desired_feature.tolist()}"
        )

        self.get_logger().info(
            "Zero-velocity burst: "
            f"{self.zero_velocity_repeat_count} messages, "
            f"interval "
            f"{self.zero_velocity_repeat_interval_sec:.3f} s."
        )

        self.get_logger().info(
            "Three-translation IBVS: "
            f"xy_gain=[{self.xy_gain_min:.3f}, "
            f"{self.xy_gain_max:.3f}], "
            f"radius_gain={self.radius_gain:.3f}, "
            f"max_xy={self.max_xy_speed:.3f} m/s, "
            f"max_z={self.max_z_speed:.3f} m/s, "
            f"depth_control={self.enable_depth_control}."
        )

        # 每秒读取约30帧图像
        self.camera_timer = self.create_timer(
            1.0 / 30.0,
            self.camera_callback
        )


    def get_required_parameter(self, name):
        """
        读取必须由YAML或命令行提供的参数。

        如果参数没有设置，则直接报错。
        """

        param = self.get_parameter(name)

        if param.type_ == rclpy.Parameter.Type.NOT_SET:
            raise ValueError(
                f"Required parameter '{name}' is not set. "
                f"Please provide it in YAML or command line."
            )

        return param.value

    def camera_callback(self):
        """
        读取一帧USB相机图像并执行视觉伺服。
        """

        image = self.camera.read()

        if image is None:
            self.request_zero_velocity_burst(
                "Camera image is unavailable."
            )
            return

        self.process_image(image)


    def process_image(
        self,
        image
    ):
        """
        从图像中检测绿色小球、显示检测结果，
        并计算视觉伺服速度。
        """

        current_feature = self.detector.detect(
            image
        )

        # 复制一份图像，只在复制图像上绘制，
        # 避免影响检测器使用的原始图像。
        display_image = image.copy()

        if current_feature is None:
            # 没有检测到目标时显示文字
            cv2.putText(
                display_image,
                "Target not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            # 绘制期望目标中心
            desired_u = int(
                round(self.desired_feature[0])
            )
            desired_v = int(
                round(self.desired_feature[1])
            )

            cv2.drawMarker(
                display_image,
                (desired_u, desired_v),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=25,
                thickness=2
            )

            # 显示图像
            cv2.imshow(
                "Visual Servo",
                display_image
            )
            cv2.waitKey(1)

            self.request_zero_velocity_burst(
                "Target object was not detected."
            )

            return None

        # 提取检测结果
        u = int(round(current_feature[0]))
        v = int(round(current_feature[1]))
        r = int(round(current_feature[2]))
        Z = float(current_feature[3])

        # 绘制检测到的小球轮廓
        cv2.circle(
            display_image,
            (u, v),
            r,
            (0, 255, 0),
            2
        )

        # 绘制检测到的小球中心
        cv2.circle(
            display_image,
            (u, v),
            4,
            (0, 0, 255),
            -1
        )

        # 绘制期望位置
        desired_u = int(
            round(self.desired_feature[0])
        )
        desired_v = int(
            round(self.desired_feature[1])
        )

        cv2.drawMarker(
            display_image,
            (desired_u, desired_v),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2
        )

        # 绘制期望球半径，便于观察 Z 方向控制目标。
        desired_radius = int(
            round(self.desired_feature[2])
        )

        cv2.circle(
            display_image,
            (desired_u, desired_v),
            desired_radius,
            (255, 0, 0),
            1
        )

        # 显示当前检测结果
        cv2.putText(
            display_image,
            f"u={u}, v={v}, r={r}, Z={Z:.3f} m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Visual Servo",
            display_image
        )

        # imshow后必须调用waitKey，
        # 否则窗口通常不会正常刷新。
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.publish_zero_velocity()
            rclpy.shutdown()
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
        计算三个无量纲视觉误差。

        输出：
            error = [e_x, e_y, e_r]

            e_x = (u - u_star) / fx
            e_y = (v - v_star) / fy
            e_r = log(r / r_star)

        对球心像素误差和半径误差设置死区，
        用于减小目标附近的速度抖动。
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

        current_radius = current_feature[2]
        desired_radius = desired_feature[2]

        if current_radius <= 0.0:
            raise ValueError(
                "Current radius must be positive."
            )

        if desired_radius <= 0.0:
            raise ValueError(
                "Desired radius must be positive."
            )

        error_u_pixel = (
            current_feature[0] -
            desired_feature[0]
        )

        error_v_pixel = (
            current_feature[1] -
            desired_feature[1]
        )

        error_radius_pixel = (
            current_radius -
            desired_radius
        )

        if abs(error_u_pixel) <= self.pixel_deadband:
            error_u_pixel = 0.0

        if abs(error_v_pixel) <= self.pixel_deadband:
            error_v_pixel = 0.0

        if (
            abs(error_radius_pixel) <=
            self.radius_deadband
        ):
            error_radius = 0.0
        else:
            error_radius = float(
                np.log(
                    current_radius /
                    desired_radius
                )
            )

        error = np.asarray(
            [
                error_u_pixel / self.fx,
                error_v_pixel / self.fy,
                error_radius,
            ],
            dtype=float
        )

        return error

    def filter_depth(
        self,
        measured_depth
    ):
        """
        对逆深度 rho = 1 / Z 进行一阶低通滤波。

        使用逆深度而不是直接滤波 Z，原因是图像雅可比矩阵
        的平移部分对 1 / Z 线性。
        """

        measured_depth = float(
            np.clip(
                measured_depth,
                self.min_depth,
                self.max_depth
            )
        )

        measured_inverse_depth = (
            1.0 / measured_depth
        )

        if self.filtered_inverse_depth is None:
            self.filtered_inverse_depth = (
                measured_inverse_depth
            )
        else:
            alpha = self.depth_filter_alpha

            self.filtered_inverse_depth = (
                (1.0 - alpha) *
                self.filtered_inverse_depth +
                alpha *
                measured_inverse_depth
            )

        filtered_depth = (
            1.0 /
            self.filtered_inverse_depth
        )

        filtered_depth = float(
            np.clip(
                filtered_depth,
                self.min_depth,
                self.max_depth
            )
        )

        self.last_filtered_depth = filtered_depth

        return filtered_depth

    def compute_xy_gain(
        self,
        error_x,
        error_y
    ):
        """
        根据球心误差计算自适应平移增益。

        误差大时接近 xy_gain_max，回归更快；
        误差小时接近 xy_gain_min，减小目标附近抖动。
        """

        center_error_norm = float(
            np.hypot(
                error_x,
                error_y
            )
        )

        gain = (
            self.xy_gain_min +
            (
                self.xy_gain_max -
                self.xy_gain_min
            ) *
            np.tanh(
                self.xy_gain_beta *
                center_error_norm
            )
        )

        return float(gain)

    def limit_translation_velocity(
        self,
        vx,
        vy,
        vz
    ):
        """
        对相机平移速度进行限幅。

        x-y 平面按合速度限幅，z 方向单独限幅。
        """

        planar_speed = float(
            np.hypot(vx, vy)
        )

        if planar_speed > self.max_xy_speed:
            scale = (
                self.max_xy_speed /
                planar_speed
            )

            vx *= scale
            vy *= scale

        vz = float(
            np.clip(
                vz,
                -self.max_z_speed,
                self.max_z_speed
            )
        )

        return float(vx), float(vy), vz

    def compute_velocity(
        self,
        current_feature,
        desired_feature
    ):
        """
        根据球心和半径三个视觉误差计算相机速度。

        当前控制器只输出相机平移速度：
            V_c = [vx, vy, vz, 0, 0, 0]

        这样可以避免使用一个球形目标时，
        2 个球心误差通过伪逆被分配到多个旋转自由度。
        """

        current_feature = np.asarray(
            current_feature,
            dtype=float
        )

        if current_feature.shape != (4,):
            self.get_logger().error(
                "Detector output must be [u, v, r, Z]."
            )
            self.request_zero_velocity_burst(
                "Detector output has an invalid shape."
            )
            return np.zeros(6, dtype=float)

        if not np.all(
            np.isfinite(current_feature)
        ):
            self.get_logger().warn(
                "Invalid visual feature. "
                "Publishing zero velocity."
            )
            self.request_zero_velocity_burst(
                "Visual feature contains nan or inf."
            )
            return np.zeros(6, dtype=float)

        measured_depth = float(
            current_feature[3]
        )

        if (
            current_feature[2] <= 0.0 or
            measured_depth <= 0.0
        ):
            self.get_logger().warn(
                "Invalid radius or depth. "
                "Publishing zero velocity."
            )
            self.request_zero_velocity_burst(
                "Visual radius or depth is not positive."
            )
            return np.zeros(6, dtype=float)

        try:
            error = self.compute_error(
                current_feature,
                desired_feature
            )

            filtered_depth = self.filter_depth(
                measured_depth
            )

        except ValueError as error_message:
            self.request_zero_velocity_burst(
                f"Failed to compute visual error: "
                f"{error_message}"
            )
            return np.zeros(6, dtype=float)

        error_x = float(error[0])
        error_y = float(error[1])
        error_radius = float(error[2])

        # 当前球心的归一化图像坐标。
        x = (
            current_feature[0] -
            self.cx
        ) / self.fx

        y = (
            current_feature[1] -
            self.cy
        ) / self.fy

        xy_gain = self.compute_xy_gain(
            error_x,
            error_y
        )

        if self.enable_depth_control:
            # r = fR/Z，因此 d(log(r))/dt = vz/Z。
            # 令 d(e_r)/dt = -k_r e_r，得到：
            # vz = -k_r Z e_r。
            vz = (
                -self.radius_gain *
                filtered_depth *
                error_radius
            )
        else:
            vz = 0.0

        # 必须先限制 vz，再计算 x、y 方向的耦合补偿。
        # 否则 vx、vy 会按照未限幅的 vz 计算，而最终发布的 vz
        # 已被限幅，造成补偿量不一致。
        vz = float(
            np.clip(
                vz,
                -self.max_z_speed,
                self.max_z_speed
            )
        )

        # 点特征平移动力学：
        # dx/dt = -vx/Z + x*vz/Z
        # dy/dt = -vy/Z + y*vz/Z
        # 令 dx/dt = -k*e_x、dy/dt = -k*e_y，得到：
        # vx = x*vz + k*Z*e_x
        # vy = y*vz + k*Z*e_y
        vx = (
            x * vz +
            xy_gain *
            filtered_depth *
            error_x
        )

        vy = (
            y * vz +
            xy_gain *
            filtered_depth *
            error_y
        )

        vx, vy, vz = self.limit_translation_velocity(
            vx,
            vy,
            vz
        )

        V_c = np.asarray(
            [
                vx,
                vy,
                vz,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float
        )

        if not np.all(np.isfinite(V_c)):
            self.get_logger().warn(
                "Invalid IBVS velocity. "
                "Publishing zero velocity."
            )
            self.request_zero_velocity_burst(
                "IBVS velocity contains invalid values."
            )
            return np.zeros(6, dtype=float)

        # 目标刚刚恢复时，如果上一轮零速度还没有补发完，
        # 不允许非零速度与零速度交叉发布。
        # 等补发完成后，下一帧有效目标才能恢复控制。
        if self.zero_velocity_remaining > 0:
            return np.zeros(6, dtype=float)

        if self.safe_stop_reason is not None:
            self.get_logger().info(
                "Valid target recovered. "
                "Visual servo control resumed."
            )
            self.safe_stop_reason = None

        msg = Float64MultiArray()
        msg.data = V_c.tolist()

        self.velocity_publisher.publish(msg)

        self.get_logger().info(
            "feature="
            f"[{current_feature[0]:.1f}, "
            f"{current_feature[1]:.1f}, "
            f"{current_feature[2]:.1f}] | "
            "error="
            f"[{error_x:.4f}, "
            f"{error_y:.4f}, "
            f"{error_radius:.4f}] | "
            f"Z_hat={filtered_depth:.3f} m | "
            f"k_xy={xy_gain:.3f} | "
            "V_c="
            f"{np.round(V_c, 4).tolist()}",
            throttle_duration_sec=2.0
        )

        return V_c

    def request_zero_velocity_burst(
        self,
        reason
    ):
        """
        请求连续发送多次6维零速度。

        操作：
            1. 立即发送一次零速度。
            2. 设置补发计数器。
            3. 由zero_velocity_timer按固定间隔继续补发。
            4. 相同故障持续发生时，重新补满计数器，
               从而持续发送零速度。
        """

        if reason != self.safe_stop_reason:
            self.get_logger().warn(
                "Safety stop: "
                f"{reason} "
                "Sending repeated zero velocity."
            )
            self.safe_stop_reason = reason

        # 每次再次检测到故障时都补满发送次数。
        # 这样目标持续丢失期间会持续发送零速度。
        self.zero_velocity_remaining = max(
            self.zero_velocity_remaining,
            self.zero_velocity_repeat_count
        )

        # 立即发送第一条，不等待定时器。
        self.publish_zero_velocity_once()
        self.zero_velocity_remaining -= 1

    def zero_velocity_timer_callback(self):
        """
        按固定时间间隔补发零速度。
        """

        if self.zero_velocity_remaining <= 0:
            return

        self.publish_zero_velocity_once()
        self.zero_velocity_remaining -= 1

    def publish_zero_velocity_once(self):
        """
        发布一次6维零速度。
        """

        msg = Float64MultiArray()
        msg.data = [0.0] * 6

        self.velocity_publisher.publish(msg)

    def publish_zero_velocity(self):
        """
        兼容原有调用接口。

        调用后启动一次完整的零速度连发过程。
        """

        self.request_zero_velocity_burst(
            "Zero velocity was explicitly requested."
        )


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