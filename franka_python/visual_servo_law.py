#!/usr/bin/env python3
"""
visual_servo_law.py

功能：
    基于图像雅可比矩阵(Image Jacobian)实现单目IBVS控制律。
    根据视觉特征误差计算机器人末端速度。

接口：
    VisualServoLaw.compute_velocity(
        current_feature,
        desired_feature
    )

输入：
    current_feature:
        当前视觉特征。

        [
            u,
            v,
            r
        ]

    desired_feature:
        期望视觉特征。

        [
            u_star,
            v_star,
            r_star
        ]

输出：
    V_c:
        6维末端速度。

        [
            vx,
            vy,
            vz,
            wx,
            wy,
            wz
        ]


方法：
    1. 计算视觉误差：
        e=s-s*

    2. 根据球半径估计目标深度Z。

    3. 建立图像雅可比矩阵Ls。

    4. 使用IBVS控制律：
        Vc=-lambda*pinv(Ls)*e
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np


class VisualServoLaw(Node):

    def __init__(
        self,
        camera_params,
        target_params,
        lambda_gain=0.001
    ):
        super().__init__("visual_servo_law")
        """
        初始化IBVS参数。

        输入：

            camera_params:
                相机参数。

                {
                    fx,
                    fy,
                    cx,
                    cy
                }


            target_params:
                目标参数。

                {
                    sphere_radius,
                    initial_depth
                }


            lambda_gain:
                IBVS控制增益。
        """

        self.fx = camera_params["fx"]
        self.fy = camera_params["fy"]

        self.cx = camera_params["cx"]
        self.cy = camera_params["cy"]


        # 小球真实半径(m)
        self.sphere_radius = (
            target_params["sphere_radius"]
        )


        # 初始深度(m)
        # 后续可以替换为实时估计Z
        self.initial_depth = (
            target_params["initial_depth"]
        )


        self.lambda_gain = lambda_gain

        # 发布视觉伺服计算得到的末端速度
        # V_e = [vx, vy, vz, wx, wy, wz]
        self.velocity_publisher = self.create_publisher(
            Float64MultiArray,
            "/visual_servo_velocity",
            10
        )


    def compute_error(
        self,
        current_feature,
        desired_feature
    ):
        """
        计算视觉误差。
        """

        current_feature = np.asarray(
            current_feature,
            dtype=float
        )

        desired_feature = np.asarray(
            desired_feature,
            dtype=float
        )


        return (
            current_feature -
            desired_feature
        )


    def estimate_depth(
        self,
        radius
    ):
        """
        根据球半径估计目标深度。

        Z=fR/r

        输入：
            radius:
                图像球半径(pixel)

        输出：
            Z:
                目标深度(m)

        """

        if radius <= 0:
            return self.initial_depth


        f = (
            self.fx +
            self.fy
        ) / 2.0


        Z = (
            f *
            self.sphere_radius /
            radius
        )


        return Z



    def interaction_matrix(
        self,
        feature,
        Z
    ):
        """
        计算球心特征的interaction matrix。

        feature:
            [u,v,r]

        输出:
            Ls
        """

        u = feature[0]
        v = feature[1]
        r = feature[2]


        # 像素坐标转换为归一化坐标

        x = (
            u-self.cx
        ) / self.fx


        y = (
            v-self.cy
        ) / self.fy


        # 点特征interaction matrix

        L_uv = np.array(
            [
                [
                    -1/Z,
                    0,
                    x/Z,
                    x*y,
                    -(1+x*x),
                    y
                ],

                [
                    0,
                    -1/Z,
                    y/Z,
                    1+y*y,
                    -x*y,
                    -x
                ]
            ]
        )


        return L_uv



    def compute_velocity(
        self,
        current_feature,
        desired_feature
    ):
        """
        IBVS计算末端速度。
        """

        error = self.compute_error(
            current_feature,
            desired_feature
        )


        Z = self.estimate_depth(
            current_feature[2]
        )


        Ls = self.interaction_matrix(
            current_feature,
            Z
        )


        # 只使用u,v误差
        e_uv = error[:2]


        # IBVS控制律
        V_c = (
            -self.lambda_gain *
            np.linalg.pinv(Ls) @
            e_uv
        )


        msg = Float64MultiArray()
        msg.data = V_c.tolist()
        self.velocity_publisher.publish(msg)

        return V_c


# ROS2运行入口
def main(args=None):
    rclpy.init(args=args)

    # 后续由YAML提供相机参数和目标参数
    camera_params = {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
    }

    target_params = {
        "sphere_radius": 0.02,
        "initial_depth": 0.5,
    }

    node = VisualServoLaw(
        camera_params,
        target_params
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()