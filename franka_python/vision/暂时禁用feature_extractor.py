#!/usr/bin/env python3
"""
green_ball_detector.py

功能：
    使用传统计算机视觉方法检测RGB图像中的绿色小球。
    通过颜色分割、形态学处理、轮廓提取以及圆度筛选，
    直接提取目标小球的视觉特征 [u, v, r]，
    用于后续视觉伺服控制。

接口：
    GreenBallDetector.detect(image)

输入：
    image:
        RGB相机采集的图像。
        数据类型为 numpy.ndarray。
        图像格式为 (H, W, 3)。

输出：
    detect():
        检测到目标时返回：

        [
            u,
            v,
            r
        ]

        其中：
            u:
                球心横向像素坐标。

            v:
                球心纵向像素坐标。

            r:
                球半径，单位为像素。

        未检测到目标时返回 None。

方法：
    1. RGB图像转换到HSV颜色空间。
    2. 根据绿色HSV范围进行颜色分割。
    3. 使用形态学操作去除噪声。
    4. 提取轮廓并计算圆度。
    5. 根据面积和圆度筛选目标。
    6. 使用最小外接圆计算球心和半径。
"""

import cv2
import numpy as np


class GreenBallDetector:
    """
    绿色小球检测器。
    """

    def __init__(
        self,
        hsv_lower=(35, 50, 50),
        hsv_upper=(85, 255, 255),
        min_area=100,
        min_circularity=0.75,
    ):
        """
        初始化绿色小球检测参数。

        输入：
            hsv_lower:
                HSV绿色区域下限。

            hsv_upper:
                HSV绿色区域上限。

            min_area:
                最小轮廓面积。

            min_circularity:
                最小圆度阈值。
        """

        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)

        self.min_area = min_area
        self.min_circularity = min_circularity

    def detect(self, image):
        """
        检测图像中的绿色小球。

        输入：
            image:
                RGB格式图像。

        输出：
            current_feature:
                检测成功时返回 [u, v, r]；
                检测失败时返回 None。
        """

        if image is None:
            return None

        # RGB图像转换到HSV颜色空间
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        )

        # 根据HSV阈值提取绿色区域
        mask = cv2.inRange(
            hsv,
            self.hsv_lower,
            self.hsv_upper
        )

        # 形态学处理，去除小噪声并填补区域
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # 提取轮廓
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter == 0:
                continue

            # 计算轮廓圆度
            # 圆度越接近1，轮廓越接近圆形
            circularity = (
                4 *
                np.pi *
                area /
                (perimeter ** 2)
            )

            if circularity < self.min_circularity:
                continue

            # 计算轮廓最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(
                contour
            )

            candidates.append(
                {
                    "u": float(x),
                    "v": float(y),
                    "r": float(radius),
                    "area": area,
                    "circularity": circularity,
                }
            )

        # 没有找到符合条件的目标
        if len(candidates) == 0:
            return None

        # 选择面积和圆度综合得分最大的目标
        target = max(
            candidates,
            key=lambda item:
                item["area"] *
                item["circularity"]
        )

        # 直接返回视觉伺服所需的 [u, v, r]
        current_feature = np.array(
            [
                target["u"],
                target["v"],
                target["r"]
            ],
            dtype=float
        )

        return current_feature