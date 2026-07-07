#!/usr/bin/env python3
"""
feature_extractor.py

功能：
    将目标检测结果转换为视觉伺服使用的视觉特征向量。
    负责从目标检测输出中提取控制所需的图像特征。

接口：
    VisualFeatureExtractor.extract(detection_result)

输入：
    detection_result:
        目标检测结果字典。

        格式：
        {
            "found": bool,
            "center": [u, v],
            "radius": r,
            "confidence": score
        }

输出：
    extract():
        返回视觉特征向量。

        例如：
        [
            u,
            v,
            r
        ]

方法：
    根据指定feature_type选择视觉特征。
    当前支持：
        1. position:
            使用球心位置 [u, v]

        2. position_radius:
            使用球心位置和半径 [u, v, r]
"""

import numpy as np


class VisualFeatureExtractor:
    """
    视觉特征提取器。
    """


    def __init__(
        self,
        feature_type="position_radius"
    ):
        """
        初始化视觉特征类型。

        输入：
            feature_type:
                视觉特征类型。

                "position":
                    输出 [u, v]

                "position_radius":
                    输出 [u, v, r]
        """

        valid_types = [
            "position",
            "position_radius"
        ]

        if feature_type not in valid_types:
            raise ValueError(
                f"Unsupported feature type: {feature_type}"
            )

        self.feature_type = feature_type


    def extract(self, detection_result):
        """
        根据目标检测结果提取视觉特征。

        输入：
            detection_result:
                green_ball_detector输出结果。

        输出：
            feature:
                numpy数组形式的视觉特征。
                如果没有检测到目标，返回None。
        """

        if detection_result is None:
            return None


        if not detection_result["found"]:
            return None


        center = detection_result["center"]
        radius = detection_result["radius"]


        u = float(center[0])
        v = float(center[1])


        if self.feature_type == "position":

            feature = np.array(
                [
                    u,
                    v
                ],
                dtype=float
            )


        elif self.feature_type == "position_radius":

            feature = np.array(
                [
                    u,
                    v,
                    float(radius)
                ],
                dtype=float
            )


        return feature