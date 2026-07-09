#!/usr/bin/env python3
"""
green_ball_detector.py

概括：
    创建 GreenBallDetector 类。
    输入 RGB 图像，输出绿色小球视觉特征 [u, v, r, Z]。

功能：
    不再首先使用 HSV 阈值对图像进行硬分割。

    程序首先在整张图像中检测所有可能的闭合轮廓，
    然后分别计算：

    1. 形状得分：
       判断轮廓是否接近圆形。

    2. 颜色得分：
       判断轮廓内部是否接近目标绿色。

    最后使用形状得分和颜色得分的加权结果，
    选择最可能是绿色小球的目标。

接口：
    GreenBallDetector.detect(image)

输入：
    image:
        RGB 相机图像。
        数据类型为 numpy.ndarray。
        图像格式为 (H, W, 3)。

输出：
    detect():
        检测成功时返回：

        [
            u,
            v,
            r,
            Z
        ]

        其中：
            u:
                球心横向像素坐标。

            v:
                球心纵向像素坐标。

            r:
                球半径，单位为像素。

            Z:
                根据球半径估计的目标深度，单位为 m。

        检测失败时返回 None。

方法：
    1. 将 RGB 图像转换为灰度图和 HSV 图像。
    2. 分别从灰度和饱和度通道提取边缘。
    3. 对边缘进行闭运算，连接断裂的轮廓。
    4. 提取整张图像中的所有外部轮廓。
    5. 计算每个轮廓的形状得分。
    6. 计算每个轮廓内部的绿色得分。
    7. 对形状得分和颜色得分进行加权。
    8. 选择总得分最高的轮廓。
    9. 使用最小外接圆计算球心和半径。
    10. 根据 Z = fR / r 估计目标深度。
"""

import cv2
import numpy as np


class GreenBallDetector:
    """
    基于形状和颜色加权的绿色小球检测器。
    """

    def __init__(
        self,
        fx,
        fy,
        sphere_radius,
        initial_depth,
        hsv_lower=(35, 50, 50),
        hsv_upper=(85, 255, 255),
        min_area=100,
        max_area_ratio=0.8,
        min_radius=20.0,
        shape_weight=0.80,
        color_weight=0.20,
        min_total_score=0.58,
        canny_threshold_low=40,
        canny_threshold_high=120,
        morphology_kernel_size=5,
    ):
        """
        初始化检测参数。

        输入：
            fx:
                相机 x 方向焦距，单位 pixel。

            fy:
                相机 y 方向焦距，单位 pixel。

            sphere_radius:
                小球真实半径，单位 m。

            initial_depth:
                半径无效时使用的默认深度，单位 m。

            hsv_lower:
                目标绿色 HSV 范围下限。

            hsv_upper:
                目标绿色 HSV 范围上限。

            min_area:
                候选轮廓最小面积。

            max_area_ratio:
                候选轮廓面积占整幅图像面积的最大比例。

            min_radius:
                候选目标最小半径，单位 pixel。

            shape_weight:
                形状得分权重。

            color_weight:
                颜色得分权重。

            min_total_score:
                最低总得分。

            canny_threshold_low:
                Canny 边缘检测低阈值。

            canny_threshold_high:
                Canny 边缘检测高阈值。

            morphology_kernel_size:
                形态学操作核尺寸。
        """

        self.fx = float(fx)
        self.fy = float(fy)

        self.sphere_radius = float(
            sphere_radius
        )

        self.initial_depth = float(
            initial_depth
        )

        self.hsv_lower = np.asarray(
            hsv_lower,
            dtype=np.uint8
        )

        self.hsv_upper = np.asarray(
            hsv_upper,
            dtype=np.uint8
        )

        self.min_area = float(min_area)

        self.max_area_ratio = float(
            max_area_ratio
        )

        self.min_radius = float(
            min_radius
        )

        total_weight = (
            float(shape_weight) +
            float(color_weight)
        )

        if total_weight <= 0.0:
            raise ValueError(
                "shape_weight 和 color_weight "
                "之和必须大于 0。"
            )

        # 自动归一化权重，保证二者之和为 1
        self.shape_weight = (
            float(shape_weight) /
            total_weight
        )

        self.color_weight = (
            float(color_weight) /
            total_weight
        )

        self.min_total_score = float(
            min_total_score
        )

        self.canny_threshold_low = int(
            canny_threshold_low
        )

        self.canny_threshold_high = int(
            canny_threshold_high
        )

        self.morphology_kernel_size = int(
            morphology_kernel_size
        )

        # 保存最近一次检测到的全部候选目标，
        # 方便后续调试。
        self.last_candidates = []

    def estimate_depth(
        self,
        radius
    ):
        """
        根据图像中的球半径估计目标深度。

        Z = fR / r

        输入：
            radius:
                图像中的球半径，单位 pixel。

        输出：
            Z:
                目标深度估计，单位 m。
        """

        if radius <= 0.0:
            return self.initial_depth

        focal_length = (
            self.fx +
            self.fy
        ) / 2.0

        depth = (
            focal_length *
            self.sphere_radius /
            radius
        )

        return float(depth)

    def calculate_shape_score(
        self,
        contour
    ):
        """
        计算轮廓的圆形形状得分。

        形状得分综合考虑：
        1. circularity：轮廓圆度。
        2. circle_fill_ratio：轮廓面积与最小外接圆面积的比值。
        3. aspect_ratio_score：外接矩形宽高是否接近。
        4. solidity：轮廓面积与凸包面积的比值。

        【伺服优化】：最终输出的 (u, v) 和 radius 使用图像矩和面积积分计算，以提供极高平滑度的控制信号。
        """

        area = cv2.contourArea(
            contour
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if (
            area <= 0.0 or
            perimeter <= 0.0
        ):
            return 0.0, None

        # 1. 计算形状得分（用于过滤干扰）
        circularity = (
            4.0 *
            np.pi *
            area /
            (perimeter ** 2)
        )
        circularity = float(np.clip(circularity, 0.0, 1.0))

        # 计算最小外接圆（仅用于形态学评分，不作为最终坐标输出）
        (enc_x, enc_y), enc_radius = (
            cv2.minEnclosingCircle(
                contour
            )
        )

        if enc_radius <= 0.0:
            return 0.0, None

        circle_area = (
            np.pi *
            enc_radius *
            enc_radius
        )
        circle_fill_ratio = (
            area /
            circle_area
        )
        circle_fill_ratio = float(np.clip(circle_fill_ratio, 0.0, 1.0))

        # 外接矩形宽高比
        _, _, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        if (
            width <= 0 or
            height <= 0
        ):
            return 0.0, None

        aspect_ratio_score = (
            min(width, height) /
            max(width, height)
        )
        aspect_ratio_score = float(np.clip(aspect_ratio_score, 0.0, 1.0))

        # 凸包填充率
        convex_hull = cv2.convexHull(
            contour
        )
        hull_area = cv2.contourArea(
            convex_hull
        )

        if hull_area > 0.0:
            solidity = (
                area /
                hull_area
            )
        else:
            solidity = 0.0
        solidity = float(np.clip(solidity, 0.0, 1.0))

        # 圆形形状综合得分
        shape_score = (
            0.40 * circularity +
            0.30 * circle_fill_ratio +
            0.20 * aspect_ratio_score +
            0.10 * solidity
        )

        # 2. 【控制信号优化】计算用于视觉伺服的高质量状态信号
        M = cv2.moments(contour)
        if M["m00"] > 0:
            u = M["m10"] / M["m00"]
            v = M["m01"] / M["m00"]
        else:
            # 极小概率退化，使用外接圆中心兜底
            u = float(enc_x)
            v = float(enc_y)

        # 抛弃外接圆半径，使用面积反推等效半径，彻底消除 Z 轴深度的剧烈震荡
        equivalent_radius = np.sqrt(area / np.pi)

        shape_information = {
            "u": float(u),                          # 替换：平滑质心 u
            "v": float(v),                          # 替换：平滑质心 v
            "radius": float(equivalent_radius),     # 替换：平滑等效半径 r
            "area": float(area),
            "circularity": circularity,
            "circle_fill_ratio": circle_fill_ratio,
            "aspect_ratio_score": aspect_ratio_score,
            "solidity": solidity,
        }

        return (
            float(shape_score),
            shape_information
        )

    def calculate_color_score(
        self,
        hsv_image,
        contour
    ):
        """
        计算轮廓内部的绿色得分。

        颜色得分包括：

        1. green_ratio：
           轮廓内部满足 HSV 绿色范围的像素比例。

        2. soft_hue_score：
           轮廓内部像素色相与目标绿色中心的接近程度。

        这样即使部分绿色像素超出 HSV 范围，
        也不会立即让整个目标失效。

        输出：
            color_score:
                颜色得分，范围约为 0～1。

            color_information:
                颜色相关参数。
        """

        contour_mask = np.zeros(
            hsv_image.shape[:2],
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

        contour_pixel_count = (
            cv2.countNonZero(
                contour_mask
            )
        )

        if contour_pixel_count <= 0:
            return 0.0, None

        # 硬 HSV 范围，用于计算绿色像素比例
        green_mask = cv2.inRange(
            hsv_image,
            self.hsv_lower,
            self.hsv_upper
        )

        green_inside_contour = (
            cv2.bitwise_and(
                green_mask,
                contour_mask
            )
        )

        green_pixel_count = (
            cv2.countNonZero(
                green_inside_contour
            )
        )

        green_ratio = (
            green_pixel_count /
            contour_pixel_count
        )

        # 取出轮廓内部所有 HSV 像素
        contour_pixels = hsv_image[
            contour_mask > 0
        ].astype(np.float32)

        if contour_pixels.size == 0:
            return 0.0, None

        hue_values = contour_pixels[:, 0]
        saturation_values = (
            contour_pixels[:, 1] /
            255.0
        )

        value_values = (
            contour_pixels[:, 2] /
            255.0
        )

        # 目标绿色色相中心
        hue_lower = float(
            self.hsv_lower[0]
        )

        hue_upper = float(
            self.hsv_upper[0]
        )

        target_hue = (
            hue_lower +
            hue_upper
        ) / 2.0

        hue_tolerance = max(
            (
                hue_upper -
                hue_lower
            ) / 2.0,
            10.0
        )

        # OpenCV 的 H 为环形范围 0～179
        hue_difference = np.abs(
            hue_values -
            target_hue
        )

        hue_difference = np.minimum(
            hue_difference,
            180.0 -
            hue_difference
        )

        # 高斯形式的软色相接近程度
        hue_similarity = np.exp(
            -0.5 *
            (
                hue_difference /
                hue_tolerance
            ) ** 2
        )

        # 饱和度和亮度太低时，
        # 颜色本身不可靠，因此降低贡献。
        color_reliability = np.sqrt(
            saturation_values *
            value_values
        )

        soft_hue_score = np.mean(
            hue_similarity *
            color_reliability
        )

        # 绿色比例占主要部分，
        # 软色相距离用于允许一定颜色变化。
        color_score = (
            0.65 * green_ratio +
            0.35 * soft_hue_score
        )

        color_score = float(
            np.clip(
                color_score,
                0.0,
                1.0
            )
        )

        color_information = {
            "green_ratio":
                float(green_ratio),
            "soft_hue_score":
                float(soft_hue_score),
        }

        return (
            color_score,
            color_information
        )

    def extract_all_contours(
        self,
        image,
        hsv_image
    ):
        """
        从整张图像中提取所有可能的闭合轮廓。

        同时使用：

        1. 灰度图边缘；
        2. HSV 饱和度通道边缘。

        这样既可以检测亮度边缘，
        也可以检测颜色变化明显但亮度相近的边缘。
        """

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        gray_image = cv2.GaussianBlur(
            gray_image,
            (5, 5),
            0
        )

        saturation_image = hsv_image[
            :,
            :,
            1
        ]

        saturation_image = cv2.GaussianBlur(
            saturation_image,
            (5, 5),
            0
        )

        gray_edges = cv2.Canny(
            gray_image,
            self.canny_threshold_low,
            self.canny_threshold_high
        )

        saturation_edges = cv2.Canny(
            saturation_image,
            self.canny_threshold_low,
            self.canny_threshold_high
        )

        # 合并亮度边缘和颜色边缘
        combined_edges = cv2.bitwise_or(
            gray_edges,
            saturation_edges
        )

        kernel_size = max(
            self.morphology_kernel_size,
            3
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size
            )
        )

        # 闭运算连接小范围断裂边缘
        closed_edges = cv2.morphologyEx(
            combined_edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        # 轻微膨胀，使断裂边缘更容易闭合
        closed_edges = cv2.dilate(
            closed_edges,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            closed_edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def detect(
        self,
        image
    ):
        """
        检测图像中的绿色小球。

        输入：
            image:
                RGB 格式图像。

        输出：
            current_feature:
                检测成功时返回 [u, v, r, Z]；
                检测失败时返回 None。
        """

        self.last_candidates = []

        if image is None:
            return None

        if (
            image.ndim != 3 or
            image.shape[2] != 3
        ):
            return None

        image_height, image_width = (
            image.shape[:2]
        )

        image_area = float(
            image_height *
            image_width
        )

        hsv_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        )

        # 不再先进行绿色硬分割，
        # 而是从整幅图像检测所有轮廓。
        contours = self.extract_all_contours(
            image,
            hsv_image
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < self.min_area:
                continue

            if (
                area >
                image_area *
                self.max_area_ratio
            ):
                continue

            (
                shape_score,
                shape_information
            ) = self.calculate_shape_score(
                contour
            )

            if shape_information is None:
                continue

            radius = (
                shape_information["radius"]
            )

            if radius < self.min_radius:
                continue

            (
                color_score,
                color_information
            ) = self.calculate_color_score(
                hsv_image,
                contour
            )

            if color_information is None:
                continue

            # 形状和颜色加权
            total_score = (
                self.shape_weight *
                shape_score +
                self.color_weight *
                color_score
            )

            candidate = {
                "u":
                    shape_information["u"],

                "v":
                    shape_information["v"],

                "r":
                    radius,

                "area":
                    shape_information["area"],

                "shape_score":
                    float(shape_score),

                "color_score":
                    float(color_score),

                "total_score":
                    float(total_score),

                "circularity":
                    shape_information[
                        "circularity"
                    ],

                "circle_fill_ratio":
                    shape_information[
                        "circle_fill_ratio"
                    ],

                "aspect_ratio_score":
                    shape_information[
                        "aspect_ratio_score"
                    ],

                "solidity":
                    shape_information[
                        "solidity"
                    ],

                "green_ratio":
                    color_information[
                        "green_ratio"
                    ],

                "soft_hue_score":
                    color_information[
                        "soft_hue_score"
                    ],

                "contour":
                    contour,
            }

            candidates.append(
                candidate
            )

        # 按总得分从高到低排序
        candidates.sort(
            key=lambda item:
                item["total_score"],
            reverse=True
        )

        self.last_candidates = candidates

        if len(candidates) == 0:
            return None

        target = candidates[0]

        # 总得分不够，认为没有可靠目标
        if (
            target["total_score"] <
            self.min_total_score
        ):
            return None

        depth = self.estimate_depth(
            target["r"]
        )

        current_feature = np.array(
            [
                target["u"],
                target["v"],
                target["r"],
                depth
            ],
            dtype=float
        )

        return current_feature