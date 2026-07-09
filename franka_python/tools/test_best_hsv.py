#!/usr/bin/env python3
"""
test_green_ball_detection.py

功能：
    1. 测试 USB 相机和 GreenBallDetector 是否能够识别绿色小球。
    2. 使用鼠标点击图像中的像素，读取并记录该像素的 HSV 值。
    3. 根据多次点击的 HSV 样本，计算推荐的 HSV 检测范围。

接口：
    鼠标左键：
        记录当前点击位置的 HSV 值。

    键盘 s：
        在终端输出当前所有采样值和推荐 HSV 范围。

    键盘 c：
        清空所有 HSV 采样值。

    键盘 u：
        撤销最后一次 HSV 采样。

    键盘 q：
        输出最终 HSV 汇总结果并退出。

输出：
    1. 检测成功时，在图像上绘制球心和外接圆。
    2. 在终端打印检测结果 [u, v, r, Z]。
    3. 每次鼠标点击时，打印像素位置以及 HSV 值。
    4. 退出时输出推荐的 hsv_lower 和 hsv_upper。

说明：
    OpenCV 中 HSV 的取值范围为：
        H: 0～179
        S: 0～255
        V: 0～255

    当前 USB 摄像头使用 /dev/video3。
"""

import cv2
import numpy as np


import sys
from pathlib import Path

# Current file:
# ~/franka_ros2_ws/src/franka_python/franka_python/tools/test_best_hsv.py
#
# Move up to:
# ~/franka_ros2_ws/src/franka_python
package_root = Path(__file__).resolve().parents[2]

# Add the package root to Python's module search path
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))




from franka_python.vision.camera import USBCamera
from franka_python.vision.green_ball_detector import GreenBallDetector


class HSVSampler:
    """
    功能：
        保存鼠标点击位置的 HSV 样本，并计算推荐 HSV 范围。

    输入：
        当前摄像头图像对应的 HSV 图像。

    输出：
        HSV 样本和推荐的 HSV 上下限。
    """

    def __init__(self):
        self.current_hsv_image = None

        # 保存每次点击得到的 [H, S, V]
        self.samples = []

        # 保存点击位置，用于在图像中绘制标记
        self.points = []

    def update_image(self, hsv_image):
        """
        更新鼠标采样所使用的当前 HSV 图像。
        """
        self.current_hsv_image = hsv_image

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):
        """
        OpenCV 鼠标回调函数。

        左键：
            添加一个 HSV 样本。

        右键：
            撤销最后一个 HSV 样本。
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_sample(x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo_last_sample()

    def add_sample(self, x, y):
        """
        记录指定像素位置的 HSV 值。
        """
        if self.current_hsv_image is None:
            print("当前还没有可用的 HSV 图像。")
            return

        image_height, image_width = (
            self.current_hsv_image.shape[:2]
        )

        if not (
            0 <= x < image_width
            and 0 <= y < image_height
        ):
            print(
                f"点击位置超出图像范围："
                f"x={x}, y={y}"
            )
            return

        h, s, v = self.current_hsv_image[y, x]

        sample = [
            int(h),
            int(s),
            int(v)
        ]

        self.samples.append(sample)
        self.points.append((x, y))

        sample_index = len(self.samples)

        print(
            f"HSV sample {sample_index}: "
            f"pixel=({x}, {y}), "
            f"H={sample[0]}, "
            f"S={sample[1]}, "
            f"V={sample[2]}"
        )

    def undo_last_sample(self):
        """
        删除最后一个 HSV 样本。
        """
        if not self.samples:
            print("当前没有可以撤销的 HSV 样本。")
            return

        removed_sample = self.samples.pop()
        removed_point = self.points.pop()

        print(
            "已撤销最后一个样本："
            f"pixel={removed_point}, "
            f"HSV={removed_sample}"
        )

    def clear_samples(self):
        """
        清空全部 HSV 样本。
        """
        self.samples.clear()
        self.points.clear()

        print("已经清空全部 HSV 采样值。")

    def calculate_recommended_range(self):
        """
        根据已采集的 HSV 样本计算推荐范围。

        方法：
            样本数不少于 5 时，使用 5%～95% 分位数，
            减小个别误点击或异常像素的影响。

            样本数少于 5 时，直接使用样本最小值和最大值。

            最后增加一定余量：
                H 增加 ±5
                S 增加 ±25
                V 增加 ±25

        输出：
            hsv_lower:
                推荐 HSV 下限。

            hsv_upper:
                推荐 HSV 上限。
        """
        if not self.samples:
            return None, None

        samples_array = np.asarray(
            self.samples,
            dtype=np.float32
        )

        if len(self.samples) >= 5:
            range_lower = np.percentile(
                samples_array,
                5,
                axis=0
            )

            range_upper = np.percentile(
                samples_array,
                95,
                axis=0
            )

        else:
            range_lower = np.min(
                samples_array,
                axis=0
            )

            range_upper = np.max(
                samples_array,
                axis=0
            )

        # 为光照变化、阴影和边缘区域增加余量
        margin = np.array(
            [5.0, 25.0, 25.0],
            dtype=np.float32
        )

        hsv_lower = np.floor(
            range_lower - margin
        ).astype(np.int32)

        hsv_upper = np.ceil(
            range_upper + margin
        ).astype(np.int32)

        # OpenCV HSV 合法范围
        hsv_lower[0] = np.clip(
            hsv_lower[0],
            0,
            179
        )
        hsv_upper[0] = np.clip(
            hsv_upper[0],
            0,
            179
        )

        hsv_lower[1:] = np.clip(
            hsv_lower[1:],
            0,
            255
        )
        hsv_upper[1:] = np.clip(
            hsv_upper[1:],
            0,
            255
        )

        return (
            hsv_lower.tolist(),
            hsv_upper.tolist()
        )

    def print_summary(self):
        """
        输出所有 HSV 样本及推荐 HSV 范围。
        """
        print("\n")
        print("=" * 60)
        print("HSV 采样结果汇总")
        print("=" * 60)

        if not self.samples:
            print("当前没有任何 HSV 采样值。")
            print("=" * 60)
            return

        print(f"采样点数量：{len(self.samples)}")
        print()

        print("所有采样值：")

        for index, sample in enumerate(
            self.samples,
            start=1
        ):
            h, s, v = sample
            x, y = self.points[index - 1]

            print(
                f"  {index:03d}: "
                f"pixel=({x:3d}, {y:3d}), "
                f"HSV=[{h:3d}, {s:3d}, {v:3d}]"
            )

        samples_array = np.asarray(
            self.samples,
            dtype=np.int32
        )

        raw_minimum = np.min(
            samples_array,
            axis=0
        )

        raw_maximum = np.max(
            samples_array,
            axis=0
        )

        average = np.mean(
            samples_array,
            axis=0
        )

        hsv_lower, hsv_upper = (
            self.calculate_recommended_range()
        )

        print()
        print(
            "样本最小值："
            f"[{raw_minimum[0]}, "
            f"{raw_minimum[1]}, "
            f"{raw_minimum[2]}]"
        )

        print(
            "样本最大值："
            f"[{raw_maximum[0]}, "
            f"{raw_maximum[1]}, "
            f"{raw_maximum[2]}]"
        )

        print(
            "样本平均值："
            f"[{average[0]:.1f}, "
            f"{average[1]:.1f}, "
            f"{average[2]:.1f}]"
        )

        print()
        print("推荐 HSV 设置：")
        print(f"hsv_lower: {hsv_lower}")
        print(f"hsv_upper: {hsv_upper}")

        print()
        print("可以复制到 YAML 文件中：")
        print(f"    hsv_lower: {hsv_lower}")
        print(f"    hsv_upper: {hsv_upper}")

        print()
        print("可以复制到 Python 代码中：")
        print(
            "    hsv_lower=np.array("
            f"{hsv_lower}, dtype=np.uint8)"
        )
        print(
            "    hsv_upper=np.array("
            f"{hsv_upper}, dtype=np.uint8)"
        )

        print("=" * 60)
        print()


def draw_sample_points(
    image_bgr,
    points
):
    """
    在图像中绘制已经点击的采样位置。
    """
    for index, point in enumerate(
        points,
        start=1
    ):
        x, y = point

        # 黄色圆点
        cv2.circle(
            image_bgr,
            (x, y),
            4,
            (0, 255, 255),
            -1
        )

        # 显示样本编号
        cv2.putText(
            image_bgr,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1
        )


def main():
    camera = USBCamera(
        camera_index=2,
        width=640,
        height=480,
        fps=30
    )

    detector = GreenBallDetector(
        fx=600.0,
        fy=600.0,
        sphere_radius=0.02,
        initial_depth=0.5
    )

    hsv_sampler = HSVSampler()

    window_name = (
        "Green Ball Detection and HSV Sampling"
    )

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        hsv_sampler.mouse_callback
    )

    print("=" * 60)
    print("绿色小球 HSV 采样工具")
    print("=" * 60)
    print("鼠标左键：记录点击位置的 HSV 值")
    print("鼠标右键：撤销最后一个 HSV 样本")
    print("键盘 s：输出当前 HSV 汇总")
    print("键盘 c：清空全部 HSV 样本")
    print("键盘 u：撤销最后一个 HSV 样本")
    print("键盘 q：输出最终结果并退出")
    print("=" * 60)

    try:
        while camera.is_opened():
            image_rgb = camera.read()

            if image_rgb is None:
                print("摄像头图像读取失败。")
                break

            # 将当前图像转换为 HSV
            image_hsv = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2HSV
            )

            # 更新鼠标点击时所读取的 HSV 图像
            hsv_sampler.update_image(
                image_hsv
            )

            current_feature = detector.detect(
                image_rgb
            )

            # OpenCV 显示使用 BGR 格式
            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR
            )

            if current_feature is not None:
                u, v, radius, depth = (
                    current_feature
                )

                center = (
                    int(round(u)),
                    int(round(v))
                )

                radius_int = int(
                    round(radius)
                )

                # 绘制检测到的小球外接圆
                cv2.circle(
                    image_bgr,
                    center,
                    radius_int,
                    (0, 255, 0),
                    2
                )

                # 绘制检测到的小球球心
                cv2.circle(
                    image_bgr,
                    center,
                    4,
                    (0, 0, 255),
                    -1
                )

                detection_text = (
                    f"u={u:.1f}, "
                    f"v={v:.1f}, "
                    f"r={radius:.1f}, "
                    f"Z={depth:.3f} m"
                )

                cv2.putText(
                    image_bgr,
                    detection_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    image_bgr,
                    "Not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # 绘制所有点击过的位置
            draw_sample_points(
                image_bgr,
                hsv_sampler.points
            )

            sample_count_text = (
                f"HSV samples: "
                f"{len(hsv_sampler.samples)}"
            )

            cv2.putText(
                image_bgr,
                sample_count_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                image_bgr,
                "Left click: sample HSV",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                image_bgr,
                "s: summary  c: clear  u: undo  q: quit",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                window_name,
                image_bgr
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                hsv_sampler.print_summary()

            elif key == ord("c"):
                hsv_sampler.clear_samples()

            elif key == ord("u"):
                hsv_sampler.undo_last_sample()

    finally:
        # 退出前自动输出最终 HSV 推荐值
        hsv_sampler.print_summary()

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()