#!/usr/bin/env python3
"""
test_green_ball_detection.py

功能：
    测试USB相机和GreenBallDetector是否能够正确识别绿色小球。

输出：
    1. 检测成功时，在图像上绘制球心和外接圆。
    2. 在终端打印 [u, v, r, Z]。Z是小球球心在相机坐标系中的光轴深度近似值
    3. 检测失败时，在图像上显示 "Not detected"。

说明：
    当前USB摄像头使用 /dev/video3。
    按 q 键退出测试。
"""

import cv2

from franka_python.vision.camera import USBCamera
from franka_python.vision.green_ball_detector import GreenBallDetector


def main():
    camera = USBCamera(
        camera_index=3,
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

    try:
        while camera.is_opened():

            image_rgb = camera.read()

            if image_rgb is None:
                print("摄像头图像读取失败。")
                break

            current_feature = detector.detect(
                image_rgb
            )

            # OpenCV显示使用BGR格式
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

                # 绘制小球外接圆
                cv2.circle(
                    image_bgr,
                    center,
                    radius_int,
                    (0, 255, 0),
                    2
                )

                # 绘制球心
                cv2.circle(
                    image_bgr,
                    center,
                    4,
                    (0, 0, 255),
                    -1
                )

                text = (
                    f"u={u:.1f}, "
                    f"v={v:.1f}, "
                    f"r={radius:.1f}, "
                    f"Z={depth:.3f} m"
                )

                cv2.putText(
                    image_bgr,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                print(
                    "Detected:",
                    current_feature
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

            cv2.imshow(
                "Green Ball Detection Test",
                image_bgr
            )

            if (
                cv2.waitKey(1) &
                0xFF
            ) == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()