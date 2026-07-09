#!/usr/bin/env python3
"""
calibrate_usb_camera.py

功能：
    使用棋盘格标定/dev/video3对应的USB摄像头，
    计算fx、fy、cx、cy以及镜头畸变参数。

操作：
    s：保存当前有效棋盘格角点
    c：完成采集并开始标定
    q：退出

    弹出相机窗口后：
        把棋盘格放到相机前；
        检测成功时会显示角点连线；
        按 s 保存当前姿态；
        改变棋盘格位置和角度，再按 s；
        保存至少 15～20 组；
        按 c 开始计算；
        按 q 直接退出。

        
    RMS重投影误差:
        小于 0.5 pixel：比较好
        0.5～1.0 pixel：一般可以使用
        大于 1.0 pixel：建议重新标定


运行：    
    python3 calibrate_usb_camera.py
"""

import cv2
import numpy as np


# USB摄像头编号
CAMERA_INDEX = 3

# 必须与实际运行视觉伺服时的分辨率一致
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 棋盘格内部角点数量：横向9个，纵向6个，不是方格数量
CHECKERBOARD = (9, 6)

# 单个方格实际边长，单位m
SQUARE_SIZE = 0.0243

# 建议至少采集15组
MIN_IMAGES = 15


def main():
    # 亚像素角点优化停止条件
    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    # 构造棋盘格三维坐标
    # 例如：(0,0,0)、(0.025,0,0)、(0.050,0,0)……
    object_point = np.zeros(
        (
            CHECKERBOARD[0] * CHECKERBOARD[1],
            3
        ),
        dtype=np.float32
    )

    object_point[:, :2] = (
        np.mgrid[
            0:CHECKERBOARD[0],
            0:CHECKERBOARD[1]
        ]
        .T
        .reshape(-1, 2)
        * SQUARE_SIZE
    )

    # 保存每次采集的三维点和二维角点
    object_points = []
    image_points = []

    capture = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开/dev/video{CAMERA_INDEX}"
        )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        IMAGE_WIDTH
    )

    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        IMAGE_HEIGHT
    )

    image_size = None
    latest_corners = None
    latest_found = False

    print("操作说明：")
    print("  s：保存当前棋盘格")
    print("  c：开始标定")
    print("  q：退出")

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("图像读取失败。")
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            image_size = (
                gray.shape[1],
                gray.shape[0]
            )

            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
            )

            latest_found = found
            latest_corners = None

            if found:
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )

                latest_corners = corners_refined

                cv2.drawChessboardCorners(
                    frame,
                    CHECKERBOARD,
                    corners_refined,
                    found
                )

                status = "Chessboard detected"
                status_color = (0, 255, 0)

            else:
                status = "Chessboard not detected"
                status_color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )

            cv2.putText(
                frame,
                f"Saved: {len(image_points)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                "s: save  c: calibrate  q: quit",
                (10, IMAGE_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "USB Camera Calibration",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if (
                    latest_found
                    and latest_corners is not None
                ):
                    object_points.append(
                        object_point.copy()
                    )

                    image_points.append(
                        latest_corners.copy()
                    )

                    print(
                        "已保存：",
                        len(image_points)
                    )
                else:
                    print(
                        "当前没有检测到完整棋盘格。"
                    )

            elif key == ord("c"):
                if len(image_points) < MIN_IMAGES:
                    print(
                        f"有效图像不足，当前"
                        f"{len(image_points)}组，"
                        f"建议至少{MIN_IMAGES}组。"
                    )
                else:
                    break

            elif key == ord("q"):
                return

    finally:
        capture.release()
        cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError(
            "没有获得有效图像尺寸。"
        )

    # 执行相机标定
    (
        rms_error,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    print("\n========== 标定结果 ==========")

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(distortion_coefficients)

    print(
        f"\nRMS重投影误差：{rms_error:.6f} pixel"
    )

    print("\n写入YAML的参数：")
    print(f"fx: {fx:.6f}")
    print(f"fy: {fy:.6f}")
    print(f"cx: {cx:.6f}")
    print(f"cy: {cy:.6f}")

    # 保存完整标定结果
    np.savez(
        "usb_camera_calibration.npz",
        camera_matrix=camera_matrix,
        distortion_coefficients=(
            distortion_coefficients
        ),
        image_width=image_size[0],
        image_height=image_size[1],
        rms_error=rms_error
    )

    print(
        "\n完整结果已保存为："
        "usb_camera_calibration.npz"
    )


if __name__ == "__main__":
    main()