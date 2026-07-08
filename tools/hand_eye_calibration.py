#!/usr/bin/env python3
"""
hand_eye_calibration_node.py

概括：
    创建ROS 2眼在手上手眼标定节点。
    节点打开USB摄像头，检测固定棋盘格，并自动从TF读取机器人末端位姿。
    用户每次按S保存一组同步数据；达到设定数量后，自动计算
    T_end_effector_camera，并保存为ROS 2 YAML文件。

功能：
    1. 打开USB摄像头并实时检测棋盘格。
    2. 根据相机内参、畸变参数和棋盘格尺寸，计算 T_camera_target。
    3. 从ROS 2 TF读取 T_base_end_effector。
    4. 按S保存一组手眼标定样本。
    5. 达到指定样本数后，使用Tsai-Lenz方法自动计算手眼矩阵。
    6. 验证固定棋盘格在机器人基座坐标系中的一致性。
    7. 输出并保存 T_end_effector_camera 到YAML文件。

接口：
    HandEyeCalibrationNode

输入：
    相机图像：
        USB摄像头实时图像。

    TF：
        T_base_end_effector = ^base T_end_effector。

    全局参数：
        相机内参、畸变参数、棋盘格尺寸、TF坐标系名称等。

输出：
    T_end_effector_camera:
        ^end_effector T_camera，即相机坐标系到末端坐标系的变换矩阵。

    hand_eye_result.yaml:
        可直接复制到cartesian_servo_node参数文件中的ROS 2 YAML结果。

操作：
    S：
        保存当前一组样本。

    C：
        使用当前已保存样本立即计算。

    Q：
        退出程序。

注意：
    1. 棋盘格必须固定不动，相机必须固定安装在机械臂末端。
    2. 按S时机械臂必须完全停止。
    3. CHECKERBOARD_SIZE填写的是棋盘格“内角点”数量，不是方格数量。
    4. SQUARE_SIZE_M单位必须是m。
    5. CAMERA_MATRIX和DIST_COEFFS必须来自同一分辨率下的相机内参标定。
"""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


# ============================================================
# 全局参数：主要修改这里
# ============================================================

# USB摄像头
CAMERA_INDEX =0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 30

# 相机内参
CAMERA_MATRIX = np.array(
    [
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64
)

# 相机畸变参数
# 常见顺序：[k1, k2, p1, p2, k3]
DIST_COEFFS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64
)

# 棋盘格内角点数量：
# (每行内角点数量, 每列内角点数量)
# 例如棋盘格有10×7个方格，则内角点通常为9×6
CHECKERBOARD_SIZE = (9, 6)

# 单个方格实际边长，单位m
# 例如25 mm应填写0.025
SQUARE_SIZE_M = 0.025

# 机器人TF坐标系名称
# 请使用以下命令检查实际名称：
# ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp
BASE_FRAME = "fr3_link0"
END_EFFECTOR_FRAME = "fr3_hand_tcp"

# 达到该样本数后自动计算
REQUIRED_SAMPLES = 15

# 过于相似的相邻姿态不保存
MIN_TRANSLATION_CHANGE_M = 0.015
MIN_ROTATION_CHANGE_DEG = 5.0

# OpenCV手眼标定方法
HAND_EYE_METHOD = cv2.CALIB_HAND_EYE_TSAI

# 输出结果
OUTPUT_YAML_PATH = "hand_eye_result.yaml"
OUTPUT_DATA_PATH = "hand_eye_samples.npz"

# 结果写入哪个ROS 2节点的参数区域
OUTPUT_PARAMETER_NODE = "cartesian_servo_node"


# ============================================================
# 以下代码通常不需要修改
# ============================================================


def quaternion_to_rotation_matrix(x, y, z, w):
    """
    将四元数[x, y, z, w]转换为3x3旋转矩阵。
    """

    quaternion = np.array(
        [x, y, z, w],
        dtype=np.float64
    )

    norm = np.linalg.norm(quaternion)

    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero.")

    x, y, z, w = quaternion / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64
    )


def transform_message_to_matrix(transform_stamped):
    """
    将geometry_msgs/TransformStamped转换为4x4矩阵。

    lookup_transform(BASE_FRAME, END_EFFECTOR_FRAME, ...)
    返回的是：
        T_base_end_effector = ^base T_end_effector
    """

    translation = transform_stamped.transform.translation
    rotation = transform_stamped.transform.rotation

    transform = np.eye(
        4,
        dtype=np.float64
    )

    transform[:3, :3] = quaternion_to_rotation_matrix(
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w
    )

    transform[:3, 3] = np.array(
        [
            translation.x,
            translation.y,
            translation.z,
        ],
        dtype=np.float64
    )

    return transform


def rotation_angle_degrees(rotation):
    """
    将旋转矩阵转换为旋转角，单位degree。
    """

    cosine_value = (
        np.trace(rotation) - 1.0
    ) / 2.0

    cosine_value = np.clip(
        cosine_value,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(cosine_value)
        )
    )


def normalize_rotation_matrix(rotation):
    """
    使用SVD将存在轻微数值误差的矩阵修正为旋转矩阵。
    """

    U, _, Vt = np.linalg.svd(rotation)
    normalized_rotation = U @ Vt

    if np.linalg.det(normalized_rotation) < 0.0:
        U[:, -1] *= -1.0
        normalized_rotation = U @ Vt

    return normalized_rotation


class HandEyeCalibrationNode(Node):
    """
    眼在手上手眼标定节点。
    """

    def __init__(self):
        super().__init__("hand_eye_calibration_node")

        self.check_global_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            IMAGE_WIDTH
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            IMAGE_HEIGHT
        )

        self.camera.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS
        )

        if not self.camera.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {CAMERA_INDEX}."
            )

        self.object_points = self.create_object_points()

        self.T_base_end_effector_list = []
        self.T_camera_target_list = []

        self.current_T_base_end_effector = None
        self.current_T_camera_target = None
        self.current_board_found = False
        self.calibration_finished = False

        self.timer = self.create_timer(
            1.0 / float(CAMERA_FPS),
            self.camera_callback
        )

        self.get_logger().info(
            "Hand-eye calibration node started."
        )

        self.get_logger().info(
            f"TF: {BASE_FRAME} <- {END_EFFECTOR_FRAME}"
        )

        self.get_logger().info(
            f"Checkerboard inner corners: "
            f"{CHECKERBOARD_SIZE}"
        )

        self.get_logger().info(
            f"Required samples: {REQUIRED_SAMPLES}"
        )

        self.get_logger().info(
            "Keep the board fixed. Stop the robot, "
            "then press S to save a sample."
        )

    @staticmethod
    def check_global_parameters():
        """
        检查全局参数。
        """

        if CAMERA_MATRIX.shape != (3, 3):
            raise ValueError(
                "CAMERA_MATRIX must be a 3x3 matrix."
            )

        if not np.all(np.isfinite(CAMERA_MATRIX)):
            raise ValueError(
                "CAMERA_MATRIX contains invalid values."
            )

        if DIST_COEFFS.size < 4:
            raise ValueError(
                "DIST_COEFFS must contain at least 4 values."
            )

        if (
            CHECKERBOARD_SIZE[0] < 2
            or CHECKERBOARD_SIZE[1] < 2
        ):
            raise ValueError(
                "CHECKERBOARD_SIZE is invalid."
            )

        if SQUARE_SIZE_M <= 0.0:
            raise ValueError(
                "SQUARE_SIZE_M must be positive."
            )

        if REQUIRED_SAMPLES < 3:
            raise ValueError(
                "REQUIRED_SAMPLES must be at least 3."
            )

    @staticmethod
    def create_object_points():
        """
        创建棋盘格角点在标定板坐标系中的三维坐标。

        标定板位于z=0平面。
        """

        columns, rows = CHECKERBOARD_SIZE

        object_points = np.zeros(
            (columns * rows, 3),
            dtype=np.float32
        )

        object_points[:, :2] = (
            np.mgrid[
                0:columns,
                0:rows
            ]
            .T
            .reshape(-1, 2)
        )

        object_points *= float(
            SQUARE_SIZE_M
        )

        return object_points

    def lookup_robot_pose(self):
        """
        从TF读取T_base_end_effector。
        """

        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                END_EFFECTOR_FRAME,
                Time()
            )

        except TransformException:
            return None

        return transform_message_to_matrix(
            transform
        )

    def detect_checkerboard_pose(self, image):
        """
        检测棋盘格并计算T_camera_target。

        输出：
            found:
                是否成功检测。

            T_camera_target:
                ^camera T_target。

            corners:
                亚像素棋盘格角点。
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        )

        if not found:
            return False, None, None

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                self.object_points,
                corners,
                CAMERA_MATRIX,
                DIST_COEFFS,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:
            return False, None, corners

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        T_camera_target = np.eye(
            4,
            dtype=np.float64
        )

        T_camera_target[:3, :3] = (
            rotation_matrix
        )

        T_camera_target[:3, 3] = (
            translation_vector.reshape(3)
        )

        cv2.drawChessboardCorners(
            image,
            CHECKERBOARD_SIZE,
            corners,
            found
        )

        cv2.drawFrameAxes(
            image,
            CAMERA_MATRIX,
            DIST_COEFFS,
            rotation_vector,
            translation_vector,
            3.0 * SQUARE_SIZE_M
        )

        return (
            True,
            T_camera_target,
            corners
        )

    def camera_callback(self):
        """
        读取图像、显示检测结果并处理键盘输入。
        """

        success, image = self.camera.read()

        if not success or image is None:
            self.get_logger().warning(
                "Failed to read camera image."
            )
            return

        (
            self.current_board_found,
            self.current_T_camera_target,
            _
        ) = self.detect_checkerboard_pose(
            image
        )

        self.current_T_base_end_effector = (
            self.lookup_robot_pose()
        )

        self.draw_status(image)

        cv2.imshow(
            "Hand-Eye Calibration",
            image
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("s"), ord("S")):
            self.save_current_sample()

        elif key in (ord("c"), ord("C")):
            self.compute_calibration()

        elif key in (ord("q"), ord("Q")):
            self.get_logger().info(
                "Quit requested."
            )
            rclpy.shutdown()

    def draw_status(self, image):
        """
        在图像上显示标定状态。
        """

        board_status = (
            "FOUND"
            if self.current_board_found
            else "NOT FOUND"
        )

        tf_status = (
            "READY"
            if self.current_T_base_end_effector is not None
            else "NOT READY"
        )

        lines = [
            f"Board: {board_status}",
            f"TF: {tf_status}",
            (
                f"Samples: "
                f"{len(self.T_base_end_effector_list)}"
                f"/{REQUIRED_SAMPLES}"
            ),
            "S: save | C: compute | Q: quit",
        ]

        for index, text in enumerate(lines):
            cv2.putText(
                image,
                text,
                (20, 30 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    def is_pose_too_similar(self, new_pose):
        """
        判断新机器人姿态是否与上一组过于接近。
        """

        if not self.T_base_end_effector_list:
            return False

        previous_pose = (
            self.T_base_end_effector_list[-1]
        )

        translation_change = np.linalg.norm(
            new_pose[:3, 3]
            - previous_pose[:3, 3]
        )

        relative_rotation = (
            previous_pose[:3, :3].T
            @ new_pose[:3, :3]
        )

        rotation_change = rotation_angle_degrees(
            relative_rotation
        )

        return (
            translation_change
            < MIN_TRANSLATION_CHANGE_M
            and rotation_change
            < MIN_ROTATION_CHANGE_DEG
        )

    def save_current_sample(self):
        """
        保存当前一组机器人位姿和标定板位姿。
        """

        if self.calibration_finished:
            self.get_logger().warning(
                "Calibration has already finished."
            )
            return

        if not self.current_board_found:
            self.get_logger().warning(
                "Checkerboard is not detected."
            )
            return

        if self.current_T_camera_target is None:
            self.get_logger().warning(
                "Camera target pose is unavailable."
            )
            return

        if self.current_T_base_end_effector is None:
            self.get_logger().warning(
                "Robot TF is unavailable."
            )
            return

        if self.is_pose_too_similar(
            self.current_T_base_end_effector
        ):
            self.get_logger().warning(
                "Current robot pose is too similar "
                "to the previous sample."
            )
            return

        self.T_base_end_effector_list.append(
            self.current_T_base_end_effector.copy()
        )

        self.T_camera_target_list.append(
            self.current_T_camera_target.copy()
        )

        number_samples = len(
            self.T_base_end_effector_list
        )

        self.get_logger().info(
            f"Saved sample {number_samples}/"
            f"{REQUIRED_SAMPLES}."
        )

        self.save_raw_samples()

        if number_samples >= REQUIRED_SAMPLES:
            self.compute_calibration()

    def save_raw_samples(self):
        """
        保存原始样本，便于之后重新计算。
        """

        np.savez(
            OUTPUT_DATA_PATH,
            T_base_end_effector=np.asarray(
                self.T_base_end_effector_list,
                dtype=np.float64
            ),
            T_camera_target=np.asarray(
                self.T_camera_target_list,
                dtype=np.float64
            )
        )

    def compute_calibration(self):
        """
        计算T_end_effector_camera。
        """

        number_samples = len(
            self.T_base_end_effector_list
        )

        if number_samples < 3:
            self.get_logger().warning(
                "At least 3 samples are required."
            )
            return

        R_end_effector_to_base = []
        t_end_effector_to_base = []
        R_target_to_camera = []
        t_target_to_camera = []

        for (
            T_base_end_effector,
            T_camera_target
        ) in zip(
            self.T_base_end_effector_list,
            self.T_camera_target_list
        ):
            R_end_effector_to_base.append(
                T_base_end_effector[:3, :3]
            )

            t_end_effector_to_base.append(
                T_base_end_effector[
                    :3,
                    3
                ].reshape(3, 1)
            )

            R_target_to_camera.append(
                T_camera_target[:3, :3]
            )

            t_target_to_camera.append(
                T_camera_target[
                    :3,
                    3
                ].reshape(3, 1)
            )

        (
            R_camera_to_end_effector,
            t_camera_to_end_effector
        ) = cv2.calibrateHandEye(
            R_end_effector_to_base,
            t_end_effector_to_base,
            R_target_to_camera,
            t_target_to_camera,
            method=HAND_EYE_METHOD
        )

        if (
            R_camera_to_end_effector is None
            or t_camera_to_end_effector is None
        ):
            self.get_logger().error(
                "Hand-eye calibration failed."
            )
            return

        T_end_effector_camera = np.eye(
            4,
            dtype=np.float64
        )

        T_end_effector_camera[:3, :3] = (
            normalize_rotation_matrix(
                R_camera_to_end_effector
            )
        )

        T_end_effector_camera[:3, 3] = (
            np.asarray(
                t_camera_to_end_effector,
                dtype=np.float64
            ).reshape(3)
        )

        if not np.all(
            np.isfinite(
                T_end_effector_camera
            )
        ):
            self.get_logger().error(
                "Calibration result contains invalid values."
            )
            return

        self.print_result(
            T_end_effector_camera
        )

        self.validate_result(
            T_end_effector_camera
        )

        self.save_yaml(
            T_end_effector_camera
        )

        self.calibration_finished = True

    def validate_result(
        self,
        T_end_effector_camera
    ):
        """
        通过固定标定板的一致性验证结果。

        对每一组计算：
            T_base_target
            =
            T_base_end_effector
            @ T_end_effector_camera
            @ T_camera_target
        """

        base_target_poses = []

        for (
            T_base_end_effector,
            T_camera_target
        ) in zip(
            self.T_base_end_effector_list,
            self.T_camera_target_list
        ):
            base_target_poses.append(
                T_base_end_effector
                @ T_end_effector_camera
                @ T_camera_target
            )

        translations = np.asarray(
            [
                pose[:3, 3]
                for pose in base_target_poses
            ],
            dtype=np.float64
        )

        mean_translation = np.mean(
            translations,
            axis=0
        )

        translation_errors = np.linalg.norm(
            translations - mean_translation,
            axis=1
        )

        translation_rms = np.sqrt(
            np.mean(
                translation_errors ** 2
            )
        )

        rotations = [
            pose[:3, :3]
            for pose in base_target_poses
        ]

        mean_rotation = normalize_rotation_matrix(
            np.sum(
                rotations,
                axis=0
            )
        )

        rotation_errors = np.asarray(
            [
                rotation_angle_degrees(
                    mean_rotation.T @ rotation
                )
                for rotation in rotations
            ],
            dtype=np.float64
        )

        rotation_rms = np.sqrt(
            np.mean(
                rotation_errors ** 2
            )
        )

        self.get_logger().info(
            "Validation result: "
            f"translation RMS = "
            f"{translation_rms * 1000.0:.3f} mm, "
            f"rotation RMS = "
            f"{rotation_rms:.3f} deg."
        )

    def print_result(
        self,
        T_end_effector_camera
    ):
        """
        打印手眼标定结果。
        """

        np.set_printoptions(
            precision=10,
            suppress=True
        )

        print()
        print("=" * 70)
        print(
            "T_end_effector_camera = "
            "^end_effector T_camera"
        )
        print("=" * 70)
        print(T_end_effector_camera)
        print()

        print(
            "YAML flat list:"
        )

        print(
            T_end_effector_camera
            .reshape(-1)
            .tolist()
        )

    @staticmethod
    def create_yaml_text(
        T_end_effector_camera
    ):
        """
        创建ROS 2 YAML文本。
        """

        flat_values = (
            T_end_effector_camera
            .reshape(-1)
        )

        value_text = ", ".join(
            f"{value:.10f}"
            for value in flat_values
        )

        return (
            f"{OUTPUT_PARAMETER_NODE}:\n"
            f"  ros__parameters:\n"
            f"    T_end_effector_camera: "
            f"[{value_text}]\n"
        )

    def save_yaml(
        self,
        T_end_effector_camera
    ):
        """
        保存标定结果到YAML。
        """

        yaml_text = self.create_yaml_text(
            T_end_effector_camera
        )

        output_path = Path(
            OUTPUT_YAML_PATH
        )

        output_path.write_text(
            yaml_text,
            encoding="utf-8"
        )

        self.get_logger().info(
            "Hand-eye result saved to: "
            f"{output_path.resolve()}"
        )

    def close(self):
        """
        释放摄像头和OpenCV窗口。
        """

        if self.camera is not None:
            self.camera.release()

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    node = HandEyeCalibrationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            "Keyboard interrupt."
        )

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()