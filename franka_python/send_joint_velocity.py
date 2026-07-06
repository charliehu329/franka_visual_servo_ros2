#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class FrankaVelocityPublisher(Node):
    def __init__(self):
        super().__init__("franka_velocity_publisher")

        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/velocity_command_node/target_velocities",
            10
        )
        

        self.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 初始化为 7 个关节的速度为 0
        self.velocities[6] = -0.02   # 第 7 个关节，单位 rad/s

        self.start_time = time.time()
        self.motion_time = 3.0      # 运动 3 秒

        self.finished = False
        self.stop_count = 0
        self.stop_count_max = 20    # 停止命令多发几次

        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz

    def publish_velocity(self, data):
        msg = Float64MultiArray()
        msg.data = data
        self.publisher.publish(msg)

    def timer_callback(self):
        elapsed = time.time() - self.start_time

        if elapsed < self.motion_time:
            self.publish_velocity(self.velocities)
            self.get_logger().info(f"Publishing: {self.velocities}")

        elif self.stop_count < self.stop_count_max:
            self.publish_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.stop_count += 1
            self.get_logger().info("Sending stop command.")

        else:
            self.get_logger().info("Motion finished. Exiting node.")
            self.finished = True


def main(args=None):
    rclpy.init(args=args)

    node = FrankaVelocityPublisher()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt received.")

    finally:
        # 退出前再发几次 0 速度，确保机器人停止
        for _ in range(10):
            node.publish_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            time.sleep(0.02)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()