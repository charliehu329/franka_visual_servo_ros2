

# Franka Visual Servo ROS2

A ROS 2 based control framework for Franka FR3 robot, focusing on Cartesian velocity control, Jacobian-based velocity mapping, and a modular architecture for future visual servoing applications.

Author: Charlie Hu

## Overview

This project provides a modular ROS 2 control framework for the Franka FR3 robot.

The goal is to build a flexible foundation for image-based visual servoing (IBVS) research, where visual information can be converted into robot motion commands through a closed-loop control pipeline.

Current implemented functions:

- ROS 2 integration with Franka robot
- Cartesian velocity control
- Joint velocity control through ROS 2 topics
- Robot Jacobian based velocity mapping
- Modular controller architecture

The planned visual servo pipeline is:

```text
Camera
  ↓
Target Detection / Feature Extraction
  ↓
Visual Servo Control Law
  ↓
Cartesian Velocity Command
  ↓
Jacobian Velocity Mapping
  ↓
Joint Velocity Command
  ↓
Franka FR3 Robot
```

## System Configuration

### Hardware

- Robot: Franka Research 3 (FR3)
- Robot System Version: 5.8.2

### Software

- OS: Ubuntu 24.04
- ROS 2 Distribution: Jazzy
- libfranka: 0.17.0
- franka_ros2: v3.0.0
- Programming Language: Python

## Features

### Robot Control

- Cartesian velocity command interface
- Joint velocity command interface
- ROS 2 topic based communication
- Real robot velocity control

### Robot Kinematics

- Forward kinematics interface
- Jacobian calculation
- Cartesian velocity to joint velocity mapping

### Visual Servo Framework

The project provides modular components for future integration of:

- Camera interface
- Target detection
- Image feature extraction
- IBVS controller
- Stereo vision module

## Project Structure

```text
franka_visual_servo_ros2/

├── franka_python/
│
│   ├── cartesian_servo_node.py
│   │       # ROS 2 node for Cartesian velocity control
│   │
│   ├── robot_kinematics.py
│   │       # Robot kinematics and Jacobian calculation
│   │
│   ├── velocity_mapper.py
│   │       # Cartesian velocity to joint velocity mapping
│   │
│   ├── visual_servo_law.py
│   │       # Visual servo control law module
│   │
│   └── safety.py
│           # Velocity limitation and safety checking
│
├── config/
│       # Configuration files
│
├── launch/
│       # ROS 2 launch files
│
├── resource/
│       # ROS 2 package resources
│
├── package.xml
│       # ROS 2 package description
│
└── setup.py
        # Python package setup
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/charliehu329/franka_visual_servo_ros2.git
cd franka_visual_servo_ros2
```

### 2. Build ROS 2 Package

Source ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
```

Build package:

```bash
colcon build --symlink-install
```

For development, build only this package:

```bash
colcon build --packages-select franka_python --symlink-install
```

Source workspace:

```bash
source install/setup.bash
```

Check ROS 2 packages:

```bash
ros2 pkg list | grep franka
```

## Usage

### 1. Connect Franka Robot

Connect the computer and Franka controller through Ethernet.

Check robot connection:

```bash
ping 172.16.0.2
```

Open Franka Desk:

```text
https://172.16.0.2
```

Before running ROS 2 control:

1. Power on the robot
2. Unlock the robot
3. Activate FCI mode in Franka Desk

### 2. Start Franka ROS 2 Controller

Open Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/setup.bash
```

Launch controller:

```bash
ros2 launch franka_velocity_ctrl fr3_velocity.launch.py \
robot_ip:=172.16.0.2 \
mode:=topic
```

Check controller status:

```bash
ros2 control list_controllers
```

### 3. Send Velocity Commands

Open Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/setup.bash
```

Run velocity command node:

```bash
ros2 run franka_python send_joint_velocity
```

Publish velocity command directly:

```bash
ros2 topic pub /velocity_command_node/target_velocities \
std_msgs/msg/Float64MultiArray \
"data: [0.0,0.0,0.0,0.0,0.0,0.0,0.0]"
```

Example: move joint 7:

```bash
ros2 topic pub /velocity_command_node/target_velocities \
std_msgs/msg/Float64MultiArray \
"data: [0.0,0.0,0.0,0.0,0.0,0.0,-0.02]"
```

## Control Architecture

The framework is divided into several independent modules.

### Visual Servo Law

Input:

```text
Image feature error
```

Output:

```text
Desired Cartesian velocity
```

### Velocity Mapping

Cartesian velocity is converted into joint velocity using robot Jacobian:

```text
Cartesian Velocity
        ↓
Jacobian Mapping
        ↓
Joint Velocity
```

### Safety Module

Responsible for:

- Velocity limitation
- Joint constraints
- Robot safety protection

## Development Status

### Completed

- ROS 2 package framework
- Franka FR3 velocity control
- Cartesian velocity interface
- Joint velocity command interface
- Jacobian based velocity mapping
- Modular control architecture

### Future Work

- Camera interface
- Target detection
- Image feature extraction
- Closed-loop IBVS controller
- Stereo vision based visual servoing
- Real-time experiments

## Troubleshooting

### Robot cannot be controlled

Check:

```bash
ping 172.16.0.2
```

Make sure:

- Robot is powered on
- Ethernet connection is correct
- FCI mode is activated

### ROS 2 package cannot be found

Reload workspace:

```bash
source install/setup.bash
```

Check package:

```bash
ros2 pkg list | grep franka
```

### Controller is not active

Check:

```bash
ros2 control list_controllers
```

## Acknowledgements

This project is developed based on the open-source Franka ROS 2 ecosystem.

The original robot interface and low-level communication are provided by:

- Franka Robotics
- libfranka
- franka_ros2

This repository extends the existing framework with additional modules for:

- Cartesian velocity control
- Joint velocity control interface
- Jacobian-based velocity mapping
- Visual servo control architecture

Please refer to the original repositories for the official robot interface implementation and corresponding licenses.

## License

MIT License

# Franka Visual Servo ROS2 中文说明

## 项目简介

本项目基于 ROS 2 构建 Franka FR3 机械臂视觉伺服控制框架。

项目目标是在已有 Franka ROS 2 控制生态基础上，建立一个模块化的机器人控制框架，为后续图像视觉伺服（IBVS）、目标检测以及双目视觉控制提供基础。

当前已经实现：

- Franka FR3 ROS 2 控制接口
- 笛卡尔速度控制
- 关节速度控制
- 基于机器人雅可比矩阵的速度映射
- 模块化视觉伺服控制框架

## 系统配置

### 硬件

- 机器人：Franka Research 3 (FR3)
- Robot System Version：5.8.2

### 软件

- Ubuntu 24.04
- ROS 2 Jazzy
- libfranka 0.17.0
- franka_ros2 v3.0.0
- Python

## 项目结构

```text
franka_visual_servo_ros2/

├── franka_python/
│   ├── cartesian_servo_node.py
│   │       # 笛卡尔速度控制 ROS 2 节点
│   ├── robot_kinematics.py
│   │       # 机器人运动学和雅可比计算
│   ├── velocity_mapper.py
│   │       # 笛卡尔速度到关节速度映射
│   ├── visual_servo_law.py
│   │       # 视觉伺服控制律
│   └── safety.py
│           # 安全限制和速度约束
```

## 安装方法

下载项目：

```bash
git clone https://github.com/charliehu329/franka_visual_servo_ros2.git
cd franka_visual_servo_ros2
```

编译：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 使用方法

### 1. 连接机器人

通过 Ethernet 连接电脑和 Franka 控制柜。

检查网络：

```bash
ping 172.16.0.2
```

打开 Franka Desk：

```text
https://172.16.0.2
```

运行控制前：

1. 打开机器人电源
2. 解锁机器人
3. 在 Franka Desk 中开启 FCI 模式

### 2. 启动 ROS 2 控制器

终端 1：

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/setup.bash

ros2 launch franka_velocity_ctrl fr3_velocity.launch.py \
robot_ip:=172.16.0.2 \
mode:=topic
```

查看控制器状态：

```bash
ros2 control list_controllers
```

### 3. 发布速度指令

终端 2：

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/setup.bash
```

运行速度控制节点：

```bash
ros2 run franka_python send_joint_velocity
```

## 控制框架

整体控制流程：

```text
相机
 ↓
目标检测 / 图像特征提取
 ↓
视觉伺服控制律
 ↓
末端笛卡尔速度
 ↓
雅可比速度映射
 ↓
关节速度
 ↓
Franka FR3机器人
```

主要模块：

### 视觉伺服控制律

输入：

- 图像特征误差

输出：

- 期望末端速度

### 速度映射模块

负责：

- Cartesian velocity 到 joint velocity 转换
- 基于 Jacobian 的运动映射

### 安全模块

负责：

- 速度限制
- 关节约束
- 机器人安全保护

## 开发状态

已完成：

- ROS 2 软件框架
- Franka FR3 速度控制
- 笛卡尔速度接口
- 关节速度接口
- 雅可比速度映射

未来计划：

- 相机接口
- 目标检测
- 图像特征提取
- 闭环 IBVS 控制
- 双目视觉伺服
- 实时视觉伺服实验

## 致谢

本项目基于 Franka Robotics 提供的开源 ROS 2 控制生态开发。

底层机器人接口主要来自：

- libfranka
- franka_ros2

本项目主要扩展内容包括：

- 机器人速度控制模块
- 雅可比速度映射模块
- 面向视觉伺服的控制框架

外部开源组件版权和许可证保持原项目规定。