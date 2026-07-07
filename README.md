# Franka Visual Servo ROS2

## Overview

This project implements a visual servo control system for the Franka Emika Panda robot using ROS 2. It provides Python ROS 2 nodes that publish target poses, target velocities, and visual errors to control the Franka robot through the `franka_ros2` controller and `libfranka` interface.

## System Configuration

- Robot: Franka Emika Panda
- ROS 2 Distribution: Jazzy
- Programming Language: Python
- Control Framework: ROS 2 Control with `franka_ros2` controller

## Project Structure

```
franka_visual_servo_ros2/
├── src/                 # Source code folder
├── build/               # Intermediate build files (auto-generated)
├── install/             # Installation folder after build
└── log/                 # Build logs for debugging
```

## ROS2 Workspace Structure

```text
~/franka_ws
├── src                  # Downloaded source code folder
├── build                # Intermediate build files during compilation
├── install              # Build output location
└── log                  # Compilation logs for debugging
```

## Installation

### Step 1: Connect Computer and Franka Robot

1. Connect the robot via Ethernet cable and configure the network interface.
2. Access Franka Desk by opening your browser and navigating to the robot IP address, for example:

```bash
https://172.16.0.2/desk/
```

3. In Franka Desk, perform the following:

```text
1. Power on the robot
2. Unlock the robot
3. Activate FCI to enable robot control
```

Verify the connection by pinging the robot IP:

```bash
ping 172.16.0.2
```

### Step 2: Control Franka with ROS

1. Place the source code in the `src` folder.
2. Build the workspace:

```bash
colcon build
```

3. Source the setup script:

```bash
source install/setup.bash
```

4. Launch ROS nodes, controllers, and hardware interfaces:

```bash
ros2 launch <launch_file>
```

## Usage

### Setup Environment

Open a new terminal and enter the Franka ROS 2 workspace:

```bash
cd /home/harry/franka_ros2_ws
```

Load the ROS 2 Jazzy environment:

```bash
source /opt/ros/jazzy/setup.bash
```

Load the Franka workspace environment:

```bash
source ~/franka_ros2_ws/install/setup.bash
```

If you have modified code, rebuild the workspace:

```bash
colcon build --symlink-install
```

Or build a single package:

```bash
colcon build --packages-select <YOUR_PACKAGE>
```

Verify the environment is loaded correctly by listing Franka ROS 2 packages:

```bash
ros2 pkg list | grep franka
```

Expected output includes:

```bash
franka_bringup
franka_description
franka_example_controllers
franka_hardware
franka_msgs
franka_robot_state_broadcaster
franka_semantic_components
franka_velocity_ctrl
```

### Launch Controllers

In terminal 1, launch the controller in topic mode to connect to Franka and start the controller:

```bash
ros2 launch franka_velocity_ctrl fr3_velocity.launch.py \
    robot_ip:=172.16.0.2 \
    mode:=topic
```

Check if the controller is active:

```bash
ros2 control list_controllers
```

### Publish Velocity Commands

In terminal 2, run the Python node that publishes velocity commands:

```bash
ros2 run franka_python send_joint_velocity
```

Alternatively, publish velocity commands directly via topic:

```bash
ros2 topic pub /velocity_command_node/target_velocities \
    std_msgs/msg/Float64MultiArray \
    "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
```

Example with a non-zero velocity on the last joint:

```bash
ros2 topic pub /velocity_command_node/target_velocities \
    std_msgs/msg/Float64MultiArray \
    "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02]"
```

### Build and Run the Python Package

If you modify the Python package, rebuild it:

```bash
colcon build --packages-select franka_python --symlink-install
```

Run the node again:

```bash
ros2 run franka_python send_joint_velocity
```

## Control Architecture

```
Python ROS 2 Node
    ↓
Publishes target pose / target velocity / visual error
    ↓
franka_ros2 controller
    ↓
libfranka
    ↓
Franka Robot
```

## Development Status

- Core visual servoing nodes implemented in Python.
- Integration with `franka_ros2` controller validated.
- Velocity control tested via topic publishing.

## Troubleshooting

- Ensure the robot IP and network configuration are correct.
- Confirm Franka Desk activation of FCI.
- Verify ROS 2 environment sourcing order.
- Check controller status with `ros2 control list_controllers`.
- Use `ros2 topic echo` to monitor topics for expected messages.

## License

MIT License
