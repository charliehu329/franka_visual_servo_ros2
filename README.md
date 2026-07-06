# franka_visual_servo_ros2
```
Python ROS 2 节点
    ↓
发布目标位姿 / 目标速度 / 视觉误差
    ↓
franka_ros2 controller
    ↓
libfranka
    ↓
Franka 机器人
```


工作区代码框架
```Ros2
~/franka_ws
├── src %下载的源码文件夹
├── build %编译的中间文件夹，编译过程中产生的中间文件放在这，不是手写代码的地方
├── install %编译后结果放的地方
└── log %编译日志，编译成功与否记录在这，方便debug
```

# Step1：电脑和Franka连接

1. 用网线连接，设置有线连接的网口

2. 连接Franka Desk,

例如你机器人 IP 是：
172.16.0.2
那你就在浏览器地址栏输入：
[https://172.16.0.2](https://172.16.0.2/desk/)
```
ping 172.16.0.2
```


3. 在 Franka Desk 里面做三件事：

```
1. 机器人上电
2. 解锁机器人
3. Desk 里面点击 Activate FCI %让Franka能够被控制
```

设置好后ping一下看连接是否通畅


# Step2：用Ros控制Franka

```
colcon build：把源码“做成可运行/可识别的东西”
source：告诉当前终端“这些东西在哪里”
launch：真正启动这些 ROS 节点、控制器和硬件接口

源码放到 src
     ↓
colcon build 编译
     ↓
生成 install
     ↓
source install/setup.bash
     ↓
当前终端能找到你编译好的 ROS 包
     ↓
ros2 launch 启动 launch 文件
     ↓
真正启动 ROS 节点、控制器、硬件接口
```

**1. 进入 Franka ROS 2 工作区**

每次打开新终端后，先进入 Franka 工作区：

```
cd /home/harry/franka_ros2_ws
```

**2. 加载 ROS 2 Jazzy 环境：**

```
source /opt/ros/jazzy/setup.bash 
```

**3. 加载 Franka 工作区环境：**

如果工作区已经编译过，直接加载
```
source ~/franka_ros2_ws/install/setup.bash
```

如果没编译过，修改了代码就要重新编译，就要先编译，再加载Franka工作区环境。在运行3之前跑下面代码

```
colcon build --symlink-install %因为进入了franka_ws工作区，会自动编译下面的src里面的源码

colcon build --packages-select <YOUR PACKAGE> %只编译一个包
```

这一步的作用是让终端能够识别 ROS 2 命令、Franka 相关 package，以及自己编译出来的 launch 文件和节点。

**4. 检查是否source成功**

可以用下面命令检查 Franka ROS 2 包是否被识别：
```
ros2 pkg list | grep franka
```

如果能够看到类似下面的输出，说明环境加载成功：

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

以上是编译和source相应的包和要用的代码

**5. 终端1 launch**
launch到topic模式
这个一般要一直开着，用来
连接 Franka
启动 controller
发布 /joint_states
接收速度指令
```
ros2 launch franka_velocity_ctrl fr3_velocity.launch.py \
    robot_ip:=172.16.0.2 \
    mode:=topic
```

launch后可以看controller是否activater
```
ros2 control list_controllers
```


**6. 终端2 运行发布topic的python**
另一个终端运行控制的python代码，或者直接发布Topic

在ros里运行python（新建ros的python包详见常见问题1
```
ros2 run franka_python send_joint_velocity
```

直接发布Topic
```
ros2 topic pub /velocity_command_node/target_velocities \
    std_msgs/msg/Float64MultiArray \
    "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
```



```
ros2 topic pub /velocity_command_node/target_velocities \
    std_msgs/msg/Float64MultiArray \
    "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02]"
```


**6(2). 终端2 运行发布topic的python**

3：编译这个包（可以只编译这一个，如果有代码的改变）
```
colcon build --packages-select franka_python --symlink-install
``` 

4：运行
```
ros2 run franka_python send_joint_velocity
```

