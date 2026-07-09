from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'franka_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        # 安装 config 目录下的 yaml 文件
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),

        # 安装 config/urdf 目录下的 urdf 文件
        (os.path.join('share', package_name, 'config', 'urdf'),
            glob('config/urdf/*.urdf')),

        # 安装 launch 目录下的 launch.py 文件
        (
            os.path.join(
                "share",
                package_name,
                "launch"
            ),
            glob("launch/*.launch.py")
        ),
        (
            os.path.join(
                "share",
                package_name,
                "config"
            ),
            glob("config/*.yaml")
        ),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harry',
    maintainer_email='harry@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        # 在这里写 ROS 节点的名字
        'console_scripts': [
            'send_joint_velocity = franka_python.send_joint_velocity:main',
            'cartesian_servo_node = franka_python.cartesian_servo_node:main',
            "visual_servo_law = franka_python.visual_servo_law:main",
        ],
    },
)