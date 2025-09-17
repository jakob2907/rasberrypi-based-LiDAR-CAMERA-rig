from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    livox_launch = IncludeLaunchDescription(
         PythonLaunchDescriptionSource(
              os.path.join(
                   get_package_share_directory('livox_ros_driver'),
                   'launch',
                   'livox_lidar_launch.py'
              )
         )
    )

        #BNO085 IMU
    bno085_Node = Node(
             package='bno085',
             executable='bno085_publisher',
             output='screen'
    )

        #Camera_ros
    camera_Node = Node(
             package='camera_ros',
             executable='camera_node',
             output='screen'
    )

    return LaunchDescription([
        livox_launch,
        bno085_Node,
        camera_Node
    ])