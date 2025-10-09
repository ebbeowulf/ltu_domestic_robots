from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='synchronized_throttle',
            executable='rgbd_synchronizer',
            name='rgbd_synchronizer',
            output='screen',
            respawn=True,
            respawn_delay=10.0,
            remappings=[
                ('/camera/depth/camera_info', '/zed/zed_node/depth/camera_info'),
                ('/camera/depth/image_rect_raw', '/zed/zed_node/depth/depth_registered'),
                ('/camera/color/camera_info', '/zed/zed_node/left/camera_info'),
                ('/camera/color/image_raw','/zed/zed_node/left/image_rect_color'),
            ]
        )
    ])
