from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='synchronized_throttle',
        #     executable='rgbd_synchronizer',
        #     name='rgbd_synchronizer',
        #     output='screen',
        #     respawn=True,
        #     respawn_delay=10.0,
        #     remappings=[
        #         ('/camera/depth/camera_info', '/zed/zed_node/depth/camera_info'),
        #         ('/camera/depth/image_rect_raw', '/zed/zed_node/depth/depth_registered'),
        #         ('/camera/color/camera_info', '/zed/zed_node/left/camera_info'),
        #         ('/camera/color/image_raw','/zed/zed_node/left/image_rect_color'),
        #     ]
        # ),

        # Start rosbag2 recording
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '/camera_throttled/color/camera_info', '/camera_throttled/color/image_raw', '/camera_throttled/depth/image_rect_raw', '/camera_throttled/thermal/image_raw', '/tf', '/fix'],
            output='screen'
        )    
    ])
