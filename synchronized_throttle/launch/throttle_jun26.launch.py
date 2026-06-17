from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='synchronized_throttle',
            executable='color_depth_thermal_ir_synchronizer',
            name='color_depth_thermal_ir_synchronizer',
            output='screen',
            respawn=True,
            respawn_delay=10.0,
            remappings=[
                # Depth: use aligned depth to color (common realsense topic)
                ('/camera/depth/camera_info', '/camera/realsense/aligned_depth_to_color/camera_info'),
                ('/camera/depth/image_rect_raw', '/camera/realsense/aligned_depth_to_color/image_raw'),

                # Color
                ('/camera/color/camera_info', '/camera/realsense/color/camera_info'),
                ('/camera/color/image_raw','/camera/realsense/color/image_raw'),

                # Thermal (seek)
                ('/camera/thermal/image_raw','/seek/radiometric_raw'),

                # IR (RealSense infra/mono stream) - adjust if your device uses a different infra topic name
                ('/camera/ir/image_raw','/camera/realsense/infra1/image_rect_raw'),
                ('/camera/ir/camera_info','/camera/realsense/infra1/camera_info'),
            ]
        )
    ])
