from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Path to ZED launch file
    zed_launch_path = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'launch',
        'zed_camera.launch.py'
    )

    # Path to Seek Thermal launch file
    seek_launch_path = os.path.join(
        get_package_share_directory('seek_thermal_publisher'),
        'launch',
        'thermal_publisher.launch.py'
    )

    # Path to GPS launch file
    gps_launch_path = os.path.join(
        get_package_share_directory('nmea_navsat_driver'),
        'launch',
        'nmea_serial_driver.launch.py'
    )

    return LaunchDescription([
        # ZED Camera Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(zed_launch_path),
            launch_arguments={'camera_model': 'zed'}.items()
        ),

        # Seek Thermal Camera Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(seek_launch_path)
        ),

        # GPS Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gps_launch_path)
        ),
    ])
    