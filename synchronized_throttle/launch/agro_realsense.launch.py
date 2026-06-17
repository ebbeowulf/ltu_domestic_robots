from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense',
            output='screen',
            parameters=[

                # --- Enable streams ---
                {"enable_color": True},
                {"enable_depth": True},
                {"enable_infra1": True},
                {"enable_infra2": True},

                # --- Emitter OFF ---
                {"depth_module.emitter_enabled": 0},
                {"depth_module.laser_power": 0},

                # --- Stereo preset ---
                {"depth_module.visual_preset": 3},  # High Accuracy

                # --- IR manual exposure for plants ---
                {"infra1.exposure": 6000},
                {"infra1.gain": 20},
                {"infra2.exposure": 6000},
                {"infra2.gain": 20},

                # --- RGB auto-exposure ON ---
                {"rgb_camera.enable_auto_exposure": True},

                # --- Stereo tuning ---
                {"depth_module.confidence_threshold": 1},
                {"depth_module.receiver_gain": 20},

                # --- Resolution & FPS ---
                {"infra1.width": 848},
                {"infra1.height": 480},
                {"infra1.fps": 30},
                {"infra2.width": 848},
                {"infra2.height": 480},
                {"infra2.fps": 30},
                {"depth_module.depth_profile": "848x480x30"},
                {"rgb_camera.color_profile": "1280x720x30"},

                # --- Post-processing filters ---
                {"spatial_filter.enable": True},
                {"spatial_filter.filter_magnitude": 2},
                {"temporal_filter.enable": True},
                {"decimation_filter.enable": True},
                {"decimation_filter.filter_magnitude": 2},

                # --- Align depth to color ---
                {"align_depth.enable": True},

                # --- Misc ---
                {"publish_tf": True},
            ],
        )
    ])
