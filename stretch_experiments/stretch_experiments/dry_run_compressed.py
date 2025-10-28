#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time, cv2, os
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import CompressedImage
from control_msgs.action import FollowJointTrajectory
import numpy as np 
from datetime import datetime

# Predefined safe grid
TILT_LEVELS = [0.075, -0.075, -0.450, -0.825, -1.200]
PAN_VALUES = np.linspace(-2.6, 2.6, 20).tolist()


class DryRunCompressedCollector(Node):
    def __init__(self):
        super().__init__('dryrun_compressed_collection_node')
        self.latest_cv_image = None
        self.image_received_flag = False
        self.image_sub = None

        # Create main folder and subfolder for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("compressed_images", f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Action client for moving the head
        self.head_action_client = ActionClient(
            self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory'
        )
        self.get_logger().info("Waiting for head FollowJointTrajectory server...")
        self.head_action_client.wait_for_server()
        self.get_logger().info("Dry-run Compressed Node Initialized")

    def execute_dryrun(self):
        self.get_logger().info("Starting Dry-run (5 tilt × 20 pan) with compressed images")

        sweep_left_to_right = True
        for t_idx, tilt in enumerate(TILT_LEVELS, start=1):
            pan_list = PAN_VALUES if sweep_left_to_right else list(reversed(PAN_VALUES))
            for p_idx, pan in enumerate(pan_list, start=1):
                self.get_logger().info(f"Level {t_idx}, pan {p_idx} → pan={pan:.3f}, tilt={tilt:.3f}")
                self.move_head_to_position(pan, tilt)

                time.sleep(0.7)

                filename = os.path.join(self.output_dir, f"tilt{t_idx}_pan{p_idx}.jpg")
                self.capture_and_save_image(filename)

            sweep_left_to_right = not sweep_left_to_right

        self.get_logger().info("Dry-run sweep complete, returning head to neutral...")
        self.move_head_to_position(0.0, 0.0)
        self.get_logger().info("Dry-run complete, head is now forward and level.")

    def move_head_to_position(self, pan, tilt, settle_sec=2.0):
        traj = JointTrajectory()
        traj.joint_names = ['joint_head_pan', 'joint_head_tilt']
        pt = JointTrajectoryPoint()
        pt.positions = [pan, tilt]
        pt.time_from_start = Duration(sec=2)
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.head_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Head trajectory goal was rejected")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        response = result_future.result()
        result = response.result

        if result.error_code == 0:  # 0 = SUCCESS
            self.get_logger().info("Head movement OK")
        else:
            self.get_logger().warn(
                f"Head error_code={result.error_code}, msg='{result.error_string}'"
            )

        time.sleep(settle_sec)

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.latest_cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.image_received_flag = True
        except Exception as e:
            self.get_logger().error(f"Error decoding compressed image: {e}")

    def capture_and_save_image(self, file_path):
        self.image_received_flag = False
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed', self.image_callback, 10
        )

        timeout_start = time.time()
        while not self.image_received_flag and time.time() - timeout_start < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.image_received_flag and self.latest_cv_image is not None:
            cv2.imwrite(file_path, self.latest_cv_image)
            self.get_logger().info(f"Saved {file_path}")
        else:
            self.get_logger().error("No image saved")

        if self.image_sub:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None


def main(args=None):
    rclpy.init(args=args)
    node = DryRunCompressedCollector()
    node.execute_dryrun()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
