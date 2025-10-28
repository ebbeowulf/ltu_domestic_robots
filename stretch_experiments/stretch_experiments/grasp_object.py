#!/usr/bin/env python3
"""
Grasp sequence for Stretch 2 — using /joint_pose_cmd (driver mode)
No need for /stretch/joint_trajectory
"""

import rclpy
from rclpy.node import Node
from stretch_msgs.msg import JointPose
import time


class StretchGraspPoseCmd(Node):
    def __init__(self):
        super().__init__('stretch_grasp_posecmd')

        self.pub = self.create_publisher(JointPose, '/joint_pose_cmd', 10)
        time.sleep(1.0)  # give publisher time to connect

        self.get_logger().info("Starting grasp sequence using /joint_pose_cmd ...")
        self.perform_grasp_sequence()

    def send_pose(self, joint_name, position):
        """Send a single joint pose command"""
        msg = JointPose()
        msg.joint_name = joint_name
        msg.position = position
        self.pub.publish(msg)
        self.get_logger().info(f" → Moving {joint_name} to {position:.3f}")
        time.sleep(2.5)  # allow motion to finish

    def perform_grasp_sequence(self):
        """
        Simple grasp sequence:
        1. Raise lift
        2. Extend arm
        3. Open gripper
        4. Lower lift (approach)
        5. Close gripper (grasp)
        6. Raise lift (lift object)
        7. Retract arm
        """

        # Step 1: Raise lift
        self.send_pose('joint_lift', 0.8)

        # Step 2: Extend arm
        self.send_pose('wrist_extension', 0.45)

        # Step 3: Open gripper
        self.send_pose('gripper_aperture', 0.07)

        # Step 4: Lower slightly
        self.send_pose('joint_lift', 0.7)

        # Step 5: Close gripper
        self.send_pose('gripper_aperture', 0.0)

        # Step 6: Lift object
        self.send_pose('joint_lift', 0.9)

        # Step 7: Retract arm
        self.send_pose('wrist_extension', 0.0)

        self.get_logger().info("✅ Grasp sequence completed (driver mode).")


def main():
    rclpy.init()
    node = StretchGraspPoseCmd()
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
