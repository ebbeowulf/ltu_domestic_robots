#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class StretchJointTester(Node):
    def __init__(self):
        super().__init__('stretch_joint_tester')

        # Publishers for all main controllers
        self.arm_pub = self.create_publisher(JointTrajectory, '/stretch_arm_controller/joint_trajectory', 10)
        self.lift_pub = self.create_publisher(JointTrajectory, '/stretch_lift_controller/joint_trajectory', 10)
        self.head_pub = self.create_publisher(JointTrajectory, '/stretch_head_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/stretch_gripper_controller/joint_trajectory', 10)
        self.wrist_pub = self.create_publisher(JointTrajectory, '/stretch_wrist_yaw_controller/joint_trajectory', 10)

        self.get_logger().info("Stretch Joint Tester initialized. Starting motion tests...")

        time.sleep(2.0)  # small delay to ensure driver is ready
        self.run_tests()

    def send_joint_goal(self, pub, joint_names, positions, duration=2.0):
        traj = JointTrajectory()
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = int(duration)
        traj.points.append(pt)
        pub.publish(traj)
        self.get_logger().info(f"Sent goal to {joint_names}: {positions}")

    def run_tests(self):
        # --- Lift up and down ---
        self.get_logger().info("Testing lift...")
        self.send_joint_goal(self.lift_pub, ['joint_lift'], [0.55])
        time.sleep(3)
        self.send_joint_goal(self.lift_pub, ['joint_lift'], [0.45])
        time.sleep(3)

        # --- Arm extend and retract ---
        self.get_logger().info("Testing arm...")
        self.send_joint_goal(self.arm_pub, ['wrist_extension'], [0.25])
        time.sleep(3)
        self.send_joint_goal(self.arm_pub, ['wrist_extension'], [0.05])
        time.sleep(3)

        # --- Head pan left and right ---
        self.get_logger().info("Testing head pan...")
        self.send_joint_goal(self.head_pub, ['joint_head_pan'], [0.4])
        time.sleep(3)
        self.send_joint_goal(self.head_pub, ['joint_head_pan'], [-0.4])
        time.sleep(3)
        self.send_joint_goal(self.head_pub, ['joint_head_pan'], [0.0])
        time.sleep(2)

        # --- Wrist yaw left and right ---
        self.get_logger().info("Testing wrist yaw...")
        self.send_joint_goal(self.wrist_pub, ['joint_wrist_yaw'], [0.5])
        time.sleep(3)
        self.send_joint_goal(self.wrist_pub, ['joint_wrist_yaw'], [-0.5])
        time.sleep(3)
        self.send_joint_goal(self.wrist_pub, ['joint_wrist_yaw'], [0.0])
        time.sleep(2)

        # --- Gripper open and close ---
        self.get_logger().info("Testing gripper...")
        self.send_joint_goal(self.gripper_pub, ['joint_gripper_finger_left'], [0.5])
        time.sleep(2)
        self.send_joint_goal(self.gripper_pub, ['joint_gripper_finger_left'], [0.0])
        time.sleep(2)

        self.get_logger().info("All joint tests complete! Robot responding normally.")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = StretchJointTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Test interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
