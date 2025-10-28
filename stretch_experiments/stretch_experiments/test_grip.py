#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
import time


class PickSequenceNode(Node):
    def __init__(self):
        super().__init__('pick_sequence_node')

        # Action client for Stretch driver
        self.client = ActionClient(
            self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory'
        )

        self.client.wait_for_server()
        self.get_logger().info("Connected to trajectory server ✅")

    def send_group(self, name, joint_targets, duration=5.0):
        """Send a trajectory command for one or more joints."""
        traj = JointTrajectory()
        traj.joint_names = list(joint_targets.keys())

        pt = JointTrajectoryPoint()
        pt.positions = list(joint_targets.values())
        pt.time_from_start = Duration(sec=int(duration))
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()

        if not handle.accepted:
            self.get_logger().error(f"❌ Trajectory [{name}] was rejected")
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future) 

        time.sleep(0.5)  # pause between steps
        return True

    def pick_sequence(self):
        self.get_logger().info("Starting pick sequence...")

        # Step 1: Arm down & extend
        self.send_group("arm_extend", {"joint_lift": 0.13, "joint_arm": 0.3, "gripper_aperture": 0.1}, duration=7.0)

        # Step 2: Adjust wrist
        self.send_group("wrist_down", {"wrist_pitch": -0.6}, duration=3.0)

        # Step 3: Close gripper to grab
        self.send_group("gripper_close", {"gripper_aperture": 0.0}, duration=2.0)

        # Step 4: Lift & retract arm
        self.send_group("arm_retract", {"joint_lift": 0.4, "joint_arm": 0.2}, duration=5.0)

        # Step 5: Wrist adjust while holding
        self.send_group("wrist_hold", {"wrist_pitch": -0.3}, duration=2.0)

        self.get_logger().info("Pick sequence complete ✅")


def main(args=None):
    rclpy.init(args=args)
    node = PickSequenceNode()
    node.pick_sequence()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
