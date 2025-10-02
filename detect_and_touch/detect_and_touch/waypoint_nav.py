#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger
import sys
import random


class FunmapWaypointNavigator(Node):
    def __init__(self):
        super().__init__('funmap_waypoint_navigator')

        # Define waypoints as (x, y, z) and orientation (qx, qy, qz, qw)
        self.waypoints = [
            [(1.3, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)],  # Waypoint 1
            [(1.3, 1.5, 0.0), (0.0, 0.0, 0.0, 1.0)],  # Waypoint 2
            [(1.3, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)],  # Waypoint 3
            [(1.3, 3.0, 0.0), (0.0, 0.0, 0.0, 1.0)]   # Waypoint 4
            #[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
        ]

        # Action client for Navigation2
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info("Waiting for navigate_to_pose action server...")
        self.client.wait_for_server()

        # Service client for trigger_head_scan_service
        self.get_logger().info("Waiting for trigger_head_scan_service...")
        self.trigger_head_scan = self.create_client(Trigger, '/funmap/trigger_head_scan')
        while not self.trigger_head_scan.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /funmap/trigger_head_scan service...")

    def trigger_head_scan_service(self):
        """Calls the head scan service."""
        req = Trigger.Request()
        future = self.trigger_head_scan.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Head scan service completed successfully: {response.message}")
            else:
                self.get_logger().warn(f"Head scan service failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Head scan service call failed: {e}")
            sys.exit(1)

    def navigate_waypoints(self):
        """Navigate to each waypoint with retries on failure."""
        self.get_logger().info("Starting navigation with initial head scan.")
        self.trigger_head_scan_service()

        self.get_logger().info("Starting waypoint navigation using FUNMAP...")
        for i, (position, orientation) in enumerate(self.waypoints):
            self.get_logger().info(f"Navigating to waypoint {i + 1}: {position}")

            success = self.retry_goal(position, orientation)
            if not success:
                self.get_logger().warn(f"Failed to reach waypoint {i + 1} after retries.")

            self.trigger_head_scan_service()
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=2))

        self.get_logger().info("Completed waypoint navigation.")

    def retry_goal(self, goal_position, goal_orientation, max_retries=5):
        """Attempts to reach the goal with retries."""
        for attempt in range(max_retries):
            if self.move_to_goal(goal_position, goal_orientation):
                return True
            else:
                noisy_position = [
                    goal_position[0] + random.uniform(-0.1, 0.1),
                    goal_position[1] + random.uniform(-0.1, 0.1),
                    goal_position[2]
                ]
                self.get_logger().warn(
                    f"Retrying with noise: {noisy_position}. Attempt {attempt + 1} of {max_retries}"
                )
        return False

    def move_to_goal(self, goal_position, goal_orientation):
        """Sends a navigation goal to Nav2 and checks if it succeeds."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = goal_position[0]
        goal_msg.pose.pose.position.y = goal_position[1]
        goal_msg.pose.pose.position.z = goal_position[2]
        goal_msg.pose.pose.orientation.x = goal_orientation[0]
        goal_msg.pose.pose.orientation.y = goal_orientation[1]
        goal_msg.pose.pose.orientation.z = goal_orientation[2]
        goal_msg.pose.pose.orientation.w = goal_orientation[3]

        self.client.wait_for_server()
        send_goal_future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected: {goal_position}")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"Successfully reached goal: {goal_position}")
            return True
        else:
            self.get_logger().error(f"Failed to reach goal: {goal_position}. Status: {status}")
            return False


def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = FunmapWaypointNavigator()
        navigator.navigate_waypoints()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()


