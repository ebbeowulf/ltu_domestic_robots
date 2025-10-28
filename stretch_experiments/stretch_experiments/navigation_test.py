#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class NavTest(Node):
    def __init__(self):
        super().__init__('nav_test_node')

        # Nav2 action client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info("Waiting for 'navigate_to_pose' action server...")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("'navigate_to_pose' server not available, waiting...")

        self.get_logger().info("NavTest Node Initialized")

    def execute_test(self):
        self.get_logger().info("Starting Navigation Goal Test")

        # Replace with your RViz-captured waypoints
        navigation_goals = [
            {'x': 0.0009136,  'y': -0.0039599,  'z':  0.0028718,  'w': 0.9999959},
            {'x': 0.9735956,  'y': -0.0076174,  'z': -0.1090271, 'w': 0.9940388},
            {'x': 1.8587155,  'y': -1.3688953,  'z': -0.7043899, 'w': 0.7098133},
            {'x': 2.3905945,  'y': -2.3742146,  'z':  0.6990599, 'w': 0.7150631},
            {'x': 3.4109757, 'y': -0.5486044, 'z': 0.99980324, 'w': 0.01983638},
            {'x': 2.4850965, 'y': -0.4306899, 'z': 0.9295261,  'w': 0.3687564},
            {'x': 1.8385944, 'y':  0.4012201, 'z': 0.6960289,  'w': 0.7180138},
            {'x': 3.6632988, 'y':  1.7720602, 'z': -0.9987383, 'w': 0.0502166},
            {'x': 1.8199172, 'y':  1.8049228, 'z': -0.9963892, 'w': 0.0849036},
            {'x': -0.3711476,'y':  2.3193951, 'z': -0.2544262, 'w': 0.9670922},
            {'x': 0.9968805, 'y':  0.9313034, 'z': -0.7030826, 'w': 0.7111082},
            {'x': 0.0270076, 'y':  0.0047693, 'z': 0.0150486,  'w': 0.9998868},
        ]

        for i, goal in enumerate(navigation_goals, start=1):
            self.get_logger().info(f"Navigating to Goal #{i}")
            success = self.send_navigation_goal(goal)
            if success:
                self.get_logger().info(f"Goal #{i} succeeded ✅")
            else:
                self.get_logger().error(f"Goal #{i} failed ❌ — skipping to next")
            time.sleep(2.0)  # short pause between goals
        

        for i, goal in enumerate(navigation_goals, start=1):
            self.get_logger().info(f"Navigating to Goal #{i}")
            success = self.send_navigation_goal(goal)
            if success:
                self.get_logger().info(f"Goal #{i} succeeded ✅")
            else:
                self.get_logger().error(f"Goal #{i} failed ❌ — skipping to next")
            time.sleep(2.0)  # short pause between goals

        self.get_logger().info("Navigation Goal Test Complete")

    def send_navigation_goal(self, goal_dict):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_dict['x']
        goal_msg.pose.pose.position.y = goal_dict['y']
        goal_msg.pose.pose.orientation.z = goal_dict['z']
        goal_msg.pose.pose.orientation.w = goal_dict['w']

        send_goal_future = self.nav_to_pose_client.send_goal_async(goal)

        self.get_logger().info("Navigation Goal Test Complete")

    def send_navigation_goal(self, goal_dict):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_dict['x']
        goal_msg.pose.pose.position.y = goal_dict['y']
        goal_msg.pose.pose.orientation.z = goal_dict['z']
        goal_msg.pose.pose.orientation.w = goal_dict['w']

        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        response = result_future.result()   # NavigateToPose_GetResult_Response
        status = response.status if response is not None else None
        return status == GoalStatus.STATUS_SUCCEEDED


def main(args=None):
    rclpy.init(args=args)
    node = NavTest()
    node.execute_test()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
