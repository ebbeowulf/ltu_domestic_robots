#!/usr/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Trigger
import sys
import random

class FunmapWaypointNavigator:
    def __init__(self):
        rospy.init_node('funmap_waypoint_navigator', anonymous=True)

        # Define waypoints as (x, y, z) and orientation (qx, qy, qz, qw)
        self.waypoints = [
            [(1.3, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)],  # Waypoint 1
            [(1.3, 1.5, 0.0), (0.0, 0.0, 0.0, 1.0)],  # Waypoint 2
            [(1.3, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)],  # Waypoint 3
            [(1.3, 3.0, 0.0), (0.0, 0.0, 0.0, 1.0)]   # Waypoint 4
            #[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
        ]

        # Action client for move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()

        # Service proxy for trigger_head_scan_service using the Trigger type
        rospy.loginfo("Waiting for trigger_head_scan_service...")
        rospy.wait_for_service('/funmap/trigger_head_scan')
        self.trigger_head_scan = rospy.ServiceProxy('/funmap/trigger_head_scan', Trigger)

    def trigger_head_scan_service(self):
        """ Calls the head scan service. """
        try:
            rospy.loginfo("Calling head scan service...")
            response = self.trigger_head_scan()
            if response.success:
                rospy.loginfo(f"Head scan service completed successfully: {response.message}")
            else:
                rospy.logwarn(f"Head scan service failed: {response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Head scan service call failed: {e}")
            sys.exit(1)

    def navigate_waypoints(self):
        """ Navigate to each waypoint with retries on failure. """
        rospy.loginfo("Starting navigation with initial head scan.")
        self.trigger_head_scan_service()

        rospy.loginfo("Starting waypoint navigation using FUNMAP...")
        for i, (position, orientation) in enumerate(self.waypoints):
            rospy.loginfo(f"Navigating to waypoint {i + 1}: {position}")

            success = self.retry_goal(position, orientation)
            if not success:
                rospy.logwarn(f"Failed to reach waypoint {i + 1} after retries.")

            self.trigger_head_scan_service()
            rospy.sleep(2)

        rospy.loginfo("Completed waypoint navigation.")

    def retry_goal(self, goal_position, goal_orientation, max_retries=5):
        """ Attempts to reach the goal with retries. """
        for attempt in range(max_retries):
            if self.move_to_goal(goal_position, goal_orientation):
                return True
            else:
                # Add random noise to the goal position for retries
                noisy_position = [
                    goal_position[0] + random.uniform(-0.1, 0.1),
                    goal_position[1] + random.uniform(-0.1, 0.1),
                    goal_position[2]
                ]
                rospy.logwarn(f"Retrying with noise: {noisy_position}. Attempt {attempt + 1} of {max_retries}")
        return False

    def move_to_goal(self, goal_position, goal_orientation):
        """ Sends a navigation goal to move_base and checks if it succeeds. """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = goal_position[0]
        goal.target_pose.pose.position.y = goal_position[1]
        goal.target_pose.pose.position.z = goal_position[2]
        goal.target_pose.pose.orientation.x = goal_orientation[0]
        goal.target_pose.pose.orientation.y = goal_orientation[1]
        goal.target_pose.pose.orientation.z = goal_orientation[2]
        goal.target_pose.pose.orientation.w = goal_orientation[3]

        self.client.send_goal(goal)
        self.client.wait_for_result()

        state = self.client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo(f"Successfully reached goal: {goal_position}")
            return True
        else:
            rospy.logerr(f"Failed to reach goal: {goal_position}. Status: {state}")
            return False

if __name__ == '__main__':
    try:
        navigator = FunmapWaypointNavigator()
        navigator.navigate_waypoints()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

