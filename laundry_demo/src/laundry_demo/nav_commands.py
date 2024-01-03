#Adapted from code written by Devson Butani, Annalia Schoenherr, Kaushik Mehta
#https://github.com/Aeolus96/stretch_commander

import sys
import actionlib
import rospy
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Trigger
from stretch_srvs.srv import MoveArm
from tf import transformations


########################################################################################################################
class StretchNavigation:
    def __init__(self, frame_id="map"):
        # Set up the MoveBaseAction client and initialize sub/pubs
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        if not self.client.wait_for_server(rospy.Duration(5)):
            rospy.logerr("Shutting down - failure to connect to the move_base server.")
            sys.exit(1)  # exit because Navigation wont work without a MoveBase server
        rospy.loginfo("Made contact with move_base server")

        self.move_base_goal = MoveBaseGoal()
        self.move_base_goal.target_pose.header.frame_id = frame_id  # map frame for convenience
        self.pick_up_point = Point()  # for MoveArm service
        # publisher for manipulation goal points as fallback
        self.clicked_point_pub = rospy.Publisher("/clicked_point", PointStamped, queue_size=1)
        self.clicked_point_goal = PoseStamped()

        # Subscriber for pickup point for MoveArm service
        self.target_point_topic = "/target_point"
        self.target_point_sub = rospy.Subscriber(self.target_point_topic, PointStamped, self.target_point_callback)
        self.target_point = Point()

    # Update the target point when received
    def target_point_callback(self, msg: PointStamped):
        self.target_point = msg.point

    # Re-locates the robot in the pre-loaded map
    def trigger_global_localization(self):
        rospy.loginfo("Re-locating the robot...")
        rospy.wait_for_service("/funmap/trigger_global_localization")
        try:
            trigger_loc = rospy.ServiceProxy("/funmap/trigger_global_localization", Trigger)
            response = trigger_loc()
            rospy.loginfo("Re-locating the robot complete")
            return response.message
        except rospy.ServiceException as e:
            rospy.logerr(f"Global localization service call failed: {e}")
            return None

    # Scans the surrounding area using the RealSense depth camera
    def trigger_head_scan(self):
        rospy.loginfo("Scanning the area...")
        rospy.wait_for_service("/funmap/trigger_head_scan")
        try:
            trigger_scan = rospy.ServiceProxy("/funmap/trigger_head_scan", Trigger)
            response = trigger_scan()
            rospy.loginfo("Scanning the area complete")
            return response.message
        except rospy.ServiceException as e:
            rospy.logerr(f"Head scan service call failed: {e}")
            return None

    # Scans the surrounding area using the RealSense depth camera
    def trigger_explore(self):
        rospy.loginfo("Moving to new exploration area...")
        rospy.wait_for_service("/funmap/trigger_drive_to_scan")
        try:
            trigger_scan = rospy.ServiceProxy("/funmap/trigger_drive_to_scan", Trigger)
            response = trigger_scan()
            rospy.loginfo("Moving to new area complete")
            return response.message
        except rospy.ServiceException as e:
            rospy.logerr(f"Move to new area service call failed: {e}")
            return None
        
    # Converts an angle in radians to a quaternion for use with MoveBaseGoal
    def get_quaternion(self, theta):
        return Quaternion(*transformations.quaternion_from_euler(0.0, 0.0, theta))

    # Converts an angle in radians to a quaternion for use with MoveBaseGoal
    def get_theta(self, quat):
        return (transformations.euler_from_quaternion(quat)[2])

    # Callback for when the goal is reached
    def done_callback(self, status, result):
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"<CHECK> {self.__class__.__name__}: SUCCEEDED in reaching the goal.")
        else:
            rospy.loginfo(f"<CHECK> {self.__class__.__name__}: FAILED in reaching the goal.")
        return

    # x, y, and a are in meters and radians. Uses MoveBase Action Server directly
    # Use math.radians(theta) to use degrees
    def go_to_xya(self, x, y, theta, max_retries=3):
        rospy.loginfo(f"{self.__class__.__name__}: Heading for ({x}, {y}) at {theta} radians")
        self.move_base_goal.target_pose.header.stamp = rospy.Time()
        self.move_base_goal.target_pose.pose.position.x = x
        self.move_base_goal.target_pose.pose.position.y = y
        self.move_base_goal.target_pose.pose.position.z = 0.0  # Ground plane as default
        self.move_base_goal.target_pose.pose.orientation = self.get_quaternion(theta)

        # Send the goal and wait for the result, with retry logic
        retries = 0
        while retries < max_retries:
            self.client.send_goal(self.move_base_goal, done_cb=self.done_callback)
            self.client.wait_for_result()
            if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo(f"{self.__class__.__name__}: Reached at ({x}m, {y}m) at {theta} radians")
                break
            else:
                rospy.loginfo(f"{self.__class__.__name__}: Navigation attempt {retries + 1}/{max_retries} failed.")
                retries += 1
        return

    # x, y, and a are in meters and radians. Uses MoveBase Action Server directly
    # Use math.radians(theta) to use degrees
    def go_to_xya(self, x, y, theta, max_retries=3):
        rospy.loginfo(f"{self.__class__.__name__}: Heading for ({x}, {y}) at {theta} radians")
        self.move_base_goal.target_pose.header.stamp = rospy.Time()
        self.move_base_goal.target_pose.pose.position.x = x
        self.move_base_goal.target_pose.pose.position.y = y
        self.move_base_goal.target_pose.pose.position.z = 0.0  # Ground plane as default
        self.move_base_goal.target_pose.pose.orientation = self.get_quaternion(theta)

        # Send the goal and wait for the result, with retry logic
        retries = 0
        while retries < max_retries:
            self.client.send_goal(self.move_base_goal, done_cb=self.done_callback)
            self.client.wait_for_result()
            if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo(f"{self.__class__.__name__}: Reached at ({x}m, {y}m) at {theta} radians")
                break
            else:
                rospy.loginfo(f"{self.__class__.__name__}: Navigation attempt {retries + 1}/{max_retries} failed.")
                retries += 1
        return
        
    # x, y, and z are in meters. Uses FunMap clicked_point topic OR MoveArm service
    def pick_up_at_xyz(self, x, y, z, using_service=True):
        if using_service:  # MoveArm service call: Default
            rospy.loginfo(f"Sending the pickup point {x}, {y}, {z} to MoveArm service...")
            rospy.wait_for_service("/funmap/move_arm")
            try:
                move_arm = rospy.ServiceProxy("/funmap/move_arm", MoveArm)
                self.pick_up_point.x = x
                self.pick_up_point.y = y
                self.pick_up_point.z = z
                response = move_arm(self.pick_up_point)
                rospy.loginfo(f"Point {x}, {y}, {z} reached.")
                return f"Point {x}, {y}, {z} reached."
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                return f"Failed to reach point {x}, {y}, {z}"

        else:  # FunMap clicked_point topic: Alternative
            rospy.loginfo(f"Sending the pickup point {x}, {y}, {z} to FunMap clicked_point topic...")
            self.clicked_point_goal.header.stamp = rospy.Time.now()
            self.clicked_point_goal.header.frame_id = "map"
            self.clicked_point_goal.pose.position.x = x
            self.clicked_point_goal.pose.position.y = y
            z_offset = -0.1  # Adjust this value as needed
            self.clicked_point_goal.pose.position.z = z + z_offset
            self.clicked_point_pub.publish(self.clicked_point_goal)
            rospy.loginfo(f"Point {x}, {y}, {z} reached.")
            return f"Point {x}, {y}, {z} sent."


# End of class
########################################################################################################################
