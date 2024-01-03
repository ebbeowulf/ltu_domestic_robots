#Adapted from code written by Devson Butani, Annalia Schoenherr, Kaushik Mehta
#https://github.com/Aeolus96/stretch_commander

from typing import List
import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
import time


########################################################################################################################
class StretchManipulation:
    def __init__(self):
        # Setup the trajectory client for manipulation of joints
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        server_reached = self.trajectory_client.wait_for_server(timeout=rospy.Duration(15.0))
        if not server_reached:
            rospy.logerr("Failed to connect to the trajectory server.")
            return
        rospy.loginfo(f"<CHECK> {self.__class__.__name__}: Made contact with trajectory server")
        self.trajectory_goal = FollowJointTrajectoryGoal()
        self.point0 = JointTrajectoryPoint()

    # Home the robot for the first time after boot. Not necessary after that.
    def trigger_home_the_robot(self):
        rospy.wait_for_service("/calibrate_the_robot")
        try:
            trigger_homing = rospy.ServiceProxy("/calibrate_the_robot", Trigger)
            response = trigger_homing()
            return response.message
        except rospy.ServiceException as e:
            rospy.logerr(f"Home the robot service call failed: {e}")
            return None

    def send_joint_goals(self, joint_names: List[str], goal_values: List[float]):
        if len(joint_names) != len(goal_values):
            rospy.logwarn("Joint names and goal values lists must have the same length.")
            return

        self.trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_goal.trajectory.header.frame_id = "base_link"  # relative to robot base
        self.trajectory_goal.trajectory.joint_names = joint_names
        self.point0.positions = goal_values
        self.trajectory_goal.trajectory.points = [self.point0]

        self.trajectory_client.send_goal(self.trajectory_goal)

        goal_info = ", ".join(f"{joint}: {value}" for joint, value in zip(joint_names, goal_values))
        rospy.loginfo(f"Sent joint goals: {goal_info}")
        # rospy.loginfo(f"Trajectory goal message = {self.trajectory_goal}")  # DEBUG
        self.trajectory_client.wait_for_result()

    def gripper_close(self):
        rospy.loginfo(f"{self.__class__.__name__}: Closing the gripper")
        self.send_joint_goals(["joint_gripper_finger_left"], [-0.25])
        rospy.loginfo("-*- -*- -*-")

    def gripper_open(self):
        rospy.loginfo(f"{self.__class__.__name__}: Opening the gripper")
        self.send_joint_goals(["joint_gripper_finger_left"], [1.65])
        rospy.loginfo("-*- -*- -*-")

    def arm_extend(self):
        rospy.loginfo(f"{self.__class__.__name__}: Extending the arm")
        self.send_joint_goals(["wrist_extension"], [0.45])
        rospy.loginfo("-*- -*- -*-")

    def arm_fold(self):
        rospy.loginfo(f"{self.__class__.__name__}: Folding the arm")
        self.send_joint_goals(["wrist_extension"], [0.0])  # 0.10 to avoid collision with the lift axis
        rospy.loginfo("-*- -*- -*-")

    def arm_up(self):
        rospy.loginfo(f"{self.__class__.__name__}: Lifting the arm up")
        self.send_joint_goals(["joint_lift"], [1.00])
        rospy.loginfo("-*- -*- -*-")

    def arm_down(self):
        rospy.loginfo(f"{self.__class__.__name__}: Dropping the arm down")
        self.send_joint_goals(["joint_lift"], [0.15])
        rospy.loginfo("-*- -*- -*-")

    def wrist_down(self):
        rospy.loginfo(f"{self.__class__.__name__}: Moving the wrist down")
        self.send_joint_goals(["joint_wrist_pitch"], [-1.5])
        rospy.loginfo("-*- -*- -*-")

    def wrist_up(self):
        rospy.loginfo(f"{self.__class__.__name__}: Moving the wrist up")
        self.send_joint_goals(["joint_wrist_pitch"], [0.0])
        rospy.loginfo("-*- -*- -*-")

    def wrist_out(self):
        rospy.loginfo(f"{self.__class__.__name__}: Moving the wrist out")
        self.send_joint_goals(["joint_wrist_yaw"], [-0.25])
        rospy.loginfo("-*- -*- -*-")

    def wrist_in(self):
        rospy.loginfo(f"{self.__class__.__name__}: Moving the wrist in")
        self.send_joint_goals(["joint_wrist_yaw"], [4.00])  # 3.14 is pointing towards the lift axis
        rospy.loginfo("-*- -*- -*-")

    def look_for_shirts(self, pos):
        rospy.loginfo(f"{self.__class__.__name__}: Tilting camera - 60 degrees")
        self.send_joint_goals(["joint_head_tilt"], [-1.0])
        rospy.loginfo("-*- -*- -*-")
        self.gripper_close()
        self.wrist_up()
        self.wrist_in()
        time.sleep(1)
        self.arm_fold()
        self.arm_down()
        if pos == 1:
            rospy.loginfo(f"{self.__class__.__name__}: Moving camera - Center")
            self.send_joint_goals(["joint_head_pan"], [0])
            rospy.loginfo("-*- -*- -*-")
        elif pos == 2:
            rospy.loginfo(f"{self.__class__.__name__}: Moving camera - Mid Right")
            self.send_joint_goals(["joint_head_pan"], [-0.7])
            rospy.loginfo("-*- -*- -*-")
        elif pos == 3:
            rospy.loginfo(f"{self.__class__.__name__}: Moving camera - Right")
            self.send_joint_goals(["joint_head_pan"], [-1.4])
            rospy.loginfo("-*- -*- -*-")
        elif pos == 4:
            rospy.loginfo(f"{self.__class__.__name__}: Moving camera - Mid Left")
            self.send_joint_goals(["joint_head_pan"], [0.7])
            rospy.loginfo("-*- -*- -*-")
        elif pos == 5:
            rospy.loginfo(f"{self.__class__.__name__}: Moving camera - Left")
            self.send_joint_goals(["joint_head_pan"], [1.4])
            rospy.loginfo("-*- -*- -*-")


# End of class
########################################################################################################################
"""
############################# JOINT LIMITS #############################
joint_lift: {0.15m - 1.10m}
wrist_extension: {0.00m - 0.50m}
joint_wrist_yaw: {-1.75rad - 4.00rad}
joint_head_pan: {-2.80rad - 2.90rad}
joint_head_tilt: {-1.60rad - 0.40rad}
joint_gripper_finger_left: {-0.35rad - 0.165rad}

########################## All Joint Names ############################
[Arm] joint_lift, wrist_extension = (joint_arm_l0 + joint_arm_l1 + joint_arm_l2 + joint_arm_l3)
[Gripper] joint_gripper_finger_left OR joint_gripper_finger_right
[Head] joint_head_pan, joint_head_tilt
[Wrist] joint_wrist_pitch, joint_wrist_yaw, joint_wrist_roll
[Base] joint_left_wheel, joint_right_wheel

# INCLUDED JOINTS IN POSITION MODE - Relative (not tested yet)
translate_mobile_base: No lower or upper limit. Defined by a step size in meters
rotate_mobile_base:    No lower or upper limit. Defined by a step size in radians
########################################################################
"""
