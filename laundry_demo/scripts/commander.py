#!/usr/bin/env python3

#Adapted from code written by Devson Butani, Annalia Schoenherr, Kaushik Mehta
#https://github.com/Aeolus96/stretch_commander

import time

import rospy
from laundry_demo.joint_commands import StretchManipulation
from laundry_demo.nav_commands import StretchNavigation
from laundry_demo.perception_commands import StretchPerception
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray, Detection2D, Detection2DArray
from nav_msgs.msg import Odometry

# Commands to start everything on the robot (using ssh on robot):
"""
roslaunch stretch_funmap mapping.launch rviz:=false
roslaunch synchronized_throttle synchronize_stretch.launch
roslaunch synchronized_throttle generate_pcloud.launch
"""
# Use the custom RViz config (lightweight and fast):
"""
rviz -d $(rospack find laundry_demo)/rviz/command_view.rviz
"""
# DEPRECATED --->
"""
roslaunch stretch_funmap mapping.launch map_yaml:=/home/hello-robot/stretch_user/debug/merged_maps/merged_map_20231130152931 rviz:=false
"""
# <---

# Calibrate the robot motors (every time robots boots up):
"""
rosservice call /calibrate_the_robot "{}"
"""

class state_maching():
    def __init__(self, start_state: str):
        # Initialize the modules:
        self.nav = StretchNavigation()
        self.man = StretchManipulation()
        self.per = StretchPerception()
        self.state=start_state
        self.objects_collected = 0
    
    def timestep(self):
        if self.state == "mapping":
            # # Mapping Sequence:
            self.nav.trigger_head_scan()  # Look around to update the map
            rospy.loginfo("Mapping complete")
            self.state = "detecting"

        elif self.state == "exploring":
            self.nav.trigger_explore()
            self.state = "mapping"

        elif self.state == "returning":
            self.nav.go_to_xya(self.last_pose.pose.point.x, self.last_pose.pose.point.x, self.nav.get_theta(self.last_pose.pose.orientation))
            self.state = "detecting"

        elif self.state == "detecting":
            for i in range(5):  # Cycle through 3 camera positions
                self.man(i + 1)
                time.sleep(3)  # wait for movement to complete
                self.per.trigger_yolo()
                # per.publish_test_box()
                time.sleep(5)  # wait for detection to complete
                if self.per.detected_objects:  # Prevents multiple Detection2DArray publishes
                    break

            if self.per.detected_objects:
                rospy.loginfo(
                    f"Detected Object. Closest Point: {self.nav.target_point.x}, {self.nav.target_point.y}, {self.nav.target_point.z}"
                )
                self.state = "collecting"
                self.man.arm_up()
                self.per.detected_objects = False  # Reset detection flag
            else:
                rospy.loginfo("No objects detected")
                self.state = "exploring"

        elif self.state == "collecting":
            # Save the end position of the exploration state so that we can return in the future
            self.last_pose=rospy.wait_for_message("/odometry",Odometry,timeout=10)

            # Get ready to pick up:
            self.man.arm_up()
            self.man.gripper_open()
            self.man.wrist_down()
            time.sleep(1)
            rospy.loginfo(self.nav.pick_up_at_xyz(self.nav.target_point.x, self.nav.target_point.y, 1.0))
            self.man.arm_extend()
            self.man.arm_down()
            time.sleep(5)
            self.man.gripper_close()
            time.sleep(1)
            self.man.arm_up()
            # man.arm_fold() # Not needed for demo purposes, might obstruct the lidar
            self.state = "dropoff"

        elif self.state == "dropoff":
            # Laundry drop off point (map frame):
            fixed_dropoff_x = 1.0
            fixed_dropoff_y = 0.0
            fixed_dropoff_z = 1.0

            # Go to drop off point
            rospy.loginfo(self.nav.pick_up_at_xyz(fixed_dropoff_x, fixed_dropoff_y, fixed_dropoff_z))
            self.man.gripper_open()
            self.man.wrist_up()
            self.man.wrist_in()
            self.man.arm_fold()

            # Add to objects collected - no feedback or verification implemented for demo purposes
            objects_collected += 1

            # Exit Condition:
            if objects_collected >= 2:  # 2 objects available for demo purposes
                rospy.loginfo("All objects collected! Exiting...")
                return

            self.state = "detecting"


# Start of script:
#######################################################################################################################
if __name__ == "__main__":
    rospy.init_node("laundry_demo")

    try:
        # Start state machine from desired state, mapping | detecting | collecting | dropoff
        SM=state_machine("detecting")  # starting with detecting for testing purposes
        
        while not rospy.is_shutdown():
            # Wait for key press to continue
            input("Press Enter to start ..."+SM.state)
            SM.timestep()



    except rospy.ROSInterruptException:
        pass

# End of script
#######################################################################################################################
