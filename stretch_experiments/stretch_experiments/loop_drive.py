#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time, cv2, os
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from control_msgs.action import FollowJointTrajectory
import numpy as np
from datetime import datetime
from std_msgs.msg import Empty, Header

# ---- Head sweep grid (safe values) ----
TILT_LEVELS = [-0.20, -0.60]
PAN_VALUES = np.linspace(-2.6, 1.6, 20).tolist()


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collection_node')

        self.bridge = CvBridge()
        self.image_received_flag = False
        self.latest_cv_image = None
        self.image_sub = None

        # --- Create main folder for this mission ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("nav_images", f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Head trajectory action client
        self.head_action_client = ActionClient(
            self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory'
        )
        self.get_logger().info("Waiting for head FollowJointTrajectory server...")
        self.head_action_client.wait_for_server()

        # Nav2 navigation action client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info("Waiting for 'navigate_to_pose' action server...")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("'navigate_to_pose' server not available, waiting...")

        # publisher before saving image
        self.header_publisher = self.create_publisher(Header, '/saved_trigger', 10)

        self.get_logger().info("Data Collector Node Initialized")

    def execute_mission(self):
        self.get_logger().info("Starting Data Collection ")

        # Check these navigation goals 
        navigation_goals = [
            {'x': 0.0009136,  'y': -0.0039599,  'z':  0.0028718,  'w': 0.9999959},
            {'x': 0.9735956,  'y': -0.0076174,  'z': -0.1090271, 'w': 0.9940388},
            # {'x': 1.8587155,  'y': -1.3688953,  'z': -0.7043899, 'w': 0.7098133},
            #{'x': 2.3905945,  'y': -2.3742146,  'z':  0.6990599, 'w': 0.7150631},
            # {'x': 3.4109757, 'y': -0.5486044, 'z': 0.99980324, 'w': 0.01983638},
            #{'x': 2.4850965, 'y': -0.4306899, 'z': 0.9295261,  'w': 0.3687564},
            #{'x': 1.8385944, 'y':  0.4012201, 'z': 0.6960289,  'w': 0.7180138},
            # {'x': 3.6632988, 'y':  1.7720602, 'z': -0.9987383, 'w': 0.0502166},
            # {'x': 1.8199172, 'y':  1.8049228, 'z': -0.9963892, 'w': 0.0849036},
            # {'x': -0.3711476,'y':  2.3193951, 'z': -0.2544262, 'w': 0.9670922},
            # {'x': 0.9968805, 'y':  0.9313034, 'z': -0.7030826, 'w': 0.7111082},
            #{'x': 0.0270076, 'y':  0.0047693, 'z': 0.0150486,  'w': 0.9998868},
        ]

        spot_counter = 0
        for goal in navigation_goals:
            spot_counter += 1
            self.get_logger().info(f'Navigating to spot #{spot_counter} ---')

            if self.send_navigation_goal(goal):
                self.get_logger().info(f'Arrived at Spot #{spot_counter}. Waiting to settle...')
                # time.sleep(2.0)  # pause before head sweep

                sweep_left_to_right = True
                for t_idx, tilt in enumerate(TILT_LEVELS, start=1):
                    pan_list = PAN_VALUES if sweep_left_to_right else list(reversed(PAN_VALUES))
                    for p_idx, pan in enumerate(pan_list, start=1):
                        # self.lock.acquire() 
                        self.get_logger().info(f"Level {t_idx}, pan {p_idx} → pan={pan:.3f}, tilt={tilt:.3f}")
                        self.move_head_to_position(pan, tilt)

                        # self.lock.release() 

                        # Add extra wait only for the *first pan* at a new tilt
                        if p_idx == 1:
                            time.sleep(1.0)   # let camera/head settle

                        msg_header = Header()
                        msg_header.stamp = self.get_clock().now().to_msg()
                        self.header_publisher.publish(msg_header)  # publish header msg before saving image
                        time.sleep(0.1)  # short delay to ensure message is sent

                        # save images in directory 
                        # filename = os.path.join(self.output_dir, f"spot{spot_counter}_tilt{t_idx}_pan{p_idx}.jpg")
                        # self.capture_and_save_image(filename)

                    sweep_left_to_right = not sweep_left_to_right

                # Return to neutral after the sweep
                self.get_logger().info("Returning head to neutral (0.0, 0.0)")
                self.move_head_to_position(0.0, 0.0)

            else:
                self.get_logger().error(f'Failed to reach Spot #{spot_counter}. Skipping.')

        self.get_logger().info("Data Collection Completed")

    def send_navigation_goal(self, goal_dict):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_dict['x']
        goal_msg.pose.pose.position.y = goal_dict['y']
        goal_msg.pose.pose.orientation.z = goal_dict['z']
        goal_msg.pose.pose.orientation.w = goal_dict['w']

        self.get_logger().info(f"Sending goal x={goal_dict['x']}, y={goal_dict['y']}")
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected the goal")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        response = result_future.result()   # NavigateToPose_GetResult_Response
        status = response.status if response is not None else None
        return status == GoalStatus.STATUS_SUCCEEDED

    def move_head_to_position(self, pan, tilt, settle_sec=0.25):
        traj = JointTrajectory()
        traj.joint_names = ['joint_head_pan', 'joint_head_tilt']
        pt = JointTrajectoryPoint()
        pt.positions = [pan, tilt]
        pt.time_from_start = Duration(sec=0, nanosec=800_000_000)
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
        response = result_future.result()   # FollowJointTrajectory_GetResult_Response
        result = response.result            # FollowJointTrajectory.Result

        if result.error_code == 0:          # 0 = SUCCESS
            self.get_logger().info("Head movement OK")
        else:
            self.get_logger().warn(
                f"Head error_code={result.error_code}, msg='{result.error_string}'"
            )

        time.sleep(settle_sec)

    # def image_callback(self, msg):
    #     try:
    #         self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    #         self.image_received_flag = True
    #     except Exception as e:
    #         self.get_logger().error(f"cv_bridge error: {e}")

    # def capture_and_save_image(self, file_path):
    #     self.image_received_flag = False
    #     self.image_sub = self.create_subscription(
    #         Image, '/camera/color/image_raw', self.image_callback, 10
    #     )
    #     timeout_start = time.time()
    #     while not self.image_received_flag and time.time() - timeout_start < 3.0:
    #         rclpy.spin_once(self, timeout_sec=0.1)

    #     if self.image_received_flag and self.latest_cv_image is not None:
    #         cv2.imwrite(file_path, self.latest_cv_image)
    #         self.get_logger().info(f"Saved {file_path}")
    #     else:
    #         self.get_logger().error("No image saved")

    #     if self.image_sub:
    #         self.destroy_subscription(self.image_sub)
    #         self.image_sub = None


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    node.execute_mission()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
