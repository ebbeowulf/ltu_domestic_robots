#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
import time
import math 
from scipy.spatial.transform import Rotation  


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# --- Parameters ---
TARGET_CLASS = "bottle"  # Change as needed
STOP_DISTANCE = 0.5          # Desired stop distance (m)
REACH_DISTANCE = 0.6         # Threshold for direct grasp (m)
ALIGN_TOLERANCE = 0.05       # Radians (~3°) 
CAMERA_FRAME = "camera_link"


class MoveAndPickUpSmart(Node):
    def __init__(self):
        super().__init__('move_and_pickup_smart_node')

        # Load YOLOv5s
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.5

        self.bridge = CvBridge()
        self.navigator = BasicNavigator()
        self.marker_pub = self.create_publisher(Marker, '/detected_goal_marker', 10)

        self.camera_matrix = None
        self.depth_image = None
        self.robot = None
        self.target_found = False

        # Subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info("Move+Pickup node ready (using odom frame).")

    # -------------------------------------------------
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("📷 Camera intrinsics received.")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # -------------------------------------------------
    def publish_marker(self, x, y):
        """Publish a green sphere marker showing detected goal."""
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Marker published at (x={x:.2f}, y={y:.2f}) in odom frame")

    # -------------------------------------------------
    def image_callback(self, msg):
        """Main callback for color images + detection logic."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]

        # Draw detections
        for _, det in detections.iterrows():
            label = f"{det['name']} {det['confidence']:.2f}"
            cv2.rectangle(cv_image, (int(det['xmin']), int(det['ymin'])),
                          (int(det['xmax']), int(det['ymax'])), (0, 255, 0), 2)
            cv2.putText(cv_image, label, (int(det['xmin']), int(det['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("YOLO Detections", cv_image)
        cv2.waitKey(1)

        if self.target_found or self.depth_image is None or self.camera_matrix is None:
            return

        # --- Select target class ---
        target = detections[detections['name'] == TARGET_CLASS]
        if target.empty:
            return

        # --- Bounding box region ---
        xmin = int(target.iloc[0]['xmin'])
        xmax = int(target.iloc[0]['xmax'])
        ymin = int(target.iloc[0]['ymin'])
        ymax = int(target.iloc[0]['ymax'])
        h, w = self.depth_image.shape[:2]
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)

        # --- Median depth computation ---
        depth_region = self.depth_image[ymin:ymax, xmin:xmax].astype(float)
        valid_depths = depth_region[(depth_region > 100) & (depth_region < 2000)]  # 0.1–2.0 m valid
        if valid_depths.size == 0:
            self.get_logger().warn("No valid depth pixels; skipping detection.")
            return

        Z = np.median(valid_depths) / 1000.0  # meters
        if Z > 2.0:
            self.get_logger().warn(f"Unrealistic Z={Z:.2f}m, clamping to 0.45m.")
            Z = 0.45

        # --- Compute 3D offsets ---
        x_center = int((xmin + xmax) / 2)
        y_center = int((ymin + ymax) / 2)
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        X = (x_center - cx) * Z / fx
        Y = (y_center - cy) * Z / fy

        self.get_logger().info(f"{TARGET_CLASS} @ (X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}) m")

        goal_x = Z - STOP_DISTANCE
        goal_y = -X
        self.publish_marker(goal_x, goal_y)

        angle_offset = np.arctan2(X, Z)
        self.get_logger().info(f"Angle offset: {np.degrees(angle_offset):.1f}°")

        # --- Behavior selection ---
        if Z <= REACH_DISTANCE:
            self.handle_local_grasp(angle_offset, Z)
            return

        # --- Ask for confirmation before navigating ---
        print(f"\n Object '{TARGET_CLASS}' detected at {Z:.2f} m ahead. Move toward it? [y/n]: ", end="")
        choice = input().strip().lower()
        if choice != 'y':
            self.get_logger().info("Movement canceled by user.")
            return

        self.navigate_to(goal_x, goal_y)

    # -------------------------------------------------
    def handle_local_grasp(self, angle_offset, distance):
        """Align arm and camera, or just reach if already aligned."""
        if self.robot is None:
            from stretch_body.robot import Robot
            self.robot = Robot()
            self.robot.startup()

        current_yaw = self.robot.end_of_arm.status['wrist_yaw']['pos']
        if abs(angle_offset + current_yaw) < ALIGN_TOLERANCE:
            self.get_logger().info("Already aligned. Reaching out to grasp.")
        else:
            self.get_logger().info(f"Close object ({distance:.2f} m). Aligning head + wrist...")
            self.robot.head.move_to('head_pan', -angle_offset)
            self.robot.end_of_arm.move_to('wrist_yaw', -angle_offset)
            time.sleep(1)

        # Reach and grasp
        self.robot.arm.move_to(0.3)
        self.robot.lift.move_to(0.55)
        time.sleep(1)
        self.robot.end_of_arm.move_to('gripper', 40)
        time.sleep(1)
        self.robot.end_of_arm.move_to('gripper', 0)
        self.robot.stop()

        self.get_logger().info("Local grasp complete (no navigation).")
        self.target_found = True

   
    # -------------------------------------------------
    def navigate_to(self, goal_x, goal_y):
        """Drive toward target using Nav2, using map frame."""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y

    
        euler = [0, 0, 1.57]  # Roll, pitch, yaw  
        quat = Rotation.from_euler('xyz', euler).as_quat()  # x, y, z, w
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]   

        self.get_logger().info(f"Navigating (odom frame): forward {goal_x:.2f} m, left {goal_y:.2f} m")
        self.navigator.clearAllCostmaps()
        self.navigator.goToPose(goal)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f"Distance remaining: {feedback.distance_remaining:.2f} m")

        result = self.navigator.getResult()
        if result == 0:
            self.get_logger().info("Arrived near target. Starting pickup...")
            self.handle_local_grasp(0.0, STOP_DISTANCE)
        else:
            self.get_logger().warn(f"Navigation failed. getResult()={result}")
            
        self.target_found = True


# -------------------------------------------------
def main():
    rclpy.init()
    node = MoveAndPickUpSmart()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
