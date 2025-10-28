#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray
from nav2_simple_commander.robot_navigator import BasicNavigator
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Parameters ---
TARGET_CLASS = "bottle"     
STOP_DISTANCE = 0.5
REACH_DISTANCE = 0.6
CAMERA_FRAME = "camera_link"


class MoveAndPickUpSmart(Node):
    def __init__(self):
        super().__init__('move_and_pickup_smart_node')

        # --- YOLO setup ---
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.5
        self.bridge = CvBridge()
        self.navigator = BasicNavigator()

        # --- Publishers ---
        self.marker_pub = self.create_publisher(Marker, '/detected_goal_marker', 10)
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_pose_cmd', 10)
        self.pub_base = self.create_publisher(Twist, '/stretch/cmd_vel', 10)

        # --- TF listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Internal state ---
        self.camera_matrix = None
        self.depth_image = None
        self.target_found = False

        # --- Subscriptions ---
        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info("Move+Pickup node initialized.")

    # -------------------------------------------------
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("📷 Camera intrinsics received.")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # -------------------------------------------------
    def publish_marker(self, pose_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_map.pose          # ✅ Pose inside PoseStamped
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0
        self.marker_pub.publish(marker)
        pos = pose_map.pose.position
        self.get_logger().info(f" Marker published at map (x={pos.x:.2f}, y={pos.y:.2f})")

    # -------------------------------------------------
    def image_callback(self, msg):
        """Main callback for YOLO detection + depth logic."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]

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

        target = detections[detections['name'] == TARGET_CLASS]
        if target.empty:
            return

        # --- Depth ---
        xmin, xmax = int(target.iloc[0]['xmin']), int(target.iloc[0]['xmax'])
        ymin, ymax = int(target.iloc[0]['ymin']), int(target.iloc[0]['ymax'])
        depth_region = self.depth_image[ymin:ymax, xmin:xmax].astype(float)
        valid = depth_region[(depth_region > 100) & (depth_region < 2000)]
        if valid.size == 0:
            return
        Z = np.median(valid) / 1000.0
        Z = min(Z, 2.0)

        # --- 3D offset ---
        h, w = self.depth_image.shape[:2]
        x_center, y_center = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        X = (x_center - cx) * Z / fx
        Y = (y_center - cy) * Z / fy
        self.get_logger().info(f"🎯 {TARGET_CLASS} at X={X:.2f}, Z={Z:.2f}")

        goal_x = Z - STOP_DISTANCE
        goal_y = -X
        angle_offset = np.arctan2(X, Z)

        # --- Transform goal into map frame ---
        goal_camera = PoseStamped()
        goal_camera.header.frame_id = CAMERA_FRAME

        # ✅ Use matching timestamp or latest TF to prevent extrapolation
        stamp = self.get_clock().now().to_msg()
        goal_camera.header.stamp = stamp
        target_time = rclpy.time.Time.from_msg(stamp)

        goal_camera.pose.position.x = float(goal_x)
        goal_camera.pose.position.y = float(goal_y)
        goal_camera.pose.position.z = 0.0
        goal_camera.pose.orientation.w = 1.0
        self.get_logger().info("Print 1")

        timeout = Duration(seconds=1.0)
        if not self.tf_buffer.can_transform('map', CAMERA_FRAME, target_time, timeout):
            self.get_logger().info("Print 2")
            self.get_logger().warn("No TF map - camera_link available yet; retrying.")
            return

        try:
            transform = self.tf_buffer.lookup_transform('map', CAMERA_FRAME, target_time, timeout)
            goal_map = tf2_geometry_msgs.do_transform_pose(goal_camera, transform)
            self.get_logger().info(
                f"TF success: goal_map x={goal_map.pose.position.x:.2f}, "
                f"y={goal_map.pose.position.y:.2f}"
            )
            self.publish_marker(goal_map)
        except Exception as e:
            self.get_logger().warn(f" TF to map failed ({e}); skipping.")
            return

        print(f"\n Object '{TARGET_CLASS}' at {Z:.2f} m. Move toward it? [y/n]: ", end="")
        if input().strip().lower() != 'y':
            return

        self.navigate_to(goal_map, angle_offset)

    # -------------------------------------------------
    def rotate_and_grasp(self, angle_offset):
        """Rotate base via /stretch/cmd_vel and move joints via /joint_pose_cmd."""
        self.get_logger().info(f"🔄 Rotating base by {np.degrees(angle_offset):.1f}° ...")

        twist = Twist()
        twist.angular.z = 0.3 * np.sign(angle_offset)
        duration = abs(angle_offset) / 0.3
        end_time = time.time() + duration
        while time.time() < end_time:
            self.pub_base.publish(twist)
            time.sleep(0.1)
        self.pub_base.publish(Twist())
        self.get_logger().info(" Base rotation complete.")

        # --- Arm/lift/gripper ---
        self.get_logger().info(" Moving arm, lift, and gripper...")
        joint_msg = Float64MultiArray()
        arm = 0.32
        lift = 0.55
        gripper_open = 0.5
        gripper_close = 0.0

        joint_msg.data = [np.nan, lift, arm, np.nan, np.nan, np.nan, gripper_open]
        self.pub_joints.publish(joint_msg)
        time.sleep(1.2)

        joint_msg.data = [np.nan, lift, arm, np.nan, np.nan, np.nan, gripper_close]
        self.pub_joints.publish(joint_msg)
        time.sleep(1.0)

        self.get_logger().info(" Grasp complete.")
        self.target_found = True

    # -------------------------------------------------
    def navigate_to(self, goal_map, angle_offset):
        self.get_logger().info(" Navigating to object (map frame)...")
        self.navigator.goToPose(goal_map)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f"Distance remaining: {feedback.distance_remaining:.2f} m")

        result = self.navigator.getResult()
        if result == 0:
            self.get_logger().info(" Arrived near target. Rotating + grasping.")
            self.rotate_and_grasp(angle_offset)
        else:
            self.get_logger().warn(f" Navigation failed ({result}). Trying grasp anyway...")
            self.rotate_and_grasp(angle_offset)


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
