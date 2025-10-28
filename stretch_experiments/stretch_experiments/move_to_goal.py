#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET_CLASS = "bottle"
CONF_THRESHOLD = 0.5


class NavGoalDetectAndGrasp(Node):
    def __init__(self):
        super().__init__('nav_goal_detect_and_grasp')

        # --- Navigation ---
        self.navigator = BasicNavigator()

        # --- Publishers ---
        self.pose_pub = self.create_publisher(Float64MultiArray, '/joint_pose_cmd', 10)

        # --- YOLO setup ---
        self.bridge = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                    pretrained=True, skip_validation=True, force_reload=False)
        self.model.conf = CONF_THRESHOLD
        self.model.eval()
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.latest_frame = None
        self.detected_once = False  # prevent repeated grasps

        # --- Subscribe to RViz goal ---
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.get_logger().info("🟢 Node ready. Click '2D Goal Pose' in RViz to send a goal.")

    # --------------------------------------------------
    def goal_callback(self, msg):
        self.get_logger().info(f"🎯 New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.navigator.waitUntilNav2Active()
        self.navigator.goToPose(msg)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f"Moving... {feedback.distance_remaining:.2f} m left")

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Reached goal. Starting live detection...")
        else:
            self.get_logger().warn("Navigation failed.")

    # --------------------------------------------------
    def image_callback(self, msg):
        """Runs YOLO detection and triggers grasp once the target is seen."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]

        found = False

        for _, det in detections.iterrows():
            label = det['name']
            conf = det['confidence']
            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                color = (0, 255, 0) if label == TARGET_CLASS else (255, 0, 0)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if label == TARGET_CLASS:
                    found = True

        # Show visualization
        cv2.imshow("YOLO Live View", cv_image)
        cv2.waitKey(1)

        if found and not self.detected_once:
            self.detected_once = True
            self.get_logger().info(f"✅ Detected {TARGET_CLASS}, attempting grasp...")
            self.perform_grasp()

    # --------------------------------------------------
    def perform_grasp(self):
        self.get_logger().info("🤖 Executing grasp sequence...")

        neutral = [0.0] * 14

        def send(offsets, sleep=1.5):
            cmd = Float64MultiArray()
            cmd.data = [neutral[i] + offsets.get(i, 0.0) for i in range(14)]
            self.pose_pub.publish(cmd)
            time.sleep(sleep)

        # --- Step 1: Lower lift, extend arm, open gripper ---
        send({
            2: 0.25,   # joint_lift
            6: 0.3,    # joint_arm_l0
            13: 0.4,   # left gripper open
            12: -0.4   # right gripper open
        })

        # --- Step 2: Close gripper ---
        send({
            12: 0.0,
            13: 0.0
        }, sleep=1.0)

        # --- Step 3: Retract and lift ---
        send({
            6: 0.05,
            2: 0.4
        })

        self.get_logger().info("🪶 Grasp complete.")
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = NavGoalDetectAndGrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
