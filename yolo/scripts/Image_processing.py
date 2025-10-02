#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import numpy as np
import torch
from ultralytics import YOLO

########### GET MODEL PATH FROM OG CODE IN GITHUB ####################

class ImageProcessingNode(Node):
    def __init__(self, model_path):
        super().__init__('object_detection_node')

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Publisher for YOLO detections
        self.bounding_box_publisher = self.create_publisher(
            Detection2DArray, "/yolo/results", 10
        )

        # Trigger subscriber
        self.trigger = False
        self.create_subscription(Bool, "/trigger_yolo/", self.trigger_callback, 10)

        # Image subscriber
        self.create_subscription(Image, "/camera_throttled/color/image_raw", self.image_callback, 10)

        self.get_logger().info("YOLO Object Detection Node initialized.")

    def trigger_callback(self, msg: Bool):
        self.trigger = msg.data
        self.get_logger().info("Trigger received")

    def image_callback(self, msg: Image):
        self.get_logger().info("Image received")

        if self.trigger:
            self.trigger = False
            self.get_logger().info("Running YOLO detection...")

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")
                return

            # Run YOLO detection
            results = self.model(
                cv_image, show=True, conf=0.5, iou=0.7,
                agnostic_nms=True, classes=0, device=0
            )
            cv2.waitKey(1)

            detection_array_msg = Detection2DArray()
            detection_array_msg.header = msg.header

            # Parse YOLO results
            for result in results:
                for box in result.boxes.cpu().numpy():
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    self.get_logger().info(f"Detected class {cls} with confidence {conf:.2f}")

                    det = ObjectHypothesisWithPose()
                    det.id = cls
                    det.score = conf

                    detection_msg = Detection2D()
                    detection_msg.header = msg.header
                    detection_msg.bbox.size_x = x2 - x1
                    detection_msg.bbox.size_y = y2 - y1
                    detection_msg.bbox.center.x = x1 + detection_msg.bbox.size_x / 2
                    detection_msg.bbox.center.y = y1 + detection_msg.bbox.size_y / 2
                    detection_msg.results.append(det)

                    detection_array_msg.detections.append(detection_msg)

            # Publish detection results
            self.bounding_box_publisher.publish(detection_array_msg)


def main(args=None):
    rclpy.init(args=args)

    # TODO: Replace with actual path or make it a ROS2 parameter
    model_path = "path/to/model.pt"
    node = ImageProcessingNode(model_path)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

