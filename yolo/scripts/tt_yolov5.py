#!/usr/bin/env python3

# Code originally written by Jacob Hallett (https://github.com/Aeolus96/stretch_commander)


import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics import YOLO
import sys


class ImageProcessingNode(Node):
    def __init__(self, model_path):
        super().__init__('object_detection_node')

        # Load YOLOv5 model
        self.get_logger().info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)

        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Publisher for detections
        self.bounding_box_publisher = self.create_publisher(
            Detection2DArray, "/yolo/results", 10
        )

        # Trigger subscriber
        self.trigger = False
        self.create_subscription(Bool, "/trigger_yolo/", self.trigger_callback, 10)

        # Image subscriber
        self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)

    def trigger_callback(self, msg: Bool):
        self.trigger = msg.data

    def image_callback(self, msg: Image):
        if self.trigger:
            self.trigger = False
            self.get_logger().info("Running YOLO detection...")

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")
                return

            # Run YOLO
            results = self.model(
                cv_image, show=True, conf=0.5, iou=0.7,
                agnostic_nms=True, classes=0, device=0
            )
            cv2.waitKey(1)

            detection_array_msg = Detection2DArray()
            detection_array_msg.header = msg.header

            # Process detections
            for result in results:
                for box in result.boxes.cpu().numpy():
                    x1 = box.xyxy[0][0]
                    y1 = box.xyxy[0][1]
                    x2 = box.xyxy[0][2]
                    y2 = box.xyxy[0][3]
                    conf = box.conf[0]
                    cls = box.cls[0]

                    print(conf, cls)

                    det = ObjectHypothesisWithPose()
                    det.id = int(cls)
                    det.score = float(conf)

                    detection_msg = Detection2D()
                    detection_msg.header = msg.header
                    detection_msg.bbox.size_x = x2 - x1
                    detection_msg.bbox.size_y = y2 - y1
                    detection_msg.bbox.center.x = (x1 + detection_msg.bbox.size_x / 2)
                    detection_msg.bbox.center.y = (y1 + detection_msg.bbox.size_y / 2)
                    detection_msg.results.append(det)

                    detection_array_msg.detections.append(detection_msg)

            # Publish detections
            self.bounding_box_publisher.publish(detection_array_msg)


def main(args=None):
    rclpy.init(args=args)

    # Handle model path parameter
    node = rclpy.create_node("param_loader")
    if node.has_parameter("model"):
        model_path = node.get_parameter("model").get_parameter_value().string_value
    else:
        node.declare_parameter("model", "")
        model_path = node.get_parameter("model").get_parameter_value().string_value

    if not model_path:
        print("No model parameter provided - bailing")
        sys.exit(-1)

    node.destroy_node()

    # Start main detection node
    detection_node = ImageProcessingNode(model_path=model_path)
    rclpy.spin(detection_node)

    detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

