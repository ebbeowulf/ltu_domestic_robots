#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2DArray
import torch


class Yolov5Node(Node):
    def __init__(self):
        super().__init__('yolov5')

        # Load YOLOv5 model (TODO: replace with actual model)
        self.model = torch.hub.load('todo')
        self.get_logger().info("YOLOv5 model loaded.")

        # Class colors for visualization
        self.class_colors = {
            0: (255, 255, 0),
            24: (255, 0, 255),
            39: (0, 255, 255),
            56: (255, 0, 0),
            60: (0, 255, 0),
            67: (0, 0, 255),
        }

        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Publisher for detections (will stay mostly empty)
        self.detect_pub = self.create_publisher(
            Detection2DArray, "/yolo/detections", 10
        )

        # Subscriber for camera images
        self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, 10
        )

        self.get_logger().info("Subscribed to /camera/color/image_raw")

    def image_callback(self, img_msg: Image):
        # Convert ROS image → OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Rotate + convert BGR → RGB
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        im2 = cv_image[..., ::-1]

        # Inference
        results = self.model([im2], size=640)

        # Build empty detection array (like ROS1)
        res_msg = Detection2DArray()
        res_msg.header = img_msg.header

        # Draw boxes if class is in our list and confidence > 0.5
        for det_ in results.xyxy[0]:
            d2 = det_.cpu().numpy()
            cls_num = int(d2[-1])
            conf = float(d2[-2])

            if cls_num in self.class_colors and conf > 0.5:
                start_pt = (int(d2[0]), int(d2[1]))
                end_pt = (int(d2[2]), int(d2[3]))
                text_pt = (int(d2[0]), int(d2[1]) - 5)

                cv_image = cv2.rectangle(cv_image, start_pt, end_pt, self.class_colors[cls_num], 2)
                cv_image = cv2.putText(
                    cv_image,
                    f"{results.names[cls_num]}, {conf:.2f}",
                    text_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.class_colors[cls_num],
                )

        # Publish empty detection array (to match ROS1 behavior)
        self.detect_pub.publish(res_msg)

        # Show annotated image
        cv2.imshow("YOLOv5 Detection", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov5Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

