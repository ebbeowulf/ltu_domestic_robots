#!/usr/bin/env python3

import cv2
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import pdb
import torch
# from PIL import Image
from ultralytics import YOLO


# Initialize the ROS node
rospy.init_node('object_detection_node')

# Specify the path to the model

# model_path = ('/home/terminus/rosproj/models/best.pt')
model_path = rospkg.RosPack().get_path('stretch_commander') +'/models/best.pt'

# print(model_path)

# print(torch.cuda.is_available())

# Load the YOLOv5 model
model = YOLO(model_path)
#model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Create a CvBridge object to convert ROS messages to OpenCV images
bridge = CvBridge()

class ImageProcessingNode:

    def __init__ (self):
        self.bounding_box_publisher = rospy.Publisher("/yolo/results", Detection2DArray, queue_size=10)

        # Subscribe to the image topic
        rospy.Subscriber("/trigger_yolo/", Bool, self.trigger_callback)
        self.trigger = False

        # image_topic = "/usb_cam/image_raw"
        image_topic = "/camera/color/image_raw"
        rospy.Subscriber(image_topic, Image, self.image_callback)

    def trigger_callback(self, msg):
        self.trigger = msg.data
    
    # Define a callback function to process image messages and perform object detection
    def image_callback(self, msg):

        if (self.trigger):
            self.trigger = False
            print("Running...")
            try:
                # Convert the ROS image message to an OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            except Exception as e:
                rospy.logerr("Error processing image: %s", str(e))
                
            # Perform object detection using YOLOv5
            results = model(cv_image, show=True, conf=0.5, iou=0.7, agnostic_nms=True, classes=0, device=0)
            cv2.waitKey(1)
            #pdb.set_trace()
            
            detection_array_msg = Detection2DArray()
            detection_array_msg.header = msg.header

            # Draw bounding boxes on the image
            for result in results[0]:
                for box in result[0].boxes.cpu().numpy():
                    x1 = box.xyxy[0][0]
                    y1 = box.xyxy[0][1]
                    x2 = box.xyxy[0][2]
                    y2 = box.xyxy[0][3]
                    conf = box.conf[0]
                    cls = box.cls[0]
                
                print(conf)
                print(cls)

                det = ObjectHypothesisWithPose()
                det.id = int(cls)
                det.score = conf
                #pdb.set_trace()

                # cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2), (0, 255, 0), 2))
                # cv2.putText(cv_image, f'Class: {int(cls)}, Conf: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detection_msg = Detection2D()
                detection_msg.header = msg.header

                # detection_msg.bbox.xmin = int(x1)
                # detection_msg.bbox.ymin = int(y1)
                # detection_msg.bbox.xmax = int(x2)
                # detection_msg.bbox.ymax = int(y2)

                detection_msg.bbox.size_x = x2 - x1
                detection_msg.bbox.size_y = y2 - y1
                
                detection_msg.bbox.center.x = (x1 + detection_msg.bbox.size_x / 2)
                detection_msg.bbox.center.y = (y1 + detection_msg.bbox.size_y / 2)

                detection_msg.results.append(det)

                detection_array_msg.detections.append(detection_msg)

                self.bounding_box_publisher.publish(detection_array_msg)

            #print(results)

            # Display the annotated image
            # cv2.imshow("Object Detection", cv_image)
            # cv2.waitKey(1)

if __name__ == '__main__':
    node = ImageProcessingNode()
    rospy.spin()