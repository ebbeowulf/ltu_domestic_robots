#!/usr/bin/env python3
import rospy
import cv2
import pdb
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from simple_camera_transformer.cfg import TransformerConfig
from vision_msgs.msg import Detection2DArray, Detection2D
import numpy as np
import torch

class yolov5:    
    def __init__(self):
        # Initization of the node, name_sub
        rospy.init_node('yolov5', anonymous=True)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.class_colors={0: (255,255,0),24: (255,0,255),39: (0,255,255), 56: (255,0,0), 60: (0,255,0),67:(0,0,255)}

        # Initialize the CvBridge class
        self.bridge = CvBridge()

        # Setup publisher function
        self.detect_pub = rospy.Publisher("/yolo/detections", Detection2DArray, queue_size=10)        

        # Setup callback function
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)        


        rospy.spin()

    def image_callback(self, img_msg):
        # Convert the ROS Image message to a CV2 Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))    
            return    
        cv_image=cv2.rotate(cv_image,cv2.ROTATE_90_CLOCKWISE)
        # frame = self.applyTransform(cv_image)
        im2 = cv_image[..., ::-1]  # OpenCV image (BGR to RGB)

        # Inference
        results = self.model([im2], size=640) # batch of images
        res_msg=Detection2DArray()
        res_msg.header=img_msg.header
        for det_ in results.xyxy[0]:
            d2=det_.cpu().numpy() # convert to numpy array
            
            cls_num=int(d2[-1])
            # Is this one of the classes we want to show?
            if cls_num in self.class_colors and d2[-2]>0.5:
                start_pt=(int(d2[0]),int(d2[1]))
                end_pt=(int(d2[2]),int(d2[3]))
                cv_image=cv2.rectangle(cv_image, start_pt, end_pt, self.class_colors[cls_num],2)
                text_pt=(int(d2[0]),int(d2[1])-5)
                cv_image=cv2.putText(cv_image,results.names[cls_num]+", %0.2f"%(d2[-2]),text_pt,cv2.FONT_HERSHEY_SIMPLEX,0.5,self.class_colors[cls_num])

        cv2.imshow("img",cv_image)
        cv2.waitKey(1)
            
if __name__ == '__main__':
    IT=yolov5()
 
