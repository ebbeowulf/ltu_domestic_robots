#Adapted from code written by Devson Butani, Annalia Schoenherr, Kaushik Mehta
#https://github.com/Aeolus96/stretch_commander

import ctypes
import math
import struct

import cv2
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Header
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray, Detection2D, Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray


class StretchPerception:
    def __init__(self):
        # Subscribers
        self.bbox_topic = "/yolo/results"
        self.camera_image_topic = "/camera_throttled/depth/color/points"
        self.bbox_sub = rospy.Subscriber(self.bbox_topic, Detection2DArray, self.bounding_box_callback)
        self.image_sub = rospy.Subscriber(self.camera_image_topic, PointCloud2, self.point_cloud_callback)
        # self.stretch_sub=rospy.Subscriber("/tf/map", tf, self.stretch_location_callback)

        # Publishers
        self.trigger_scan_topic = "/trigger_yolo/"
        self.target_point_topic = "/target_point"
        self.point_pub = rospy.Publisher(self.target_point_topic, PointStamped, queue_size=5)
        self.trigger_yolo_pub = rospy.Publisher(self.trigger_scan_topic, Bool, queue_size=5)
        self.marker_pub = rospy.Publisher("/target_marker", Marker, queue_size=5)
        # bounding box testing:
        self.test_pub = rospy.Publisher(self.bbox_topic, Detection2DArray, queue_size=5)

        # Initialize variables and buffers:
        self.all_raw_bbox_points = []
        self.raw_bbox_points = []
        self.final_point = PointStamped()
        self.detections = []  # holds detections from /yolo/results

        self.marker_array_msg = MarkerArray()
        self.marker = Marker()

        self.marker.header.frame_id = "map"
        self.marker.type = 2
        self.marker.id = 0
        # self.marker.action = self.marker.DELETEALL
        # Set the scale of the marker
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1

        # Set the color
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

        # self.bbox_time = rospy.Time()

        self.detected_objects = False

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def trigger_yolo(self):
        msg = Bool()
        msg.data = True
        self.trigger_yolo_pub.publish(msg)

    ################# BOUNDING BOX CALLBACK FUNCTIONS##########################

    def bounding_box_callback(self, boxes):
        print("bounding box callback reached")  # MODIFIED TO REFLECT THE CORRECT TOPICS
        # self.bbox_sub = rospy.Subscriber('/Camera
        rospy.loginfo("bounding box callback reached")
        for detection in boxes.detections:
            self.detections.append(detection)
            print("Bounding box detection appended to the list of detections")

    ################# POINT CLOUD CALLBACK FUNCTIONS###########################

    # Extract bounding box dimensions and convert
    def point_cloud_callback(self, pc_data):
        # print("point cloud callback reached")
        for detection in self.detections:
            print("detection from bbox callback: ", detection)
            # for testing:
            # print(detection)

            # access the bounding box points
            bbox = detection.bbox

            width = bbox.size_x
            height = bbox.size_y
            bbox_center_x = bbox.center.x
            bbox_center_y = bbox.center.y
            bbox_time = detection.header.stamp

            xmin = bbox_center_x - width / 2
            xmax = bbox_center_x + width / 2
            ymin = bbox_center_y - height / 2
            ymax = bbox_center_y + height / 2

            # print("Bounding box center: ", bbox_center_x, bbox_center_y)

            D3_bbox_points = []

            # bbox pixels to D3 points

            row = 0
            col = 0

            for row in range(int(bbox_center_y-10), int(bbox_center_y+10)):
                for col in range(int(bbox_center_x-10), int(bbox_center_x+10)):
                    index = (row * pc_data.row_step) + (col * pc_data.point_step)
                    # print("Index: ", index)

                    # Get the XYZ points [meters]

                    # print("X point converted. X coordinate: ", X)
                    # print("Y point converted. Y coordinate: ", Y)
                    # print("Z point converted. Z coordinate: ", Z)
                    try:    
                        (X, Y, Z, rgb) = struct.unpack_from("fffl", pc_data.data, offset=index)

                        D3_point = PointStamped()

                        D3_point.header.frame_id = "camera_color_optical_frame"
                        D3_point.header.stamp = detection.header.stamp
                        
                        D3_point.point.x = X
                        D3_point.point.y = Y
                        D3_point.point.z = Z
                        
                        D3_bbox_points.append(D3_point)
                    except e as error:
                        print("continuing")
                    # if row == int(bbox_center_y) and col == int(bbox_center_x):
                    #     (X, Y, Z, rgb) = struct.unpack_from("fffl", pc_data.data, offset=index)
                    #     # create point stamped object to use when transforming points:
                    #     D3_point = PointStamped()

                    #     # frame will eventually be 'usb_cam/image_raw'
                    #     D3_point.header.frame_id = "camera_color_optical_frame"
                    #     D3_point.header.stamp = detection.header.stamp



                    #     # Append to array of D3 points in camera frame:
                    #     D3_bbox_points.append(D3_point)
                    #     print("Center Point: ", D3_point)



            # Transform 3D points to map frame
            # transformation info:
            try:
                transform = self.tfBuffer.lookup_transform_full(
                    target_frame="map",
                    target_time=bbox_time,
                    source_frame="camera_color_optical_frame",
                    source_time=bbox_time,
                    fixed_frame="map",  # VERIFY THIS IF THIS IS CORRECT
                    timeout=rospy.Duration(10),
                )
                print("Transform created")


                transformed_points = [
                    tf2_geometry_msgs.do_transform_point(point, transform) for point in D3_bbox_points
                ]

                # Used to be Z height sorting and filtering clusters into a single point, since modified to return center of bbox
                # if self.filter_points(transformed_points):
                if transformed_points[0]:
                    # These are the points that will be published
                    self.detected_objects = True
                    self.final_point = transformed_points[0]
                    print("Final point to publish: ", self.final_point)
                    self.point_pub.publish(self.final_point)

                    self.marker.pose.position.x = transformed_points[0].point.x
                    self.marker.pose.position.y = transformed_points[0].point.y
                    self.marker.pose.position.z = transformed_points[0].point.z
                    self.marker.header.stamp = rospy.Time.now()
                    self.marker_array_msg.markers.append(self.marker)
                    self.marker_pub.publish(self.marker)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as error:
                print("error making transformation: ", error)

        self.final_point = PointStamped()
        self.detections = []

    # ended up not using this:
    """
    def filter_points(self, points):
        print("filtering points")
        # filters all D3 points within one bounding box, and returns the highest point

        # assuming floor height is 0.0 Confirm with team.
        # to use to filter out floor points
        floor_height = 0.0

        # making max tshirt height 5 inches off ground. Confirm with team
        max_tshirt_height = 0.5

        # will return the highest point in the filtered array
        final_point = PointStamped()
        # header must be changed to correct header
        final_point.header.frame_id = "camera_color_optical_frame"
        final_point.point.x = 0.0
        final_point.point.y = 0.0
        final_point.point.z = 0.0

        for point in points:
            # (if the point is not the floor and under 5 inches)
            if point.point.z > floor_height and point.point.z < max_tshirt_height:
                if point.point.z > final_point.point.z:
                    final_point = point

        if not (final_point.point.z == 0.0 and final_point.point.x == 0.0 and final_point.point.y == 0.0):
            self.final_point = final_point
            print("Point within threshold detected: ", final_point)
            return True
        else:
            print("Did not find a point within the threshold")
            self.detected_objects = False
            return False

    def cluster_points(self, point_array):
        # print("Array passed into cluster_points():", point_array)
        point_arr = point_array
        final_point = PointStamped()

        final_point.header = "map"
        final_point.point.x = 0.0
        final_point.point.y = 0.0
        final_point.point.z = 0.0

        # THRESHOLD TO BE MODIFIED WITH ACTUAL DATA
        threshold = 5

        # go through all points, and combine any that are within the threshold of eachother:

        for i in range(len(point_arr)):
            # point to compare:
            current_point = point_arr[i]
            # array will hold all points that are close to eachother:
            points_to_merge = []
            locations = []
            # go through the rest of the points:
            for j in range(i + 1, len(point_arr)):
                difference = self.find_distance(current_point, point_arr[j])
                if difference < threshold:
                    points_to_merge.append(point_arr[j])
                    locations.append(j)

            if len(points_to_merge) > 0:
                updated_point = self.find_average(current_point, points_to_merge)
                for index in locations:
                    point_arr[index].pop(index)
                point_arr.append(updated_point)

        # print("point_arr: ", point_arr)

        final_point = self.find_average(point_arr[0], point_arr)
        self.final_point = final_point

    def find_distance(self, point1, point2):
        x1, y1, z1 = point1.point.x, point1.point.y, point1.point.z
        x2, y2, z2 = point2.point.x, point2.point.y, point2.point.z

        # differences:
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        # distance formula:
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        return distance

    def find_average(self, current_point, point_arr):
        # print("From inside 'find_average': ")
        # print("current/first point length: ", len(current_point))
        # print("length of rest of array: ", len(point_arr))

        # print("current/first point: ", current_point)
        # print("rest of array: ", point_arr)

        # sums
        sum_x = current_point.point.x
        sum_y = current_point.point.y
        sum_z = current_point.point.z

        for point in point_arr:
            sum_x += point.point.x
            sum_y += point.point.y
            sum_z += point.point.z

        avg_x = sum_x / len(self.points)
        avg_y = sum_y / len(self.points)
        avg_z = sum_z / len(self.points)

        avg_point = PointStamped(point.header, self.Point(avg_x, avg_y, avg_z))

        return avg_point
    """

    def publish_test_box(self):
        # create a bounding box for testing:
        detection_array_msg = Detection2DArray()

        x1 = 260
        y1 = 540
        x2 = 460
        y2 = 740

        detection_msg = Detection2D()
        detection_msg.header.stamp = rospy.Time.now()

        detection_msg.bbox.size_x = x2 - x1
        detection_msg.bbox.size_y = y2 - y1

        detection_msg.bbox.center.x = x1 + detection_msg.bbox.size_x / 2
        detection_msg.bbox.center.y = y1 + detection_msg.bbox.size_y / 2
        detection_array_msg.detections.append(detection_msg)

        self.test_pub.publish(detection_array_msg)
