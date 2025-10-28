#!/usr/bin/env python3

import sys
import time
from math import atan2, sqrt, pi
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy 
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped, Transform, Twist
from geometry_msgs.msg import PoseStamped 
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 


TARGET_CLASSES = ["keyboard", "bottle", "book"]
CONF_THRESHOLD = 0.4
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 3.0
OFFSET_M = 0.6  
CAM_FRAME       = 'camera_color_optical_frame'
BASE_FRAME      = 'base_link'


def transform_to_matrix(T: TransformStamped):
    tx, ty, tz = T.transform.translation.x, T.transform.translation.y, T.transform.translation.z
    qx, qy, qz, qw = T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w
    R = quaternion_matrix((qx, qy, qz, qw))
    R[0:3, 3] = [tx, ty, tz]
    return R


def backproject_to_camera(u, v, Z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z, 1.0])


class FrameListener(Node):
    def __init__(self):
        super().__init__('detect_and_align_yolo')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.K = None
        self.depth = None
        self.color = None

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_cb, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_cb, 10)

    def camera_info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("Camera intrinsics received.")

    def depth_cb(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if msg.encoding == '16UC1':
            d = d.astype(np.float32) / 1000.0
        self.depth = d

    def color_cb(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def lookup_T(self, target_frame, source_frame, stamp=None):
        try:
            if stamp is None:
                stamp = Time()
            return self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed {source_frame}->{target_frame}: {ex}")
            return None


# ---------- Motion Controller (cmd_vel) ----------
class AlignToObject:
    def __init__(self, node: FrameListener, offset=OFFSET_M):
        self.node = node
        self.offset = offset
        self.cmd_pub = self.node.create_publisher(Twist, '/stretch/cmd_vel', 10)

    def compute_difference(self, p_obj_base):
        """
        Compute how far and how much to rotate the robot
        so it stops OFFSET_M meters behind the object,
        facing it directly.
        """
        dx, dy = p_obj_base[0], p_obj_base[1]
        phi_line = atan2(dy, dx)          

        # FIXED SIGN: object’s -Y should point back toward robot
        yaw_obj = phi_line - pi / 2.0

        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw_obj)
        R_obj = quaternion_matrix((qx, qy, qz, qw))

        P_obj = np.array([dx, dy, 0.0, 1.0])
        # Use w=0 for direction vector
        P_dash = np.array([0.0, -self.offset, 0.0, 0.0])
        P_goal = P_obj + R_obj @ P_dash

        base_x, base_y = P_goal[0], P_goal[1]
        phi = atan2(base_y, base_x)
        dist = sqrt(base_x**2 + base_y**2)
        _, _, z_rot_obj = euler_from_quaternion([qx, qy, qz, qw])
        z_rot_base = -phi + z_rot_obj + pi

        self.node.get_logger().info(
            f"[compute_difference] phi={phi:.3f} rad, dist={dist:.3f} m, z_rot_base={z_rot_base:.3f} rad"
        )
        return phi, dist, z_rot_base, P_goal

    # --- Motion helpers using cmd_vel ---
    def send_turn(self, yaw_rad, speed=0.3):
        twist = Twist()
        direction = 1.0 if yaw_rad > 0 else -1.0
        duration = abs(yaw_rad) / speed
        t_start = time.time()
        while time.time() - t_start < duration:
            twist.angular.z = speed * direction
            self.cmd_pub.publish(twist)
            time.sleep(0.05)
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.node.get_logger().info("Turn complete")

    def send_drive(self, dist, speed=0.1):
        twist = Twist()
        direction = 1.0 if dist > 0 else -1.0
        duration = abs(dist) / speed
        t_start = time.time()
        while time.time() - t_start < duration:
            twist.linear.x = speed * direction
            self.cmd_pub.publish(twist)
            time.sleep(0.05)
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)
        self.node.get_logger().info("Drive complete")

    def run_alignment(self, p_obj_base):
        phi, dist, z_rot_base, _ = self.compute_difference(p_obj_base)

        choice = input("\nMove toward object? [y/n]: ").strip().lower()
        if choice != 'y':
            self.node.get_logger().info("Movement canceled by user.")
            return

        self.send_turn(phi)
        self.send_drive(dist)
        self.send_turn(z_rot_base)
# ----------------------------------------------------


# ---------- Detection Node ----------
class DetectAlignNode(FrameListener):
    def __init__(self):
        super().__init__()
        self.get_logger().info("Loading YOLOv5 model…")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = CONF_THRESHOLD
        self.get_logger().info("YOLOv5 ready.")
        self.marker_pub = self.create_publisher(Marker, '/detected_object', 10)
        self.align = AlignToObject(self, offset=OFFSET_M)
        pose_qos = QoSProfile(depth=1)
        pose_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL 
        self.object_pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', pose_qos)
        self.get_logger().info("Publisher /detected_object_pose ready.")
        self.done = False   
        self.create_timer(0.2, self.on_timer)

    def on_timer(self):
        if self.done or self.K is None or self.depth is None or self.color is None:
            return

        frame = self.color.copy()
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        target = None
        found_class = None
        for name in TARGET_CLASSES:
            subset = detections[detections['name'] == name]
            if not subset.empty:
                target = subset
                found_class = name
                break

        if target is None:
            self.get_logger().info("No target detected yet…")
            cv2.imshow("YOLO Detections", np.squeeze(results.render()))
            cv2.waitKey(1)
            return

        det = target.iloc[0]
        xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        u, v = int((xmin + xmax) / 2), int((ymin + ymax) / 2)

        vis = np.squeeze(results.render())
        cv2.circle(vis, (u, v), 6, (0, 255, 0), -1)
        cv2.putText(vis, found_class, (u - 40, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLO Detection", vis)
        cv2.waitKey(1)

        # Depth at centroid
        h, w = self.depth.shape[:2]
        u0, v0 = max(0, u - 10), max(0, v - 10)
        u1, v1 = min(w - 1, u + 10), min(h - 1, v + 10)
        patch = self.depth[v0:v1 + 1, u0:u1 + 1]
        valid = patch[(patch >= DEPTH_MIN_M) & (patch <= DEPTH_MAX_M)]
        if valid.size == 0:
            self.get_logger().warn("No valid depth at target.")
            return

        Z = float(np.median(valid))
        P_cam = backproject_to_camera(u, v, Z, self.K)
        T_base_cam = self.lookup_T(BASE_FRAME, CAM_FRAME, Time())
        if T_base_cam is None:
            return

        H = transform_to_matrix(T_base_cam)
        P_base = H @ P_cam
        self.publish_marker_base(P_base[0], P_base[1])
        self.publish_object_pose(P_base[0], P_base[1], P_base[2])
        self.align.run_alignment(P_base)
        self.get_logger().info(f"Alignment done using YOLO for '{found_class}'.")
        self.done = True

    def publish_object_pose(self, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0
        self.object_pose_pub.publish(pose)
        self.get_logger().info(f"Published object pose: x={x:.2f}, y={y:.2f}, z={z:.2f}")


    def publish_marker_base(self, x, y):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "detected_object"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.0, 1.0
        self.marker_pub.publish(m)


# ---------- Main ----------
def main():
    time.sleep(10)
    rclpy.init()
    node = DetectAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
