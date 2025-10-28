#!/usr/bin/env python3
# Stretch 2: Detect (CLIPSeg) + Navigate + Publish Object Pose
# - Publishes detected object position in base_link frame
# - Navigates using cmd_vel (turn, drive, turn)
# - Compatible with grasp_object.py subscriber

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
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import quaternion_matrix
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


TARGET_CLASS = "keyboard"
PROMPTS = [
    "keyboard",
    "a computer keyboard",
    "grey keyboard",
    "black keyboard",
    "wireless keyboard",
]
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 3.0
OFFSET_M = 0.50

CAM_FRAME = 'camera_color_optical_frame'
BASE_FRAME = 'base_link'




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


def largest_component(mask_bin):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (labels == largest).astype(np.uint8) * 255
    return comp



class FrameListener(Node):
    def __init__(self):
        super().__init__('detect_and_navigate_clipseg')
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

    def lookup_T(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed {source_frame}->{target_frame}: {ex}")
            return None


# -------------- Alignment --------------
class AlignToObject:
    def __init__(self, node: FrameListener, offset=OFFSET_M):
        self.node = node
        self.offset = offset
        self.cmd_pub = self.node.create_publisher(Twist, '/stretch/cmd_vel', 10)

    def compute_difference(self, p_obj_base):
        dx, dy = p_obj_base[0], p_obj_base[1]
        phi = atan2(dy, dx)
        dist = sqrt(dx**2 + dy**2)
        z_rot_base = phi
        self.node.get_logger().info(
            f"[compute_difference] phi={phi:.3f} rad, dist={dist:.3f}"
        )
        return phi, dist, z_rot_base

    def run_alignment(self, p_obj_base):
        phi, dist, z_rot_base = self.compute_difference(p_obj_base)
        # Rotate toward object
        self.turn(phi)
        # Drive forward but stop short by offset
        self.drive(dist - self.offset)
        # Face object
        self.turn(z_rot_base)
        self.node.get_logger().info("Alignment sequence complete.")

    def turn(self, yaw_rad, speed=0.3):
        twist = Twist()
        direction = 1.0 if yaw_rad > 0 else -1.0
        duration = abs(yaw_rad) / speed
        start = time.time()
        while time.time() - start < duration:
            twist.angular.z = speed * direction
            self.cmd_pub.publish(twist)
            time.sleep(0.05)
        self.cmd_pub.publish(Twist())
        self.node.get_logger().info("Turn complete")

    def drive(self, dist, speed=0.1):
        twist = Twist()
        direction = 1.0 if dist > 0 else -1.0
        duration = abs(dist) / speed
        start = time.time()
        while time.time() - start < duration:
            twist.linear.x = speed * direction
            self.cmd_pub.publish(twist)
            time.sleep(0.05)
        self.cmd_pub.publish(Twist())
        self.node.get_logger().info("Drive complete")


# -------------- Detection --------------
class DetectAlignNode(FrameListener):
    def __init__(self):
        super().__init__()
        self.get_logger().info("Loading CLIPSeg...")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.eval()
        self.get_logger().info("CLIPSeg ready.")

        # Marker + Pose publishers
        self.marker_pub = self.create_publisher(Marker, '/detected_object', 10)
        pose_qos = QoSProfile(depth=1)
        pose_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', pose_qos)
        self.get_logger().info("Publisher /detected_object_pose ready.")

        self.align = AlignToObject(self, offset=OFFSET_M)
        self.done = False
        self.create_timer(0.2, self.on_timer)

    def on_timer(self):
        if self.done or self.K is None or self.depth is None or self.color is None:
            return

        image_bgr = self.color.copy()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (384, 384))

        mask = None
        comp = None
        u = v = None
        chosen_prompt = None

        for prompt in PROMPTS:
            with torch.no_grad():
                inputs = self.processor(text=[prompt], images=image_rgb, return_tensors="pt")
                out = self.model(**inputs)
                mask_try = torch.sigmoid(out.logits)[0, 0].cpu().numpy()
            print(f"Prompt: {prompt}, mask mean={mask_try.mean():.2f}, max={mask_try.max():.2f}")
            mask_try = (mask_try * 255).astype(np.uint8)
            _, mask_bin = cv2.threshold(mask_try, 40, 255, cv2.THRESH_BINARY)
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            mask_bin = cv2.dilate(mask_bin, np.ones((5,5), np.uint8), iterations=1)
            comp_try = largest_component(mask_bin)
            if comp_try is not None:
                chosen_prompt = prompt
                mask = mask_try
                comp = comp_try
                break

        if comp is None:
            self.get_logger().info("No sufficiently large mask for target yet…")
            if mask is not None:
                cv2.imshow("CLIPSeg Mask", mask)
                cv2.waitKey(1)
            return

        # Centroid
        M = cv2.moments(comp)
        if M["m00"] <= 1e-5:
            return
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # Depth around centroid
        h, w = self.depth.shape[:2]
        u0, v0 = max(0, u - 10), max(0, v - 10)
        u1, v1 = min(w - 1, u + 10), min(h - 1, v + 10)
        patch = self.depth[v0:v1 + 1, u0:u1 + 1]
        valid = patch[(patch >= DEPTH_MIN_M) & (patch <= DEPTH_MAX_M)]
        if valid.size == 0:
            self.get_logger().warn("No valid depth at target.")
            return
        Z = float(np.median(valid))

        # Backproject to 3D and transform
        P_cam = backproject_to_camera(u, v, Z, self.K)
        T_base_cam = self.lookup_T(BASE_FRAME, CAM_FRAME)
        if T_base_cam is None:
            return
        H = transform_to_matrix(T_base_cam)
        P_base = H @ P_cam

        # Visualize and publish
        self.publish_marker_base(P_base[0], P_base[1])
        self.publish_object_pose(P_base[0], P_base[1], P_base[2])

        # Run movement
        self.align.run_alignment(P_base)

        self.get_logger().info(f"Alignment sequence sent using prompt '{chosen_prompt}'.")
        self.done = True

    def publish_object_pose(self, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
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


# ----------------------------------------------------
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
