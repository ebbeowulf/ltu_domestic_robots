# stretch_commander

ROS package for controlling the Hello Robot Stretch RE2

## Overview

This package includes scripts and functionalities to control the Hello Robot Stretch RE2 for mapping, object detection, and manipulation tasks.

### 1. Manipulation Module

The `StretchManipulation` module provides functionalities to control the manipulation actions of the Hello Robot Stretch RE2. It includes methods for triggering the homing sequence, controlling joints, and performing various manipulation actions.

**Class Methods:**

- `trigger_home_the_robot()`
- `send_joint_goals(joint_names: List[str], goal_values: List[float])`
- Gripper Actions
  - `gripper_close()`
  - `gripper_open()`
- Arm Actions
  - `arm_extend()`
  - `arm_fold()`
  - `arm_up()`
  - `arm_down()`
- Wrist Actions
  - `wrist_down()`
  - `wrist_up()`
  - `wrist_out()`
  - `wrist_in()`
- Movement Sequences
  - `look_for_shirts(pos: int)`

**Joint Limits and Names for the Hello Robot Stretch RE2:**

```plaintext
Names:
[Arm] joint_lift, wrist_extension = (joint_arm_l0 + joint_arm_l1 + joint_arm_l2 + joint_arm_l3)
[Gripper] joint_gripper_finger_left OR joint_gripper_finger_right
[Head] joint_head_pan, joint_head_tilt
[Wrist] joint_wrist_pitch, joint_wrist_yaw, joint_wrist_roll
[Base] joint_left_wheel, joint_right_wheel

Limits:
joint_lift: {0.15m - 1.10m}
wrist_extension: {0.00m - 0.50m}
joint_wrist_yaw: {-1.75rad - 4.00rad}
joint_head_pan: {-2.80rad - 2.90rad}
joint_head_tilt: {-1.60rad - 0.40rad}
joint_gripper_finger_left: {-0.35rad - 0.165rad}

Relative joints:
translate_mobile_base: No lower or upper limit. Defined by a step size in meters
rotate_mobile_base:    No lower or upper limit. Defined by a step size in radians
```

**Dependencies:**

- Required ROS packages: control_msgs, std_srvs, trajectory_msgs

**Notes:**

- The state machine script (`commander.py`) assumes specific poses and actions for demonstration purposes in the testing environment. These may be different in another environment.
- The provided joint limits are crucial for safe operation and to avoid damaging the robot.

### 2. Navigation Module

The `StretchNavigation` module provides functionalities for controlling the navigation of the Hello Robot Stretch RE2. It includes methods for re-locating the robot, scanning the surrounding area, moving to specific positions, and picking up objects.

**Class Methods:**

- `target_point_callback(msg: PointStamped)`
  - Update the target point when received from the `/target_point` topic.
- `trigger_global_localization()`
  - Re-locate the robot in the pre-loaded map.
- `trigger_head_scan()`
  - Scan the surrounding area using the RealSense depth camera.
- `get_quaternion(theta)`
  - Convert an angle in radians to a quaternion for use with MoveBaseGoal.
- `done_callback(status, result)`
  - Callback for when the goal is reached.
- `go_to_xya(x, y, theta, max_retries=3)`
  - Move the robot to a specified position (x, y, theta) using the MoveBase Action Server.
- `pick_up_at_xyz(x, y, z, using_service=True)`
  - Pick up an object at the specified position (x, y, z). Can use either the MoveArm service or the FunMap clicked_point topic.

**Dependencies:**

- Required ROS packages: geometry_msgs, move_base_msgs, std_srvs, stretch_srvs, tf

**Notes:**

- Ensure that the robot is correctly calibrated and the environment mapped before using navigation actions.
- Adjust the `z_offset` value in the `pick_up_at_xyz` method as needed for proper object manipulation. 0.01m height is the minimum required.

### 3. Perception Module

The `StretchPerception` module provides functionalities for perception tasks on the Hello Robot Stretch RE2. It includes methods for processing bounding box information, point cloud data, and object detection. Additionally, it supports transforming points between frames, triggering YOLO scans, and filtering and clustering points.

**Class Methods:**

- `trigger_yolo()`
  - Trigger the YOLO scan by publishing a `Bool` message to the `/trigger_yolo/` topic.
- `bounding_box_callback(boxes)`
  - Callback function for processing bounding box information from the `/yolo/results` topic.
- `point_cloud_callback(pc_data)`
  - Callback function for processing point cloud data from the `/camera_throttled/depth/color/points` topic.
- `filter_points(points)`
  - Filter and return the highest point within a specified threshold, considering floor height and maximum T-shirt height.
- `cluster_points(point_array)`
  - Cluster points within a specified threshold and return the average point for each cluster.
- `find_distance(point1, point2)`
  - Calculate and return the Euclidean distance between two points.
- `find_average(current_point, point_arr)`
  - Calculate and return the average point from an array of points.

**Dependencies:**

- Required ROS packages: geometry_msgs, sensor_msgs, std_msgs, vision_msgs, tf2_geometry_msgs, tf2_ros

**Notes:**

- Ensure that the YOLO node is properly configured and running before triggering the YOLO scan.
- Adjust the threshold values and height constraints as needed for accurate point filtering and clustering.

### 4. Object Detection Node

The `object_detection_node` is a ROS node responsible for performing object detection using the YOLOv8 model. It subscribes to an image topic, triggers object detection when a trigger message is received, and publishes the results in the form of bounding boxes.

**ImageProcessingNode Class Methods:**

1. **Attributes:**
   - `bounding_box_publisher`: ROS publisher for bounding box results.
   - `trigger`: A boolean flag indicating whether to trigger object detection.
   - Subscribers:
     - `/trigger_yolo/`: Triggers object detection when a boolean message is received.
     - `/camera/color/image_raw`: Subscribes to the color image topic for object detection.

2. **Methods:**
   - `trigger_callback(msg)`: Callback function for the trigger message. Sets the `trigger` flag based on the received boolean value.
   - `image_callback(msg)`: Callback function for processing image messages and performing object detection.
      - Converts ROS image messages to OpenCV images.
      - Performs object detection using the YOLOv8 model.
      - Publishes bounding box results to the `/yolo/results` topic.

**Dependencies:**

- YOLOv8 from Ultralytics
- CvBridge
- OpenCV

**Notes:**

- Verify that YOLOv8 model path is specified, and the model is loaded during node initialization.
- Bounding box results are published as `Detection2DArray` messages on the `/yolo/results` topic.
- Adjust the image topic, model path, and other parameters based on the specific ROS setup and YOLO model configuration.

## Usage

Clone this repository into the robot:

```bash
git clone https://github.com/Aeolus96/stretch_commander.git
```

Commands to start everything on the robot (using ssh on robot):

```bash
roslaunch stretch_funmap mapping.launch rviz:=false
roslaunch synchronized_throttle synchronize_stretch.launch
roslaunch synchronized_throttle generate_pcloud.launch
```

Calibrate the robot motors (run on every robot boot):

```bash
rosservice call /calibrate_the_robot "{}"
```

The `commander.py` script is a state machine demonstration for the Hello Robot Stretch RE2. It performs mapping, object detection, and object manipulation in a test environment. This is not meant for running anywhere but to serve as a guide on how to use this package.

```bash
rosrun stretch_commander commander.py
```

Custom RViz configuration for lightweight visualization:

```bash
rviz -d $(rospack find stretch_commander)/rviz/command_view.rviz
```
