import sqlite3, os
import cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

# ---- Bag and output ----
bag_path = "experiment_run2/experiment_run2_0.db3"
out_dir = "exported_images"
os.makedirs(out_dir, exist_ok=True)                    # Capture image at each tilt level (after panning) 

# ---- Head sweep grid ----
TILT_LEVELS = [0.300, -0.075, -0.450, -0.825, -1.200]
PAN_VALUES  = [-2.600, -1.733, -0.867, 0.000, 0.867, 1.733, 2.600]
TOLERANCE = 0.05

# ---- Connect to DB ----
conn = sqlite3.connect(bag_path)
cursor = conn.cursor()
cursor.execute("SELECT id, name, type FROM topics")
topic_map = {name: tid for tid, name, _ in cursor.fetchall()}

bridge = CvBridge()

# ---- Get image topic ----
if '/camera/color/image_raw' not in topic_map:
    raise RuntimeError("Bag does not contain /camera/color/image_raw")
image_topic_id = topic_map['/camera/color/image_raw']

# ---- If joint_states present, use selective export ----
if '/joint_states' in topic_map:
    print("Found /joint_states → exporting one image per head position")
    joint_topic_id = topic_map['/joint_states']

    # Load joint states
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (joint_topic_id,))
    joint_msgs = cursor.fetchall()
    joint_history = []
    for ts, data in joint_msgs:
        msg = deserialize_message(data, JointState)
        if 'joint_head_pan' in msg.name and 'joint_head_tilt' in msg.name:
            pan = msg.position[msg.name.index('joint_head_pan')]
            tilt = msg.position[msg.name.index('joint_head_tilt')]
            joint_history.append((ts, pan, tilt))

    def get_latest_joint(ts):
        candidates = [j for j in joint_history if j[0] <= ts]
        return max(candidates, key=lambda j: j[0]) if candidates else None

    saved_positions = set()
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (image_topic_id,))
    rows = cursor.fetchall()

    for ts, data in rows:
        joint = get_latest_joint(ts)
        if not joint:
            continue
        _, pan, tilt = joint
        nearest_pan = min(PAN_VALUES, key=lambda p: abs(p - pan))
        nearest_tilt = min(TILT_LEVELS, key=lambda t: abs(t - tilt))
        if abs(nearest_pan - pan) < TOLERANCE and abs(nearest_tilt - tilt) < TOLERANCE:
            key = (nearest_pan, nearest_tilt)
            if key not in saved_positions:
                msg = deserialize_message(data, Image)
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

                # Build a safe filename with valid .jpg extension
                fname = f"pan{nearest_pan:+.2f}_tilt{nearest_tilt:+.2f}.jpg"
                fname = fname.replace('.', 'p').replace('-', 'm').replace('+', 'p')
                if not fname.endswith(".jpg"):
                    fname += ".jpg"

                # Save the image
                cv2.imwrite(os.path.join(out_dir, fname), cv_img)
                
                #fname = f"pan{nearest_pan:+.2f}_tilt{nearest_tilt:+.2f}.jpg".replace('.', 'p').replace('-', 'm')
                #cv2.imwrite(os.path.join(out_dir, fname), cv_img)
                print(f"Saved {fname}")
                saved_positions.add(key)

    print(f"Done. Saved {len(saved_positions)} unique head positions to {out_dir}")

# ---- Otherwise, save all frames ----
else:
    print("No /joint_states found → saving all frames")
    cursor.execute("SELECT data FROM messages WHERE topic_id=?", (image_topic_id,))
    rows = cursor.fetchall()
    for i, (data,) in enumerate(rows):
        msg = deserialize_message(data, Image)
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        fname = f"frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), cv_img)
        if i % 50 == 0:
            print(f"Saved {fname}")
    print(f"Done. Saved {len(rows)} frames to {out_dir}")
