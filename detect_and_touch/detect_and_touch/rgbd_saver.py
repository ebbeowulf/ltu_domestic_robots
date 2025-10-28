#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
# from tf_transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.transform import Rotation as R
import message_filters
from .camera_params import camera_params
import csv

TRACK_COLOR=True

def normalizeAngle(angle):
    while angle>=np.pi:
        angle-=2*np.pi
    while angle<-np.pi:
        angle+=2*np.pi
    return angle

class rgbd_saver(Node):
    def __init__(self, travel_params):
        # Initization of the node, name_sub
        super().__init__('rgbd_saver')

        # Setup TF listener 
        self.tf_buffer = Buffer() 
        self.listener = TransformListener(self.tf_buffer, self) 

        # Initialize the CvBridge class
        self.bridge = CvBridge()
        self.im_count=1

        self.pose_queue=[]
        self.pose_lock=Lock()
        self.pose_sub = self.create_subscription(Odometry, "/odom", self.pose_callback, 10)

        # Last image stats
        self.last_image=None
        self.travel_params=travel_params

        # Setup callback function
        # self.camera_params_sub = self.create_subscription(CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.cam_info_callback, 10)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/aligned_depth_to_color/camera_info')

        # self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.cam_info_sub], 10, 0.1
        )

        self.ts.registerCallback(self.rgbd_callback)

    def is_new_image(self, new_poseM):
        if self.last_image is None:
            return True
        
        deltaP=self.last_image['poseM'][:,3]-new_poseM[:,3]
        dist=np.sqrt((deltaP**2).sum())
        vec1=np.matmul(self.last_image['poseM'][:3,:3],[1,0,0])
        vec2=np.matmul(new_poseM[:3,:3],[1,0,0])
        deltaAngle=np.arccos(np.dot(vec1,vec2))
        if dist>self.travel_params[0] or deltaAngle>self.travel_params[1]:
            print(f"Dist {dist}, angle {deltaAngle} - new")        
            return True
        return False

    def pose_callback(self, odom_msg):
        print("pose received")
        self.pose_lock.acquire()
        self.pose_queue.append(odom_msg)
        if len(self.pose_queue)>20:
            self.pose_queue.pop(0)
        self.pose_lock.release()

    def get_pose(self, tStamp):
        # pdb.set_trace()
        
        def to_nsec(tStamp):
            return tStamp.sec*1e9+tStamp.nanosec
        
        t=to_nsec(tStamp)
        # Find the two odometry messages that bound this time
        self.pose_lock.acquire()
        top=None
        bottom=None
        for count, value in enumerate(self.pose_queue):
            if to_nsec(value.header.stamp)>t:
                top=value
                if count>0:
                    bottom=self.pose_queue[count-1]
                break
        self.pose_lock.release()
        if top is None or bottom is None:
            return None
        # Linear Interpolation between timestamps
        slopeT=(t-to_nsec(bottom.header.stamp))/(to_nsec(top.header.stamp)-to_nsec(bottom.header.stamp))
        topP=np.array([top.pose.pose.position.x,top.pose.pose.position.y,top.pose.pose.position.z])
        bottomP=np.array([bottom.pose.pose.position.x,bottom.pose.pose.position.y,bottom.pose.pose.position.z])
        pose = bottomP + slopeT*(topP-bottomP)

        # Also need to calculate orientation - interpolation between euler angles
        topQ=[top.pose.pose.orientation.x, top.pose.pose.orientation.y, top.pose.pose.orientation.z, top.pose.pose.orientation.w]
        bottomQ=[bottom.pose.pose.orientation.x, bottom.pose.pose.orientation.y, bottom.pose.pose.orientation.z, top.pose.pose.orientation.w]
        a1, b1, topYaw = R.from_quat(topQ).as_euler('xyz')
        a2, b2, bottomYaw = R.from_quat(bottomQ).as_euler('xyz')    
        
        #[a1,b1,topYaw]=tf_transformations.euler_from_quaternion(topQ)
        #[a2,b2,bottomYaw]=tf_transformations.euler_from_quaternion(bottomQ)
        # We are going to assume that the shortest delta is the direction of rotation
        deltaY=normalizeAngle(topYaw-bottomYaw)
        if deltaY>np.pi:
            deltaY=2*np.pi-deltaY
        if deltaY<-np.pi:
            deltaY=2*np.pi + deltaY
        orientation=bottomYaw+deltaY*slopeT

        rotM = np.eye(4)
        rotM[:3, :3] = R.from_rotvec(orientation * np.array([0, 0, 1])).as_matrix()
        poseM = rotM 
        #poseM=tf_transformations.rotation_matrix(orientation,(0,0,1))
        poseM[:3,3]=pose
        return poseM

    def get_camera_pose(self, header):
        if 0: # if the /map transform is correctly setup, then use tf all the way
            try:
                transform = self.tf_buffer.lookup_transform('map', header.frame_id, Time.from_msg(header.stamp))  
                trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]   

                transM = np.eye(4)
                transM[:3, 3] = trans

                rotM = np.eye(4)
                rotM[:3, :3] = R.from_quat(rot).as_matrix()

                poseM = transM @ rotM
        
                #poseM=np.matmul(tf_transformations.translation_matrix(trans),tf_transformations.quaternion_matrix(rot))
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                print("No Transform found")
                return
        else: # otherwise just get the transform to the base and then use the published odometry ... less accurate
            try:
                transform = self.tf_buffer.lookup_transform('base_link', header.frame_id, Time.from_msg(header.stamp))  
                trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]  
                
                transM = np.eye(4)
                transM[:3, 3] = trans

                rotM = np.eye(4)
                rotM[:3, :3] = R.from_quat(rot).as_matrix()     

                base_relativeM = transM @ rotM 
                
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                print("No Transform found")
                return
            odom=self.get_pose(header.stamp)
            if odom is None:
                print("Missing odometry information - skipping")
                return
            
            # Convert the ROS Image message to a CV2 Image
            poseM=np.matmul(odom,base_relativeM)
        return poseM
    
    # def rgbd_callback(self, rgb_img:Image, depth_img:Image):
    def rgbd_callback(self, rgb_img: Image, depth_img: Image, cam_info: CameraInfo):
        print("RGB-D images received")
        
        color_fName=f'color_{self.im_count:05}.png'
        depth_fName=f'depth_{self.im_count:05}.png'
        try:
            poseM=self.get_camera_pose(depth_img.header)

            if self.is_new_image(poseM):
                cv_image_rgb = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
                cv_image_depth = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding='passthrough')
                with open("poses.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)

                    # Flatten the 4x4 matrix to a 1D list of 16 elements
                    row = poseM.flatten().tolist()
                    writer.writerow(row)

                cv2.imwrite(color_fName,cv_image_rgb)
                cv2.imwrite(depth_fName,cv_image_depth)
                self.last_image={'depth': cv_image_rgb, 'rgb': cv_image_depth, 'poseM': poseM, 'time': depth_img.header.stamp}
                self.im_count+=1

        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")  
            return



            
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_travel_dist',type=float,default=0.1,help='Minimum distance the robot must travel before adding a new image to the point cloud (default = 0.1m)')
    parser.add_argument('--min_travel_angle',type=float,default=0.1,help='Minimum angle the camera must have moved before adding a new image to the point cloud (default = 0.1 rad)')
    args = parser.parse_args()
    
    rclpy.init() 

    IT=rgbd_saver([args.min_travel_dist,args.min_travel_angle])
    rclpy.spin(IT) 

    IT.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': 
    main()
