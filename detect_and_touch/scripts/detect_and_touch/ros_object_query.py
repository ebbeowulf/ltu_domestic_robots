#!/usr/bin/env python3
import rospy
import cv2
import pdb
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
import tf
import message_filters
from camera_params import camera_params
from map_utils import pcloud_from_images, create_object_clusters, calculate_iou
from std_srvs.srv import Trigger, TriggerResponse
from stretch_srvs.srv import GetCluster, GetClusterRequest, GetClusterResponse, SetInt, SetIntRequest, SetIntResponse
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

TRACK_COLOR=True

def normalizeAngle(angle):
    while angle>=np.pi:
        angle-=2*np.pi
    while angle<-np.pi:
        angle+=2*np.pi
    return angle

class multi_query_localize:
    def __init__(self, query_list, cluster_min_points, detection_threshold, travel_params, storage_dir=None, is_yolo=False):
        # Initization of the node, name_sub
        rospy.init_node('object_localize', anonymous=True)
        self.listener = tf.TransformListener()
        self.storage_dir = storage_dir
        self.is_yolo=is_yolo

        # Initialize the CvBridge class
        self.bridge = CvBridge()
        self.im_count=1

        self.pose_queue=[]
        self.pose_lock=Lock()
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        
        # Last image stats
        self.last_image=None
        self.travel_params=travel_params

        # Previously detected clusters
        self.known_objects=[]

        # Create Pcloud Data Structures
        self.pcloud_creator=None
        self.query_list=query_list
        self.pcloud=dict()
        for query in self.query_list:
            self.pcloud[query]={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
        self.cluster_min_points=cluster_min_points
        self.cluster_iou=0.5
        self.detection_threshold=detection_threshold

        # Setup callback function
        self.camera_params_sub = rospy.Subscriber('/camera_throttled/depth/camera_info', CameraInfo, self.cam_info_callback)
        self.rgb_sub = message_filters.Subscriber('/camera_throttled/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera_throttled/depth/image_rect_raw', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.rgbd_callback)

        # Setup service calls
        self.setclustersize_srv = rospy.Service('set_cluster_size', SetInt, self.set_cluster_size_service)
        self.setiou_srv = rospy.Service('set_cluster_iou_pct', SetInt, self.set_cluster_iou_pct)
        self.clear_srv = rospy.Service('clear_clouds', Trigger, self.clear_clouds_service)
        self.top1_cluster_srv = rospy.Service('get_top1_cluster', GetCluster, self.top1_cluster_service)
        self.marker_pub=rospy.Publisher('cluster_markers',MarkerArray,queue_size=5)

    def set_cluster_size_service(self, req):
        self.cluster_min_points=req.value
        print(f"Changing minimum cluster size to {self.cluster_min_points} cm2")
        return SetIntResponse()

    def set_cluster_iou_pct(self, req):
        self.cluster_iou=req.value/100.0
        print(f"Changing minimum cluster iou to {self.cluster_iou} pct")
        return SetIntResponse()

    def clear_clouds_service(self, msg):
        resp=TriggerResponse()
        resp.success=True
        for query in self.query_list:
            self.pcloud[query]={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
        self.last_image=None
        resp.message="clouds cleared"
        return resp

    # create the object clusters and filter by the number of points in each
    def get_clusters_ros(self, pcd):
        all_objects=create_object_clusters(pcd['xyz'],pcd['probs'], -1.0, self.detection_threshold, compress_clusters=False)
        objects_out=[]
        for obj in all_objects:
            if obj.size()>self.cluster_min_points:
                objects_out.append(obj)
        return objects_out

    def publish_object_markers(self, known_object_boxes):
        msg=MarkerArray()
        for idx, obj_ in enumerate(known_object_boxes):
            M=Marker()
            M.header.frame_id="map"
            M.header.stamp=self.last_image['time']
            M.type=Marker.SPHERE
            M.action=Marker.ADD
            mn_=obj_.mean(0)
            M.id=idx
            M.pose.position.x=mn_[0]
            M.pose.position.y=mn_[1]
            M.pose.position.z=mn_[2]
            M.pose.orientation.x=0.0
            M.pose.orientation.y=0.0
            M.pose.orientation.z=0.0
            M.pose.orientation.w=1.0
            M.scale.x=0.5
            M.scale.y=0.5
            M.scale.z=0.5
            M.color.a=1.0
            M.color.r=1.0
            M.color.g=0.0
            M.color.b=0.0
            msg.markers.append(M)
        self.marker_pub.publish(msg)

    # create the clusters and match them, returning any that exceed the detection threshold
    def create_and_publish_clusters(self, main_query):
        if main_query in self.query_list:
            positive_clusters=self.get_clusters_ros(self.pcloud[main_query])
        else:
            positive_clusters=[]

        # Update current object list so that markers are published correctly
        known_object_boxes=[]
        for obj_ in positive_clusters:
            known_object_boxes.append(obj_.box)

        # publish the markers
        self.publish_object_markers(known_object_boxes)

        return positive_clusters

    def top1_cluster_service(self, request:GetClusterRequest):
        resp=GetClusterResponse()
        resp.success=False

        positive_clusters=self.create_and_publish_clusters(request.main_query)

        if len(positive_clusters)==0:
            resp.message="No clusters found"
            return resp

        if request.criterion=='mean' or request.criterion=='max' or request.criterion=='pcount':
            pos_likelihoods=[ obj_.prob_stats[request.criterion] for obj_ in positive_clusters]
        else:
            resp.message="Criterion not recognized"
            return resp

        whichC=np.argmax(pos_likelihoods)
        for idx in range(request.num_points):
            fPx=positive_clusters[whichC].farthestP[idx]
            pt=Point()
            pt.x=positive_clusters[whichC].pts[fPx][0]
            pt.y=positive_clusters[whichC].pts[fPx][1]
            pt.z=positive_clusters[whichC].pts[fPx][2]
            resp.pts.append(pt)
        resp.bbox3d=np.hstack((positive_clusters[whichC].box[0],positive_clusters[whichC].box[1])).tolist()
        return resp
    
    def cam_info_callback(self, cam_info):
        # print("Cam info received")
        self.params=camera_params(cam_info.height, cam_info.width, cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5], np.identity(4,dtype=float))
        self.pcloud_creator=pcloud_from_images(self.params,self.is_yolo)
        self.camera_params_sub.unregister()

    def pose_callback(self, odom_msg):
        # print("pose received")
        self.pose_lock.acquire()
        self.pose_queue.append(odom_msg)
        if len(self.pose_queue)>20:
            self.pose_queue.pop(0)
        self.pose_lock.release()

    def get_pose(self, tStamp):
        t=tStamp.to_nsec()
        self.pose_lock.acquire()
        top=None
        bottom=None
        for count, value in enumerate(self.pose_queue):
            if value.header.stamp.to_nsec()>t:
                top=value
                if count>0:
                    bottom=self.pose_queue[count-1]
                break
        self.pose_lock.release()
        if top is None or bottom is None:
            return None
        # Linear Interpolation between timestamps
        slopeT=(t-bottom.header.stamp.to_nsec())/(top.header.stamp.to_nsec()-bottom.header.stamp.to_nsec())
        topP=np.array([top.pose.pose.position.x,top.pose.pose.position.y,top.pose.pose.position.z])
        bottomP=np.array([bottom.pose.pose.position.x,bottom.pose.pose.position.y,bottom.pose.pose.position.z])
        pose = bottomP + slopeT*(topP-bottomP)

        # Also need to calculate orientation - interpolation between euler angles
        topQ=[top.pose.pose.orientation.x, top.pose.pose.orientation.y, top.pose.pose.orientation.z, top.pose.pose.orientation.w]
        bottomQ=[bottom.pose.pose.orientation.x, bottom.pose.pose.orientation.y, bottom.pose.pose.orientation.z, top.pose.pose.orientation.w]
        [a1,b1,topYaw]=tf.transformations.euler_from_quaternion(topQ)
        [a2,b2,bottomYaw]=tf.transformations.euler_from_quaternion(bottomQ)
        # We are going to assume that the shortest delta is the direction of rotation
        deltaY=normalizeAngle(topYaw-bottomYaw)
        if deltaY>np.pi:
            deltaY=2*np.pi-deltaY
        if deltaY<-np.pi:
            deltaY=2*np.pi + deltaY
        orientation=bottomYaw+deltaY*slopeT
        poseM=tf.transformations.rotation_matrix(orientation,(0,0,1))
        poseM[:3,3]=pose
        return poseM

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

    def rgbd_callback(self, rgb_img:Image, depth_img:Image):
        print("RGB-D images received")
        if self.pcloud_creator is None:
            return
        if 1: # if the /map transform is correctly setup, then use tf all the way
            try:
                (trans,rot) = self.listener.lookupTransform('/map',depth_img.header.frame_id,depth_img.header.stamp)
                poseM=np.matmul(tf.transformations.translation_matrix(trans),tf.transformations.quaternion_matrix(rot))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("No Transform found")
                return
        else: # otherwise just get the transform to the base and then use the published odometry ... less accurate
            try:
                (trans,rot) = self.listener.lookupTransform('/base_link',depth_img.header.frame_id,depth_img.header.stamp)
                base_relativeM=np.matmul(tf.transformations.translation_matrix(trans),tf.transformations.quaternion_matrix(rot))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("No Transform found")
                return
            odom=self.get_pose(rgb_img.header.stamp)
            if odom is None:
                print("Missing odometry information - skipping")
                return
            
            # Convert the ROS Image message to a CV2 Image
            poseM=np.matmul(odom,base_relativeM)
        
        # Only update the point clouds if this image is significantly different pose from last image
        if self.is_new_image(poseM):
            try:
                cv_image_rgb = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
                cv_image_depth = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding='passthrough')
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))    
                return
            self.update_point_cloud(cv_image_rgb, cv_image_depth, poseM, rgb_img.header)
    
    def update_point_cloud(self, rgb, depth, poseM, header):
        self.last_image={'depth': rgb, 'rgb': depth, 'poseM': poseM, 'time': header.stamp}
        uid_key="%0.3f_%0.3f_%0.3f"%(poseM[0][3],poseM[1][3],poseM[2][3])
        self.pcloud_creator.load_image(rgb, depth, poseM, uid_key=uid_key)
        if self.storage_dir is not None:
            cv2.imwrite(self.storage_dir+"/rgb"+uid_key+".png",rgb)
            cv2.imwrite(self.storage_dir+"/depth"+uid_key+".png",rgb)
        results=self.pcloud_creator.multi_prompt_process(self.query_list, self.detection_threshold, rotate90=True)
        for query in self.query_list:
            if results[query]['xyz'].shape[0]>0:
                self.pcloud[query]['xyz']=np.vstack((self.pcloud[query]['xyz'],results[query]['xyz']))
                self.pcloud[query]['probs']=np.hstack((self.pcloud[query]['probs'],results[query]['probs']))
                if TRACK_COLOR:
                    self.pcloud[query]['rgb']=np.vstack((self.pcloud[query]['rgb'],results[query]['rgb']))
            print(f"Adding {query}:{results[query]['xyz'].shape[0]}.... Total:{self.pcloud[query]['xyz'].shape[0]}")

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', type=str, nargs='*', default=None,
                    help='Set of target queries to build point clouds for - must include at least one')
    parser.add_argument('--num_points',type=int,default=200, help='number of points per cluster')
    parser.add_argument('--detection_threshold',type=float,default=0.5, help='fixed detection threshold')
    parser.add_argument('--min_travel_dist',type=float,default=0.1,help='Minimum distance the robot must travel before adding a new image to the point cloud (default = 0.1m)')
    parser.add_argument('--min_travel_angle',type=float,default=0.1,help='Minimum angle the camera must have moved before adding a new image to the point cloud (default = 0.1 rad)')
    parser.add_argument('--storage_dir',type=str,default=None,help='A place to store intermediate files - but only if specified (default = None)')
    parser.add_argument('--use_yolo', dest='use_yolo', action='store_true')
    parser.set_defaults(use_cc=False)
    args = parser.parse_args()

    if args.queries is None:
        print("Must include at least one target query to execute")
        import sys
        sys.exit(-1)

    IT=multi_query_localize(args.queries,
                          args.num_points,
                          args.detection_threshold,
                          [args.min_travel_dist,args.min_travel_angle],
                          args.storage_dir,
                          args.use_yolo)
    rospy.spin()
