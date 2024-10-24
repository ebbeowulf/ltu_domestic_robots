#!/usr/bin/env python

import rospy
import argparse
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from stretch_srvs.srv import GetCluster, GetClusterRequest, GetClusterResponse, MoveArm, MoveArmResponse, MoveArmRequest
import numpy as np
import pdb

def call_cluster_service(main_query, llm_query, num_points):
    # Wait for the service to be available
    rospy.wait_for_service('/get_top1_cluster')
    
    try:
        # Create a service proxy
        get_cluster = rospy.ServiceProxy('/get_top1_cluster', DynamicCluster)
        
        # Prepare the request
        response = get_cluster(main_query=main_query, llm_query=llm_query, num_points=num_points)
        
        if len(response.pts)>0:
            rospy.loginfo("Service call succeeded")
            pts=np.array([ [pt.x, pt.y, pt.z] for pt in response.pts ])
            return pts  # Return the points from the service
        else:
            rospy.logwarn("Service call failed: %s", response.message)
            return None
        
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", str(e))
        return None

# def move_arm(centroid, max_height):
#     # Wait for the service to be available
#     rospy.wait_for_service('/funmap/move_arm')
    
#     try:
#         # Create a service proxy
#         move_arm = rospy.ServiceProxy('/funmap/move_arm', MoveArm)
        
#         # Prepare the request
#         response = move_arm(main_query=main_query, llm_query=llm_query, num_points=num_points)
        
#         if len(response.pts)>0:
#             rospy.loginfo("Service call succeeded")
#             pts=np.array([ [pt.x, pt.y, pt.z] for pt in response.pts ])
#             return pts  # Return the points from the service
#         else:
#             rospy.logwarn("Service call failed: %s", response.message)
#             return None
        
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call failed: %s", str(e))
#         return None

def publish_centroid(centroid, max_height):
    # Create a publisher to /clicked_point
    point_pub = rospy.Publisher('/clicked_point', PointStamped, queue_size=10)
    
    # Wait for a moment to make sure the publisher is ready
    rospy.sleep(1)
    
    # Create the PointStamped message for the centroid
    point_msg = PointStamped()
    
    # Fill the header with current time and frame id
    point_msg.header = Header()
    point_msg.header.stamp = rospy.Time.now()
    point_msg.header.frame_id = 'map'  # Set appropriate frame ID
    
    # Set the x, y, z coordinates of the centroid
    point_msg.point.x = centroid[0]
    point_msg.point.y = centroid[1]
    point_msg.point.z = max_height+0.2
    
    # Publish the centroid
    point_pub.publish(point_msg)
    rospy.loginfo(f"Published centroid: ({centroid[0]}, {centroid[1]}, {centroid[2]})")

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('cluster_centroid_publisher')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Call /get_top1_cluster service with custom queries")
    parser.add_argument('--main_query', type=str, required=True, help="Main query for the service")
    parser.add_argument('--num_points', type=int, default=10, help="Number of points to request")

    args = parser.parse_args()

    # Call the service to get points based on terminal parameters
    points = call_cluster_service(args.main_query, args.num_points)
    
    if points is not None:
        # If points are retrieved successfully, calculate and publish the centroid
        centroid = points.mean(0)
        max_height=points[:,2].max()
        print(centroid)
        publish_centroid(centroid, max_height)
    
    # rospy.spin()

