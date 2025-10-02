#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import argparse
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from stretch_srvs.srv import GetCluster, MoveArm
import numpy as np
import pdb


class ClusterCentroidPublisher(Node):
    def __init__(self, main_query, num_points):
        super().__init__('cluster_centroid_publisher')

        # Save parameters
        self.main_query = main_query
        self.num_points = num_points

        # Publisher
        self.point_pub = self.create_publisher(PointStamped, '/clicked_point', 10)

        # Client for get_top1_cluster service
        self.cluster_client = self.create_client(GetCluster, '/get_top1_cluster')

        while not self.cluster_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_top1_cluster service...')

        # Call service once at startup
        self.call_cluster_service()

    def call_cluster_service(self):
        req = GetCluster.Request()
        req.main_query = self.main_query
        req.num_points = self.num_points
        # NOTE: llm_query was not in argparse → skipped

        future = self.cluster_client.call_async(req)
        future.add_done_callback(self.cluster_response_callback)

    def cluster_response_callback(self, future):
        try:
            response = future.result()
            if len(response.pts) > 0:
                self.get_logger().info("Service call succeeded")
                pts = np.array([[pt.x, pt.y, pt.z] for pt in response.pts])
                centroid = pts.mean(0)
                max_height = pts[:, 2].max()
                self.publish_centroid(centroid, max_height)
            else:
                self.get_logger().warn(f"Service call failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def publish_centroid(self, centroid, max_height):
        # Prepare PointStamped message
        point_msg = PointStamped()
        point_msg.header = Header()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'map'

        point_msg.point.x = float(centroid[0])
        point_msg.point.y = float(centroid[1])
        point_msg.point.z = float(max_height + 0.2)

        self.point_pub.publish(point_msg)
        self.get_logger().info(
            f"Published centroid: ({centroid[0]}, {centroid[1]}, {centroid[2]})"
        )


def main(args=None):
    rclpy.init(args=args)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Call /get_top1_cluster service with custom queries")
    parser.add_argument('--main_query', type=str, required=True, help="Main query for the service")
    parser.add_argument('--num_points', type=int, default=10, help="Number of points to request")
    cli_args = parser.parse_args()

    node = ClusterCentroidPublisher(cli_args.main_query, cli_args.num_points)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


