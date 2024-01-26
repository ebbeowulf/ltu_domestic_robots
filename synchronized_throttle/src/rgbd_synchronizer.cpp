#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <unistd.h>

using namespace sensor_msgs;
using namespace message_filters;

// Publishers
image_transport::Publisher color_pub;
image_transport::Publisher depth_pub;
ros::Publisher color_info_pub;
ros::Publisher depth_info_pub;

// Camera Info variables
CameraInfo color_info_msg, depth_info_msg;
ros::Subscriber cinfo_sub, dinfo_sub;

void copyCameraInfo(const CameraInfo& cam_info_in, CameraInfo& cam_info_out){
  cam_info_out.header=cam_info_in.header;
  cam_info_out.height=cam_info_in.height;
  cam_info_out.width=cam_info_in.width;
  cam_info_out.distortion_model=cam_info_in.distortion_model;
  cam_info_out.D=cam_info_in.D;
  cam_info_out.K=cam_info_in.K;
  cam_info_out.R=cam_info_in.R;
  cam_info_out.P=cam_info_in.P;
  cam_info_out.binning_x=cam_info_in.binning_x;
  cam_info_out.binning_y=cam_info_in.binning_y;
  cam_info_out.roi=cam_info_in.roi;
}

void color_info_callback(const CameraInfo& cam_info)
{
  std::cout << "Received color info message" << std::endl;
  copyCameraInfo(cam_info, color_info_msg);
  cinfo_sub.shutdown();
}

void depth_info_callback(const CameraInfo& cam_info)
{
  std::cout << "Received depth info message" << std::endl;
  copyCameraInfo(cam_info, depth_info_msg);
  dinfo_sub.shutdown();
}

void callback(const ImageConstPtr& color_image,
                const ImageConstPtr& depth_image)
{
    std::cout << "Received message" << std::endl;
    color_pub.publish(color_image);
    depth_pub.publish(depth_image);
    color_info_msg.header=color_image->header;
    depth_info_msg.header=depth_image->header;
    color_info_pub.publish(color_info_msg);
    depth_info_pub.publish(depth_info_msg);
    usleep(300000);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rgbd_synchronizer");

  color_info_msg.height=0;
  color_info_msg.width=0;
  depth_info_msg.height=0;
  depth_info_msg.width=0;

  ros::NodeHandle nh("~");
  // Subscribe to camera info topics - will retrieve single message and shutdown
  cinfo_sub = nh.subscribe("/camera/color/camera_info", 1, color_info_callback);
  dinfo_sub = nh.subscribe("/camera/depth/camera_info", 1, depth_info_callback);

  // Create Publishers
  image_transport::ImageTransport itC(nh);
  image_transport::ImageTransport itD(nh);
  color_pub = itC.advertise("/camera_throttled/color/image_raw", 1);
  depth_pub = itD.advertise("/camera_throttled/depth/image_rect_raw", 1);
  color_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/camera_throttled/color/camera_info", 1);
  depth_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/camera_throttled/depth/camera_info", 1);

  // Create Synchronized Callback
  message_filters::Subscriber<Image> color_image_sub(nh, "/camera/color/image_raw", 1);
  message_filters::Subscriber<Image> depth_image_sub(nh, "/camera/depth/image_rect_raw", 1);
  TimeSynchronizer<Image,  Image> sync(color_image_sub, depth_image_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return 0;
}
