#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <chrono>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std::chrono_literals;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::CameraInfo;

// Global publishers
image_transport::Publisher color_pub;
image_transport::Publisher depth_pub;
image_transport::Publisher thermal_pub;
rclcpp::Publisher<CameraInfo>::SharedPtr color_info_pub;
rclcpp::Publisher<CameraInfo>::SharedPtr depth_info_pub;
rclcpp::Publisher<CameraInfo>::SharedPtr thermal_info_pub;

// Global camera info messages
CameraInfo color_info_msg;
CameraInfo depth_info_msg;
// CameraInfo thermal_info_msg;

// One-shot subscriptions
rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub;
rclcpp::Subscription<CameraInfo>::SharedPtr dinfo_sub;
// rclcpp::Subscription<CameraInfo>::SharedPtr tinfo_sub;

typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image> MySyncPolicy;

void copyCameraInfo(const CameraInfo &in, CameraInfo &out)
{
  out = in;
}

void colorInfoCallback(const CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_synchronizer"), "Received color camera info");
  copyCameraInfo(*msg, color_info_msg);
  cinfo_sub.reset();  // Shutdown
}

void depthInfoCallback(const CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_synchronizer"), "Received depth camera info");
  copyCameraInfo(*msg, depth_info_msg);
  dinfo_sub.reset();  // Shutdown
}

// void thermalInfoCallback(const CameraInfo::SharedPtr msg)
// {
//   RCLCPP_INFO(rclcpp::get_logger("rgbd_synchronizer"), "Received thermal camera info");
//   copyCameraInfo(*msg, thermal_info_msg);
//   tinfo_sub.reset();  // Shutdown
// }

void syncCallback(const Image::ConstSharedPtr &color, const Image::ConstSharedPtr &depth, const Image::ConstSharedPtr &thermal)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_synchronizer"), "Received synchronized images");

  color_pub.publish(color);
  depth_pub.publish(depth);
  thermal_pub.publish(thermal);

  color_info_msg.header = color->header;
  depth_info_msg.header = depth->header;
  // thermal_info_msg.header = thermal->header;

  color_info_pub->publish(color_info_msg);
  depth_info_pub->publish(depth_info_msg);
  // thermal_info_pub->publish(thermal_info_msg);

  std::this_thread::sleep_for(100ms);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("rgbd_synchronizer");

  // One-shot camera info subscriptions
  cinfo_sub = node->create_subscription<CameraInfo>(
    "/camera/color/camera_info", 1, colorInfoCallback);
  dinfo_sub = node->create_subscription<CameraInfo>(
    "/camera/depth/camera_info", 
    1, depthInfoCallback);

  // Publishers
  image_transport::ImageTransport it(node);
  color_pub = it.advertise("/camera_throttled/color/image_raw", 1);
  depth_pub = it.advertise("/camera_throttled/depth/image_rect_raw", 1);
  thermal_pub = it.advertise("/camera_throttled/thermal/image_raw", 1);
  color_info_pub = node->create_publisher<CameraInfo>("/camera_throttled/color/camera_info", 1);
  depth_info_pub = node->create_publisher<CameraInfo>("/camera_throttled/depth/camera_info", 1);
  // thermal_info_pub = node->create_publisher<CameraInfo>("/camera_throttled/thermal/camera_info", 1);

  // Synchronized subscribers
  static message_filters::Subscriber<Image> color_sub(node, "/camera/color/image_raw");
  static message_filters::Subscriber<Image> depth_sub(node, "/camera/depth/image_rect_raw");
  static message_filters::Subscriber<Image> thermal_sub(node, "/camera/thermal/image_raw"); //"/camera/depth/image_rect_raw");
  // static message_filters::TimeSynchronizer<Image, Image, Image> sync(color_sub, depth_sub, thermal_sub, 10);
  static message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), color_sub, depth_sub, thermal_sub);
  sync.registerCallback(syncCallback);  
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

