#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <chrono>

using namespace std::chrono_literals;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::CameraInfo;

// Global publishers
image_transport::Publisher color_pub;
image_transport::Publisher depth_pub;
rclcpp::Publisher<CameraInfo>::SharedPtr color_info_pub;
rclcpp::Publisher<CameraInfo>::SharedPtr depth_info_pub;

// Global camera info messages
CameraInfo color_info_msg;
CameraInfo depth_info_msg;

// One-shot subscriptions
rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub;
rclcpp::Subscription<CameraInfo>::SharedPtr dinfo_sub;

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

void syncCallback(const Image::ConstSharedPtr &color, const Image::ConstSharedPtr &depth)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_synchronizer"), "Received synchronized images");

  color_pub.publish(color);
  depth_pub.publish(depth);

  color_info_msg.header = color->header;
  depth_info_msg.header = depth->header;

  color_info_pub->publish(color_info_msg);
  depth_info_pub->publish(depth_info_msg);

  // std::this_thread::sleep_for(300ms);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("rgbd_synchronizer");

  // One-shot camera info subscriptions
  cinfo_sub = node->create_subscription<CameraInfo>(
    "/camera/aligned_depth_to_color/camera_info", 1, colorInfoCallback);
    // "/camera/color/camera_info", 1, colorInfoCallback);
  dinfo_sub = node->create_subscription<CameraInfo>(
    "aligned_depth_to_color", // "/camera/depth/camera_info", 
    1, depthInfoCallback);

  // Publishers
  image_transport::ImageTransport it(node);
  color_pub = it.advertise("/camera_throttled/color/image_raw", 1);
  depth_pub = it.advertise("/camera_throttled/depth/image_rect_raw", 1);
  color_info_pub = node->create_publisher<CameraInfo>("/camera_throttled/color/camera_info", 1);
  depth_info_pub = node->create_publisher<CameraInfo>("/camera_throttled/depth/camera_info", 1);

  // Synchronized subscribers
  static message_filters::Subscriber<Image> color_sub(node, "/camera/color/image_raw");
  static message_filters::Subscriber<Image> depth_sub(node, "/camera/aligned_depth_to_color/image_raw"); //"/camera/depth/image_rect_raw");
  static message_filters::TimeSynchronizer<Image, Image> sync(color_sub, depth_sub, 10);
  sync.registerCallback(syncCallback);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

// #include <rclcpp/rclcpp.hpp>
// #include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>
// #include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/msg/camera_info.hpp>
// #include <sensor_msgs/msg/point_cloud2.hpp>
// #include <image_transport/image_transport.hpp>
// #include <unistd.h>

// using namespace sensor_msgs::msg;
// using namespace message_filters;

// class RGBDSynchronizer : public rclcpp::Node
// {
// public:
//   RGBDSynchronizer()
//   : Node("rgbd_synchronizer")
//   {
//     std::cout << "Starting RGBD Synchronizer Node" << std::endl;
//     // Init camera info
//     color_info_msg_.height = 0;
//     color_info_msg_.width = 0;
//     depth_info_msg_.height = 0;
//     depth_info_msg_.width = 0;

//     // Publishers
//     std::shared_ptr<image_transport::ImageTransport> it = std::make_shared<image_transport::ImageTransport>(shared_from_this());

//     color_pub_ = it->advertise("/camera_throttled/color/image_raw", 1);
//     depth_pub_ = it->advertise("/camera_throttled/depth/image_rect_raw", 1);

//     color_info_pub_ = this->create_publisher<CameraInfo>(
//         "/camera_throttled/color/camera_info", 10);
//     depth_info_pub_ = this->create_publisher<CameraInfo>(
//         "/camera_throttled/depth/camera_info", 10);

//     std::cout << "P1" << std::endl;

//     // One-shot subscriptions for camera info
//     cinfo_sub_ = this->create_subscription<CameraInfo>(
//         "/camera/color/camera_info", 1,
//         std::bind(&RGBDSynchronizer::color_info_callback, this, std::placeholders::_1));

//     dinfo_sub_ = this->create_subscription<CameraInfo>(
//         "/camera/depth/camera_info", 1,
//         std::bind(&RGBDSynchronizer::depth_info_callback, this, std::placeholders::_1));

//     std::cout << "P2" << std::endl;

//     // Message filters for sync
//     color_image_sub_.subscribe(this, "/camera/color/image_raw");
//     depth_image_sub_.subscribe(this, "/camera/depth/image_rect_raw");

//     sync_ = std::make_shared<TimeSynchronizer<Image, Image>>(
//         color_image_sub_, depth_image_sub_, 10);
//     sync_->registerCallback(
//         std::bind(&RGBDSynchronizer::callback, this,
//                   std::placeholders::_1, std::placeholders::_2));
//     std::cout << "P3" << std::endl;
//   }

// private:
//   // Publishers
//   image_transport::Publisher color_pub_;
//   image_transport::Publisher depth_pub_;
//   rclcpp::Publisher<CameraInfo>::SharedPtr color_info_pub_;
//   rclcpp::Publisher<CameraInfo>::SharedPtr depth_info_pub_;

//   // Camera info msgs
//   CameraInfo color_info_msg_, depth_info_msg_;

//   // One-shot subs
//   rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub_;
//   rclcpp::Subscription<CameraInfo>::SharedPtr dinfo_sub_;

//   // Message filters
//   message_filters::Subscriber<Image> color_image_sub_;
//   message_filters::Subscriber<Image> depth_image_sub_;
//   std::shared_ptr<TimeSynchronizer<Image, Image>> sync_;

//   // Callbacks
//   void copyCameraInfo(const CameraInfo &in, CameraInfo &out)
//   {
//     out = in; // assignment operator works fine in ROS2
//   }

//   void color_info_callback(const CameraInfo::SharedPtr cam_info)
//   {
//     RCLCPP_INFO(this->get_logger(), "Received color info message");
//     copyCameraInfo(*cam_info, color_info_msg_);
//     cinfo_sub_.reset();  // unsubscribes
//   }

//   void depth_info_callback(const CameraInfo::SharedPtr cam_info)
//   {
//     copyCameraInfo(*cam_info, depth_info_msg_);
//     dinfo_sub_.reset();  // unsubscribes
//   }

//   void callback(const Image::ConstSharedPtr &color_image,
//                 const Image::ConstSharedPtr &depth_image)
//   {
//     RCLCPP_INFO(this->get_logger(), "Received synced images");
//     color_pub_.publish(color_image);
//     depth_pub_.publish(depth_image);

//     color_info_msg_.header = color_image->header;
//     depth_info_msg_.header = depth_image->header;

//     color_info_pub_->publish(color_info_msg_);
//     depth_info_pub_->publish(depth_info_msg_);

//     usleep(300000);
//   }
// };

// int main(int argc, char **argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<RGBDSynchronizer>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
