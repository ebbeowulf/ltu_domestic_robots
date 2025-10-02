#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <image_transport/image_transport.hpp>
#include <unistd.h>

using namespace sensor_msgs::msg;
using namespace message_filters;

class RGBDSynchronizer : public rclcpp::Node
{
public:
  RGBDSynchronizer()
  : Node("rgbd_synchronizer")
  {
    // Init camera info
    color_info_msg_.height = 0;
    color_info_msg_.width = 0;
    depth_info_msg_.height = 0;
    depth_info_msg_.width = 0;

    // Publishers
    image_transport::ImageTransport it(shared_from_this());
    color_pub_ = it.advertise("/camera_throttled/color/image_raw", 1);
    depth_pub_ = it.advertise("/camera_throttled/depth/image_rect_raw", 1);

    color_info_pub_ = this->create_publisher<CameraInfo>(
        "/camera_throttled/color/camera_info", 10);
    depth_info_pub_ = this->create_publisher<CameraInfo>(
        "/camera_throttled/depth/camera_info", 10);

    // One-shot subscriptions for camera info
    cinfo_sub_ = this->create_subscription<CameraInfo>(
        "/camera/color/camera_info", 1,
        std::bind(&RGBDSynchronizer::color_info_callback, this, std::placeholders::_1));

    dinfo_sub_ = this->create_subscription<CameraInfo>(
        "/camera/depth/camera_info", 1,
        std::bind(&RGBDSynchronizer::depth_info_callback, this, std::placeholders::_1));

    // Message filters for sync
    color_image_sub_.subscribe(this, "/camera/color/image_raw");
    depth_image_sub_.subscribe(this, "/camera/depth/image_rect_raw");

    sync_ = std::make_shared<TimeSynchronizer<Image, Image>>(
        color_image_sub_, depth_image_sub_, 10);
    sync_->registerCallback(
        std::bind(&RGBDSynchronizer::callback, this,
                  std::placeholders::_1, std::placeholders::_2));
  }

private:
  // Publishers
  image_transport::Publisher color_pub_;
  image_transport::Publisher depth_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr color_info_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr depth_info_pub_;

  // Camera info msgs
  CameraInfo color_info_msg_, depth_info_msg_;

  // One-shot subs
  rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr dinfo_sub_;

  // Message filters
  message_filters::Subscriber<Image> color_image_sub_;
  message_filters::Subscriber<Image> depth_image_sub_;
  std::shared_ptr<TimeSynchronizer<Image, Image>> sync_;

  // Callbacks
  void copyCameraInfo(const CameraInfo &in, CameraInfo &out)
  {
    out = in; // assignment operator works fine in ROS2
  }

  void color_info_callback(const CameraInfo::SharedPtr cam_info)
  {
    RCLCPP_INFO(this->get_logger(), "Received color info message");
    copyCameraInfo(*cam_info, color_info_msg_);
    cinfo_sub_.reset();  // unsubscribes
  }

  void depth_info_callback(const CameraInfo::SharedPtr cam_info)
  {
    RCLCPP_INFO(this->get_logger(), "Received depth info message");
    copyCameraInfo(*cam_info, depth_info_msg_);
    dinfo_sub_.reset();  // unsubscribes
  }

  void callback(const Image::ConstSharedPtr &color_image,
                const Image::ConstSharedPtr &depth_image)
  {
    RCLCPP_INFO(this->get_logger(), "Received synced images");
    color_pub_.publish(color_image);
    depth_pub_.publish(depth_image);

    color_info_msg_.header = color_image->header;
    depth_info_msg_.header = depth_image->header;

    color_info_pub_->publish(color_info_msg_);
    depth_info_pub_->publish(depth_info_msg_);

    usleep(300000);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RGBDSynchronizer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


