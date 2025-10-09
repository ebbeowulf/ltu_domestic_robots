#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class DepthImageSaver : public rclcpp::Node
{
public:
  DepthImageSaver() : Node("depth_image_saver"), image_count_(0)
  {
    sub_ = image_transport::create_subscription(
      this, "/camera/depth/image_raw",
      std::bind(&DepthImageSaver::imageCallback, this, std::placeholders::_1),
      "raw");

    RCLCPP_INFO(this->get_logger(), "DepthImageSaver node started");
  }

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
  {
    try
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
      std::string filename = "depth_" + std::to_string(image_count_) + ".png";

      if (msg->encoding == "mono16")
      {
        cv::imwrite(filename, cv_ptr->image);
      }
      else if (msg->encoding == "32FC1")
      {
        // Convert meters to millimeters and save as 16-bit PNG
        cv::Mat converted;
        cv_ptr->image.convertTo(converted, CV_16UC1, 1000.0);
        cv::imwrite(filename, converted);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "Unsupported encoding: %s", msg->encoding.c_str());
        return;
      }

      RCLCPP_INFO(this->get_logger(), "Saved depth image: %s", filename.c_str());
      image_count_++;
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  image_transport::Subscriber sub_;
  int image_count_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthImageSaver>());
  rclcpp::shutdown();
  return 0;
}
