#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <chrono>
#include <message_filters/sync_policies/approximate_time.h>
#include <sstream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::NavSatFix;

// One-shot subscriptions
rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub;
rclcpp::Subscription<CameraInfo>::SharedPtr dinfo_sub;
rclcpp::Subscription<NavSatFix>::SharedPtr gps_sub;

int image_count = 0;

typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image> MySyncPolicy;

std::string generate_filename(std::string prefix, int image_count) {
    std::ostringstream oss;
    oss << prefix << std::setw(5) << std::setfill('0') << image_count << ".png";
    return oss.str();
}

void write_camera_info_to_file(const sensor_msgs::msg::CameraInfo &msg, const std::string &filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    out << "Camera Info:\n";
    out << "  Width: " << msg.width << "\n";
    out << "  Height: " << msg.height << "\n";
    out << "  Distortion Model: " << msg.distortion_model << "\n";

    out << "  D (Distortion coefficients): ";
    for (const auto &d : msg.d) out << d << " ";
    out << "\n";

    out << "  K (Intrinsic matrix): ";
    for (size_t i = 0; i < msg.k.size(); ++i) {
        out << msg.k[i] << ((i % 3 == 2) ? "\n                             " : " ");
    }

    out << "\n  R (Rectification matrix): ";
    for (size_t i = 0; i < msg.r.size(); ++i) {
        out << msg.r[i] << ((i % 3 == 2) ? "\n                             " : " ");
    }

    out << "\n  P (Projection matrix): ";
    for (size_t i = 0; i < msg.p.size(); ++i) {
        out << msg.p[i] << ((i % 4 == 3) ? "\n                             " : " ");
    }

    out << "\n  Binning (x, y): " << msg.binning_x << ", " << msg.binning_y << "\n";
    out << "  ROI: x_offset=" << msg.roi.x_offset
        << ", y_offset=" << msg.roi.y_offset
        << ", height=" << msg.roi.height
        << ", width=" << msg.roi.width
        << ", do_rectify=" << (msg.roi.do_rectify ? "true" : "false") << "\n";

    out.close();
}

// Routine for writing GPS data to CSV
//   Call from GPS message callback with lat/lon
//   Call from image message callback without lat/lon to help identify localize images in post processing
void append_gps_to_csv(const std::string &filename,
                       const std::string &message_type,
                       const rclcpp::Time &ros_time,
                       std::optional<double> latitude = std::nullopt,
                       std::optional<double> longitude = std::nullopt) {
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Convert ROS time to human-readable UTC
    auto sec = static_cast<time_t>(ros_time.seconds());
    std::tm tm_utc = *std::gmtime(&sec);
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_utc);

    // Extract fractional seconds
    auto ros_seconds = ros_time.seconds();  // double precision
    int millis = static_cast<int>((ros_seconds - sec) * 1000.0); 
    // Append milliseconds to time_buf 
    char temp_buf[64];  // Safe size for full timestamp
    std::snprintf(temp_buf, sizeof(temp_buf), "%s.%03d", time_buf, millis);
    std::strncpy(time_buf, temp_buf, sizeof(time_buf) - 1); 
    time_buf[sizeof(time_buf) - 1] = '\0';  // Ensure null termination

    // Format lat/lon or fallback to nan
    std::string lat_str = latitude.has_value()
        ? std::to_string(latitude.value())
        : "nan";
    std::string lon_str = longitude.has_value()
        ? std::to_string(longitude.value())
        : "nan";

    out << message_type << ","
        << time_buf << ","
        << lat_str << ","
        << lon_str << "\n";

    out.close();
}

void colorInfoCallback(const CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_thermal_saver"), "Received color camera info");
  write_camera_info_to_file(*msg, "color_camera_info.txt");
  cinfo_sub.reset();  // Shutdown
}

void depthInfoCallback(const CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_thermal_saver"), "Received depth camera info");
  write_camera_info_to_file(*msg, "depth_camera_info.txt");
  dinfo_sub.reset();  // Shutdown
}

void navSatCallback(const NavSatFix::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_thermal_saver"), "Received GPS fix");
  append_gps_to_csv("gps_log.csv", "GPS", msg->header.stamp, msg->latitude, msg->longitude);
}

void save_image(cv_bridge::CvImageConstPtr cv_ptr, const std::string &filename, const std::string &encoding)
{
  if (encoding == "mono16" || encoding == "bgr8" || encoding == "rgb8" || encoding == "bgra8" || encoding == "rgba8")
  {
    cv::imwrite(filename, cv_ptr->image);
  }
  else if (encoding == "32FC1")
  {
    // Convert meters to millimeters and save as 16-bit PNG
    cv::Mat converted;
    cv_ptr->image.convertTo(converted, CV_16UC1, 1000.0);
    cv::imwrite(filename, converted);
  }
  else
  {
    RCLCPP_INFO(rclcpp::get_logger("rgbd_thermal_saver"), "Unsupported encoding: %s", encoding.c_str());
    return;
  }
}

void syncCallback(const Image::ConstSharedPtr &color, const Image::ConstSharedPtr &depth, const Image::ConstSharedPtr &thermal)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_thermal_saver"), "Received synchronized images");

  std::string color_filename = generate_filename("color_", image_count);
  std::string depth_filename = generate_filename("depth_", image_count);
  std::string thermal_filename = generate_filename("thermal_", image_count);
  image_count++;

  append_gps_to_csv("gps_log.csv", color_filename, color->header.stamp);

  cv_bridge::CvImageConstPtr color_ptr = cv_bridge::toCvShare(color, color->encoding);
  save_image(color_ptr, color_filename, color->encoding);
  cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth, depth->encoding);
  save_image(depth_ptr, depth_filename, depth->encoding);
  cv_bridge::CvImageConstPtr thermal_ptr = cv_bridge::toCvShare(thermal, thermal->encoding);
  save_image(thermal_ptr, thermal_filename, thermal->encoding);

}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("rgbd_thermal_saver");

  // One-shot camera info subscriptions
  cinfo_sub = node->create_subscription<CameraInfo>(
    "/camera_throttled/color/camera_info", 1, colorInfoCallback);
  dinfo_sub = node->create_subscription<CameraInfo>(
    "/camera_throttled/depth/camera_info", 1, depthInfoCallback);
  gps_sub = node->create_subscription<NavSatFix>(
    "/fix", 1, navSatCallback);

  // Synchronized subscribers
  static message_filters::Subscriber<Image> color_sub(node, "/camera_throttled/color/image_raw");
  static message_filters::Subscriber<Image> depth_sub(node, "/camera_throttled/depth/image_rect_raw");
  static message_filters::Subscriber<Image> thermal_sub(node, "/camera_throttled/thermal/image_raw"); 
  static message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), color_sub, depth_sub, thermal_sub);
  sync.registerCallback(syncCallback);  
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

