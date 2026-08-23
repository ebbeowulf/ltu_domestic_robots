#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
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
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

using namespace std::chrono_literals;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::CameraInfo;

std::shared_ptr<tf2_ros::Buffer> tf_buffer;
std::shared_ptr<tf2_ros::TransformListener> tf_listener;

// One-shot subscriptions
rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub;
rclcpp::Subscription<CameraInfo>::SharedPtr dinfo_sub;

int image_count = 0;

typedef message_filters::sync_policies::ApproximateTime<Image, Image> MySyncPolicy;

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

void colorInfoCallback(const CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_saver"), "Received color camera info");
  write_camera_info_to_file(*msg, "color_camera_info.txt");
  cinfo_sub.reset();  // Shutdown
}

void depthInfoCallback(const CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(rclcpp::get_logger("rgbd_saver"), "Received depth camera info");
  write_camera_info_to_file(*msg, "depth_camera_info.txt");
  dinfo_sub.reset();  // Shutdown
}

void log_camera_pose(const std::string &filename, const std::string &image_id, const rclcpp::Time &stamp) {
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("rgbd_logger"), "Failed to open pose log file");
        return;
    }

    try {
        geometry_msgs::msg::TransformStamped tf = tf_buffer->lookupTransform("map", "camera_link", stamp, rclcpp::Duration::from_seconds(0.5));
        out << image_id << ","
            << tf.transform.translation.x << ","
            << tf.transform.translation.y << ","
            << tf.transform.translation.z << ","
            << tf.transform.rotation.x << ","
            << tf.transform.rotation.y << ","
            << tf.transform.rotation.z << ","
            << tf.transform.rotation.w << "\n";
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(rclcpp::get_logger("rgbd_logger"), "TF lookup failed: %s", ex.what());
        out << image_id << ",nan,nan,nan,nan,nan,nan,nan\n";
    }

    out.close();
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
    RCLCPP_INFO(rclcpp::get_logger("rgbd_saver"), "Unsupported encoding: %s", encoding.c_str());
    return;
  }
}

void syncCallback(const Image::ConstSharedPtr &color, const Image::ConstSharedPtr &depth) {
  RCLCPP_INFO(rclcpp::get_logger("rgbd_saver"), "Received synchronized RGB and depth images");

  std::string color_filename = generate_filename("color_", image_count);
  std::string depth_filename = generate_filename("depth_", image_count);
  image_count++;

  log_camera_pose("pose_log.csv", color_filename, color->header.stamp);

  cv_bridge::CvImageConstPtr color_ptr = cv_bridge::toCvShare(color, color->encoding);
  save_image(color_ptr, color_filename, color->encoding);
  cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth, depth->encoding);
  save_image(depth_ptr, depth_filename, depth->encoding);

}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("rgbd_saver");

    tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    cinfo_sub = node->create_subscription<CameraInfo>(
        "/camera_throttled/color/camera_info", 1, colorInfoCallback);
    dinfo_sub = node->create_subscription<CameraInfo>(
        "/camera_throttled/depth/camera_info", 1, depthInfoCallback);

    static message_filters::Subscriber<Image> color_sub(node, "/camera_throttled/color/image_raw");
    static message_filters::Subscriber<Image> depth_sub(node, "/camera_throttled/depth/image_rect_raw");

    static message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), color_sub, depth_sub);
    sync.registerCallback(syncCallback);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

