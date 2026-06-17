cmake_minimum_required(VERSION 3.8)
project(synchronized_throttle)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(rgbd_synchronizer src/rgbd_synchronizer.cpp)
ament_target_dependencies(rgbd_synchronizer rclcpp sensor_msgs image_transport message_filters)

add_executable(color_depth_thermal_synchronizer src/color_depth_thermal_synchronizer.cpp)
ament_target_dependencies(color_depth_thermal_synchronizer rclcpp sensor_msgs image_transport message_filters)

add_executable(depth_image_saver src/depth_image_saver.cpp)
ament_target_dependencies(depth_image_saver rclcpp sensor_msgs image_transport message_filters cv_bridge)

add_executable(color_depth_thermal_ir_synchronizer src/color_depth_thermal_ir_synchronizer.cpp)
ament_target_dependencies(color_depth_thermal_ir_synchronizer rclcpp sensor_msgs image_transport message_filters)

install(TARGETS
  rgbd_synchronizer
  color_depth_thermal_synchronizer
  color_depth_thermal_ir_synchronizer
  depth_image_saver
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

