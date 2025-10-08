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

add_executable(rgbd_synchronizer src/rgbd_synchronizer.cpp)
ament_target_dependencies(rgbd_synchronizer rclcpp sensor_msgs image_transport message_filters)

install(TARGETS
  rgbd_synchronizer
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

