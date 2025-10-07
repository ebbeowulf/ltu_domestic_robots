cmake_minimum_required(VERSION 3.5)
project(detect_and_touch)

# --- Dependencies ---
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# If your C++ node uses these, list them explicitly
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(stretch_srvs REQUIRED)

# --- Python ---
ament_python_install_package(${PROJECT_NAME})

#install(
#  PROGRAMS
#    my_package_name/my_node.py
#  DESTINATION lib/${PROJECT_NAME}
#)

# --- C++ ---
#add_executable(my_cpp_node src/my_cpp_node.cpp)
#ament_target_dependencies(my_cpp_node
#  rclcpp
#  sensor_msgs
#  nav_msgs
#  geometry_msgs
#  visualization_msgs
#  tf2_ros
#)

#install(
#  TARGETS my_cpp_node
#  DESTINATION lib/${PROJECT_NAME}
#)

# --- Package Export ---
ament_package()
