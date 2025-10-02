cmake_minimum_required(VERSION 3.5)
project(synchronized_throttle)

## Compile as C++14 (ROS2 default, newer than ROS1's C++11)
# add_compile_options(-std=c++14)

## Find ament macros and libraries
## In ROS1 we used: find_package(catkin REQUIRED COMPONENTS ...)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py (for Python nodes)
# ament_python_install_package(${PROJECT_NAME})

################################################
## Declare ROS messages, services and actions ##
################################################

## In this package we don’t declare custom msg/srv files
## But if we did, in ROS1 we used add_message_files()/generate_messages()
## In ROS2, we would use rosidl_generate_interfaces()

# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Example.msg"
#   DEPENDENCIES sensor_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Dynamic reconfigure is different in ROS2 (use parameters instead).
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## ament specific configuration  ##
###################################
## In ROS1 we had catkin_package()
## In ROS2 we use ament_package()
ament_package()

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
# include
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/synchronized_throttle.cpp
# )

## Add cmake target dependencies of the library
# ament_target_dependencies(${PROJECT_NAME}
#   rclcpp
#   sensor_msgs
# )

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/rgbd_synchronizer.cpp)

## Rename C++ executable without prefix
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  sensor_msgs
  image_transport
  message_filters
)

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${ament_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use ament DESTINATION variables

## Mark executable scripts (Python etc.) for installation
# ament_python_install_package(${PROJECT_NAME})

## Mark executables for installation
install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

## Mark libraries for installation
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION include
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION share/${PROJECT_NAME}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# ament_add_gtest(${PROJECT_NAME}-test test/test_synchronized_throttle.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python tests
# ament_add_pytest_test(test)
