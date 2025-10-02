cmake_minimum_required(VERSION 3.5)
project(stretch_srvs)

## Compile as C++14 (ROS2 default, newer than ROS1's C++11)
# add_compile_options(-std=c++14)

## Find ament macros and libraries
## In ROS2 we use ament instead of catkin
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
# ament_python_install_package(${PROJECT_NAME})

################################################
## Declare ROS messages, services and actions ##
################################################

## In ROS1 we used add_message_files() and add_service_files().
## In ROS2 we use rosidl_generate_interfaces() instead.
## All msg/srv/action files are listed directly in that macro.

# add_message_files(
#   FILES
#   Cluster.msg
#   ClusterArray.msg
# )

# add_service_files(
#   FILES
#   DrawCluster.srv
#   GetCluster.srv
#   MoveArm.srv
#   SetInt.srv
# )

# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## ROS2 equivalent:
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Cluster.msg"
  "msg/ClusterArray.msg"
  "srv/DrawCluster.srv"
  "srv/GetCluster.srv"
  "srv/MoveArm.srv"
  "srv/SetInt.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Dynamic reconfigure is different in ROS2 (use parameters instead).
## Keeping the section here for reference.
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## ament specific configuration  ##
###################################
## In ROS1 we had catkin_package().
## In ROS2 we use ament_package() instead.
ament_package()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  # ${ament_INCLUDE_DIRS}  # Not needed in ROS2, ament handles this
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/names.cpp
# )

## Add cmake target dependencies of the library
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(${PROJECT_NAME}_node src/stretch_srvs_node.cpp)

## Rename C++ executable without prefix
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${ament_LIBRARIES}
# )

#############
## Install ##
#############

# Mark executable scripts (Python etc.) for installation
# ament_python_install_package(${PROJECT_NAME})

# Mark executables for installation
# install(TARGETS ${PROJECT_NAME}_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# Mark libraries for installation
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

# Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION include
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

# Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION share/${PROJECT_NAME}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# ament_add_gtest(${PROJECT_NAME}-test test/test_stretch_srvs.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python tests
# ament_add_pytest_test(test)
