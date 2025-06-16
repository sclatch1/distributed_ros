cmake_minimum_required(VERSION 3.0.2)

project(task_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  robot_msgs  
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs robot_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)