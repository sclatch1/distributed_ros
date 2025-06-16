cmake_minimum_required(VERSION 3.0.2)
project(robot_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
  geometry_msgs
)

add_service_files(
  FILES
  RobotStatus.srv
)

add_message_files(
  FILES
  DropTask.msg
  DropTaskArray.msg
  UniverseArray.msg
  RobotStatusArray.msg
  RobotStatusM.msg
  FitnessValue.msg
  Parameters.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)
