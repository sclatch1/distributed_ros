cmake_minimum_required(VERSION 3.0.2)
    
project(coordinator_pkg)
    
find_package(catkin REQUIRED COMPONENTS rospy std_msgs message_generation)

add_message_files(FILES NumpyArray.msg DropTask.msg DropTaskArray.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS rospy std_msgs message_runtime)