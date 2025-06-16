FROM ros:noetic

# 1. Install catkin tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-catkin-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# 2. Create and initialize catkin workspace
WORKDIR /root/catkin_ws
RUN mkdir -p src
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd src && \
    catkin_init_workspace"

# 3. Create the coordinator_pkg package
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd src && \
    catkin_create_pkg task_pkg rospy std_msgs message_generation && \
    mkdir -p task_pkg/scripts task_pkg/msg"



COPY robot_msgs src/robot_msgs



# 5. Copy your node script
COPY coordinator_node.py src/task_pkg/scripts/coordinator_node.py
COPY task.py src/task_pkg/scripts/task.py
RUN chmod +x src/task_pkg/scripts/task.py



# copy CMakeLists.txt
COPY task_docker/CMakeLists.txt src/task_pkg/CMakeLists.txt


# 7. Patch package.xml
RUN /bin/bash -lc "sed -i '/<\/package>/i <depend>robot_msgs</depend>' src/task_pkg/package.xml"

# 8. Build the workspace
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && \
    catkin_make"

# 9. Environment and entrypoint
ENV ROS_MASTER_URI=http://ros-master:11311
ENTRYPOINT ["/bin/bash","-c","\
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    rosrun task_pkg task.py"]
