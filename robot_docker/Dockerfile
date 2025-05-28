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

# 3. Create the robot_pkg package
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd src && \
    catkin_create_pkg robot_pkg rospy std_msgs message_generation && \
    mkdir -p robot_pkg/scripts robot_pkg/msg"

# 4. Add the NumpyArray.msg definition
COPY NumpyArray.msg src/robot_pkg/msg/NumpyArray.msg

COPY DropTask.msg src/robot_pkg/msg/DropTask.msg
COPY DropTaskArray.msg src/robot_pkg/msg/DropTaskArray.msg


# 5. Copy your node script
COPY robot_node.py src/robot_pkg/scripts/robot_node.py
RUN chmod +x src/robot_pkg/scripts/robot_node.py

# Copy custom modules
COPY mvo.py src/robot_pkg/scripts/mvo.py
COPY pso.py src/robot_pkg/scripts/pso.py
COPY fitness_ea.py src/robot_pkg/scripts/fitness_ea.py
COPY fitness_eu.py src/robot_pkg/scripts/fitness_eu.py
COPY table.py src/robot_pkg/scripts/table.py
COPY lookup_table.pkl src/robot_pkg/scripts/lookup_table.pkl

# copy CMakeLists.txt
COPY robot_docker/CMakeLists.txt src/robot_pkg/CMakeLists.txt


# 7. Patch package.xml
RUN /bin/bash -lc "\
    sed -i '/<\/package>/i <build_depend>message_generation</build_depend>' src/robot_pkg/package.xml && \
    sed -i '/<\/package>/i <exec_depend>message_runtime</exec_depend>' src/robot_pkg/package.xml"

# 8. Build the workspace
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && \
    catkin_make"

# 9. Environment and entrypoint
ENV ROS_MASTER_URI=http://ros-master:11311
ENTRYPOINT ["/bin/bash","-c","\
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    rosrun robot_pkg robot_node.py"]