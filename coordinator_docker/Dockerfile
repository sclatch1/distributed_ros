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
    catkin_create_pkg coordinator_pkg rospy std_msgs message_generation && \
    mkdir -p coordinator_pkg/scripts coordinator_pkg/msg"



COPY robot_msgs src/robot_msgs



# 5. Copy your node script
COPY coordinator_node.py src/coordinator_pkg/scripts/coordinator_node.py
RUN chmod +x src/coordinator_pkg/scripts/coordinator_node.py

# Copy custom modules
COPY mvo.py src/coordinator_pkg/scripts/mvo.py
COPY pso.py src/coordinator_pkg/scripts/pso.py
COPY fitness_ea.py src/coordinator_pkg/scripts/fitness_ea.py
COPY fitness_eu.py src/coordinator_pkg/scripts/fitness_eu.py
COPY table.py src/coordinator_pkg/scripts/table.py
COPY utilities.py src/coordinator_pkg/scripts/utilities.py
COPY lookup_table.pkl src/coordinator_pkg/scripts/lookup_table.pkl


# copy CMakeLists.txt
COPY coordinator_docker/CMakeLists.txt src/coordinator_pkg/CMakeLists.txt


# 7. Patch package.xml
RUN /bin/bash -lc "sed -i '/<\/package>/i <depend>robot_msgs</depend>' src/coordinator_pkg/package.xml"

# 8. Build the workspace
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && \
    catkin_make"

# 9. Environment and entrypoint
ENV ROS_MASTER_URI=http://ros-master:11311
ENTRYPOINT ["/bin/bash","-c","\
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    rosrun coordinator_pkg coordinator_node.py"]
