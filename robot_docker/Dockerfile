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

# 3. Add shared robot_msgs package
COPY robot_msgs src/robot_msgs

# 4. Create the robot_pkg package
RUN /bin/bash -lc "source /opt/ros/noetic/setup.bash && \
    cd src && \
    catkin_create_pkg robot_pkg rospy std_msgs && \
    mkdir -p robot_pkg/scripts"

# 5. Copy robot node and scripts
COPY robot_node.py src/robot_pkg/scripts/robot_node.py
RUN chmod +x src/robot_pkg/scripts/robot_node.py

# Copy custom modules
COPY mvo.py src/robot_pkg/scripts/mvo.py
COPY pso.py src/robot_pkg/scripts/pso.py
COPY apso.py src/robot_pkg/scripts/apso.py
COPY ga.py src/robot_pkg/scripts/ga.py

COPY fitness_ea.py src/robot_pkg/scripts/fitness_ea.py
COPY fitness_eu.py src/robot_pkg/scripts/fitness_eu.py
COPY table.py src/robot_pkg/scripts/table.py
COPY utilities.py src/robot_pkg/scripts/utilities.py
COPY lookup_table.pkl src/robot_pkg/scripts/lookup_table.pkl

# 6. Copy CMakeLists.txt
COPY robot_docker/CMakeLists.txt src/robot_pkg/CMakeLists.txt

# 7. Patch package.xml to depend on robot_msgs
RUN /bin/bash -lc "sed -i '/<\/package>/i <depend>robot_msgs</depend>' src/robot_pkg/package.xml"

# 8. Build the workspace (messages first, then robot_pkg)
RUN /bin/bash -lc "\
    source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && \
    catkin_make"

# 9. Source environment and launch node
ENV ROS_MASTER_URI=http://ros-master:11311
ENTRYPOINT ["/bin/bash", "-c", "\
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    rosrun robot_pkg robot_node.py"]
