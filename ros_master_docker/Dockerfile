FROM ros:noetic

RUN apt-get update && apt-get install -y \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

COPY check_ros_master.sh /usr/local/bin/check_ros_master.sh
RUN chmod +x /usr/local/bin/check_ros_master.sh