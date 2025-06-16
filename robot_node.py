#!/usr/bin/env python3
import rospy
from robot_msgs.msg import DropTaskArray, UniverseArray, RobotStatusArray, FitnessValue, Parameters
from geometry_msgs.msg import Point
from robot_msgs.srv import RobotStatus, RobotStatusResponse
from std_msgs.msg import Header

import os
import numpy as np

from pso import PSO_Algorithm
import time

from dataclasses import dataclass
from typing import Tuple

from utilities import log_coordinator_timing, write_to_file, write_np_to_file

# Get robot-specific environment variables
robot_name = os.environ.get("ROBOT_NAME", "robot")
robot_x = float(os.environ.get("ROBOT_X", "0.0"))
robot_y = float(os.environ.get("ROBOT_Y", "0.0"))
battery_level = float(os.environ.get("BATTERY", "100.0"))

robot_status_are_cached = False
universe_are_cached = False
tasks_are_cached = False

@dataclass
class RobotStatusInfo:
    position: Tuple[float, float]
    battery: float
    name: str



def cach_task(msg):
    rospy.loginfo("Received task array")
    global cached_tasks


    tasks = []
    for t in msg.tasks:
        task = {
            "drop_initial_coordination" : (t.drop_initial_coordination.x, t.drop_initial_coordination.y),
            "order" : t.order,
            "drop_target_coordination" : (t.drop_target_coordination.x, t.drop_target_coordination.y)
        }
        tasks.append(task)
    cached_tasks = tasks




def cach_universe(msg):
    rospy.loginfo("Received universe array")
    universes = np.array(msg.universe).reshape((msg.rows, msg.cols))

    global cached_universe
    cached_universe = universes


        


def cach_robot_status(msg):
    global robot_status_are_cached
    robot_status_are_cached = True
    global cached_robot_statuses
    cached_robot_statuses = msg.status




def parameters_callback(msg):
    recv_time = rospy.Time.now()
    send_time = msg.header.stamp
    communication_robot_status = (recv_time - send_time).to_sec()
    rospy.loginfo(f"getting robot status in {communication_robot_status}s")
    
    # cached the tasks in cached_tasks global variable
    cach_task(msg)
    cach_robot_status(msg)
    cach_universe(msg)

    rospy.loginfo("All data cached — running PSO.")
    best = run_explotation()

    val = FitnessValue(fitness=best)
    fitness_pub = rospy.Publisher('/fitness_value', FitnessValue, queue_size=1)
    
    while fitness_pub.get_num_connections() == 0:
        rospy.logwarn(f"Waiting for subscriber on /{robot_name}/fitness.")
        rospy.sleep(0.05)

    rospy.loginfo(f"publishing best fitness of {robot_name}")
    val.header = Header(stamp=rospy.Time.now())
    fitness_pub.publish(val)
    rospy.loginfo(f"[{robot_name}] /fitness_value publisher connections: {fitness_pub.get_num_connections()}")
    #rospy.signal_shutdown("fitness value sent. Shutting down")




def handle_status_request(req):
    rospy.loginfo(f"[{robot_name}] Status requested.")
    res = RobotStatusResponse()
    res.robot_name = robot_name
    res.position = Point(x=robot_x, y=robot_y, z=0.0)
    res.battery = battery_level
    return res



def robot_node():
    rospy.init_node('robot_node', anonymous=True)

    # Service server: /get_robot_status_<robot_name>
    service_name = f"/get_robot_status_{robot_name}"
    rospy.Service(service_name, RobotStatus, handle_status_request)
    rospy.loginfo(f"[{robot_name}] Ready to respond on service {service_name}")



    rospy.Subscriber(f'/{robot_name}/parameters', Parameters , parameters_callback)
    rospy.loginfo(f"[{robot_name}] subscribe to /{robot_name}/parameters")


    rospy.loginfo(f"[{robot_name}] Node is running. Position: ({robot_x}, {robot_y}), Battery: {battery_level}%")
    rospy.spin()

def run_explotation():
    print("running explotation")
    s = 1  # number of Charge stations
    Energy_Harvesting_Rate = 3
    MAX_CHARGE = 20
    CHARGING_RATE = 10 / 3600  # w/s
    DECHARGING_RATE = 5 / 3600  # w/s
    DECHARGING_Time = 4 * 3600  # s
    CHARGING_TIME = 0.5 * 3600  # s
    iterationstop = 15
    Charging_station = (-1.6, 7.2)
    POP_SIZE = 10
    MAX_ITERATIONS = 50

    N = len(cached_robot_statuses)  # number of robots
    M = len(cached_tasks)  # number of tasks
    


    robot_charge_duration = []
    robots_coord = []
    for r in cached_robot_statuses:
        bat = (r.battery / 100.0) * 20 * 3600
        robot_charge_duration.append(bat)
        point = [r.position.x, r.position.y]
        robots_coord.append(point)

    robots_coord = np.array(robots_coord)

    pso1_d = time.time()
    best_fitness , _ = PSO_Algorithm(MAX_ITERATIONS,len(cached_universe),M,N,s,iterationstop,robot_charge_duration,robots_coord,cached_tasks,Charging_station,CHARGING_TIME, Energy_Harvesting_Rate, cached_universe)
    pso2_d = time.time()
    pso_d = pso2_d - pso1_d
    
    CSV_FILE = 'data/coord_timing_distributed.csv'
    log_coordinator_timing(pso_d, CSV_FILE)

    rospy.loginfo(f"time pso distributed {pso_d} and best fitness {best_fitness}")

    return best_fitness








if __name__ == '__main__':
    robot_node()

