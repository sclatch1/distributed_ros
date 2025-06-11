#!/usr/bin/env python3
import rospy
from robot_msgs.msg import DropTaskArray, UniverseArray, RobotStatusArray, FitnessValue
from geometry_msgs.msg import Point
from robot_msgs.srv import RobotStatus, RobotStatusResponse
import os
import numpy as np
from pso import PSO_Algorithm
import time

from dataclasses import dataclass
from typing import Tuple

from utilities import log_coordinator_timing

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



def task_callback(msg):
    rospy.loginfo("Received task array")
    global cached_tasks
    global tasks_are_cached
    tasks_are_cached = True
    tasks = []
    for t in msg.tasks:
        task = {
            "drop_initial_coordination" : (t.drop_initial_coordination.x, t.drop_initial_coordination.y),
            "order" : t.order,
            "drop_target_coordination" : (t.drop_target_coordination.x, t.drop_target_coordination.y)
        }
        tasks.append(task)
    cached_tasks = tasks




def universe_callback(msg):
    rospy.loginfo("Received universe array")
    universes = np.array(msg.universe).reshape((msg.rows, msg.cols))
    global cached_universe
    global universe_are_cached
    universe_are_cached = True
    cached_universe = universes
    if robot_status_are_cached and tasks_are_cached and universe_are_cached:
        rospy.loginfo("All data cached — running PSO.")
        best = run_explotation()

        val = FitnessValue(fitness=best)
        rospy.loginfo(f"publishing best fitness of {robot_name}")
        print(val)
        fitness_pub.publish(val)
        rospy.loginfo(f"[{robot_name}] /fitness_value publisher connections: {fitness_pub.get_num_connections()}")

        rospy.sleep(1)


def robot_status_callback(msg):
    rospy.loginfo("getting robot status")
    global robot_status_are_cached
    robot_status_are_cached = True
    global cached_robot_statuses
    cached_robot_statuses = msg.status
    



def handle_status_request(req):
    rospy.loginfo(f"[{robot_name}] Status requested.")
    res = RobotStatusResponse()
    res.robot_name = robot_name
    res.position = Point(x=robot_x, y=robot_y, z=0.0)
    res.battery = battery_level
    return res

"""
def check_and_run(event):
    if robot_status_are_cached and universe_are_cached and tasks_are_cached:
        rospy.loginfo("All data cached — running PSO.")
        best = run_explotation()
        pub = rospy.Publisher('/fitness_value', FitnessValue, queue_size=1)
        val = FitnessValue(fitness=best)
        pub.publish(val)
        check_timer.shutdown()    # stop checking
"""

def robot_node():
    rospy.init_node('robot_node', anonymous=True)

    # Service server: /get_robot_status_<robot_name>
    service_name = f"/get_robot_status_{robot_name}"
    rospy.Service(service_name, RobotStatus, handle_status_request)
    rospy.loginfo(f"[{robot_name}] Ready to respond on service {service_name}")



    rospy.Subscriber('/robot_statuses', RobotStatusArray, robot_status_callback)


    # Subscriber for tasks
    rospy.Subscriber('/robot_tasks', DropTaskArray, task_callback)

    rospy.Subscriber(f'/{robot_name}/universe', UniverseArray , universe_callback)
    global fitness_pub
    fitness_pub = rospy.Publisher('/fitness_value', FitnessValue, queue_size=1)

    rospy.spin()

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
    M = 50  # number of tasks
    
    robot_charge_duration = [(r.battery / 100.0) * 20 * 3600 for r in cached_robot_statuses]

    robot_charge_duration = []
    robots_coord = []
    for r in cached_robot_statuses:
        bat = (r.battery / 100.0) * 20 * 3600
        robot_charge_duration.append(bat)
        point = (r.position.x, r.position.y)
        robots_coord.append(point)


    pso1_d = time.time()
    best_fitness , _ = PSO_Algorithm(MAX_ITERATIONS,len(cached_universe),M,N,s,iterationstop,robot_charge_duration,robots_coord,cached_tasks,Charging_station,CHARGING_TIME, Energy_Harvesting_Rate, cached_universe)
    pso2_d = time.time()
    pso_d = pso2_d - pso1_d
    
    CSV_FILE = 'data/coord_timing_distributed.csv'
    log_coordinator_timing(pso_d, CSV_FILE)

    rospy.loginfo(f"time pso distributed {pso_d}")

    return best_fitness








if __name__ == '__main__':
    robot_node()

