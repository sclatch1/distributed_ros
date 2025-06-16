#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd


from mvo import mvo_exploration 
from pso import PSO_Algorithm


import rospy
from std_msgs.msg import Header, Int32
from robot_msgs.msg import DropTask, DropTaskArray, UniverseArray, RobotStatusArray, RobotStatusM, FitnessValue, Parameters
from geometry_msgs.msg import Point
from robot_msgs.srv import RobotStatus

from dataclasses import dataclass
from typing import List, Tuple

from utilities import log_coordinator_timing, write_to_file, write_np_to_file

E_ELEC = 50e-9     # energy per bit for electronics (J)
E_AMP = 100e-12    # energy per bit per m^2 (J)
PATH_LOSS_EXPONENT = 2
BYTES_PER_INT = 4
BITS_PER_BYTE = 8

COORDS = (0,0)

current_best = 10000

explore = os.getenv("EXPLORE", "false").lower() == "true"
distributed = os.getenv("DISTRIBUTED", "false").lower() == "true"


fitness_count = 0

@dataclass
class RobotStatusInfo:
    position: Tuple[float, float]
    battery: float
    name: str


def call_robot_status_service(robot_name):
    service_name = f"/get_robot_status_{robot_name}"
    rospy.wait_for_service(service_name)
    try:
        get_status = rospy.ServiceProxy(service_name, RobotStatus)
        resp = get_status()  # No arguments in your RobotStatus.srv request
        rospy.loginfo(f"Got status from {robot_name}: Position({resp.position.x}, {resp.position.y}), Battery: {resp.battery}%")
        return resp
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed for {robot_name}: {e}")
        return None


def get_robot_info():
    robot_names = ['robot1', 'robot2', 'robot3', 'robot4', 'robot5', 'robot6']  # Example robot names

    robots_info: List[RobotStatusInfo] = []

    for name in robot_names:
        status = call_robot_status_service(name)
        if status:
            pos = status.position
            robots_info.append(RobotStatusInfo(position=(pos.x, pos.y), battery=status.battery, name=name))
    
    return robots_info

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

def fitness_value_callback(msg):
    global current_best

    if current_best > msg.fitness:
        current_best = msg.fitness

    rospy.loginfo(f"current best_fitness is {current_best}")
    rospy.loginfo(f"best_fitness is {msg.fitness}")

    global fitness_count
    fitness_count += 1

    if fitness_count == 6:
        recv_time = rospy.Time.now()
        send_time = msg.header.stamp
        communication_robot_status = (recv_time - send_time).to_sec()
        rospy.loginfo(f"getting robot status in {communication_robot_status}s")
        #rospy.signal_shutdown("All fitness values received.")



def coordinator_node():
    rospy.init_node('coordinator_node')
    
    m_pub = rospy.Publisher('/task_count', Int32, queue_size=1)

    rospy.Subscriber('robot_tasks', DropTaskArray, task_callback)
    rospy.sleep(1)

    if distributed:
            rospy.Subscriber('/fitness_value', FitnessValue, fitness_value_callback)
    M = 50
    for _ in range(10):
        
        m_pub.publish(Int32(data=M))

        
        

        robots_info = get_robot_info()

        print(f"lenght of cached takss            {len(cached_tasks)}")
        
        MAX_ITERATIONS, SWARM_SIZE, N, \
        robot_charge_duration, robots_coord, \
        Charging_station, CHARGING_TIME, Energy_Harvesting_Rate,\
        = init_enivornment(robots_info)

        if len(cached_tasks) > 0:
            universes = explore(MAX_ITERATIONS, SWARM_SIZE, M, N, 1,
                    robot_charge_duration, robots_coord, cached_tasks, 
                    Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
        

        pso_s1 = time.time()
        best_fitness_pso = exploitation(MAX_ITERATIONS, len(universes), M, N, 1, robot_charge_duration, robots_coord, cached_tasks, 
                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
        pso_s2 = time.time()
        pso_s = pso_s2 - pso_s1
        
        print(f"this is pso solo best fitness: {best_fitness_pso} with time: {pso_s}")


        pso1_c = time.time()
        best_fitness = exploitation(MAX_ITERATIONS, len(universes), M, N, 1, robot_charge_duration, robots_coord, cached_tasks, 
                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, universes)
        pso2_c = time.time()
        pso_c = pso2_c - pso1_c

        
        CSV_FILE = 'data/coord_timing_central.csv'
        log_coordinator_timing(pso_time=pso_c, CSV_FILE=CSV_FILE)


        rospy.loginfo(f"best fitness centralised: {best_fitness} and time {pso_c}")

        if distributed:
            array_allocation = allocate_exploration(COORDS, robots_coord, 50, 50)

            send_parameters(robots_info, cached_tasks, universes, array_allocation, num_cols=M)

        M = M + 5
    rospy.spin()


def send_parameters(robots_info, tasks, universes, array_allocation, num_cols):
    robot_status_array = []
    for s in robots_info:
        robot_status = RobotStatusM()
        robot_status.robot_name = s.name
        robot_status.position = Point(x=s.position[0], y=s.position[1], z=0.0)
        robot_status.battery = s.battery
        robot_status_array.append(robot_status)
    
    task_array_msg =[ ]
    for t in tasks:
        task_msg = DropTask()
        task_msg.drop_initial_coordination = Point(x=t['drop_initial_coordination'][0], y=t['drop_initial_coordination'][1], z=0.0)
        task_msg.drop_target_coordination = Point(x=t['drop_target_coordination'][0], y=t['drop_target_coordination'][1], z=0.0)
        task_msg.order = t['order']
        task_array_msg.append(task_msg)


    num_allocated_total = sum(array_allocation)
    assert num_allocated_total <= len(universes), "Allocations exceed universe size"

    # Initialize all publishers once
    publishers = {}
    for robot in robots_info:
        topic = f"/{robot.name}/parameters"
        publishers[robot.name] = rospy.Publisher(topic, Parameters, queue_size=10)
    


    flat_index = 0  # pointer in universes
    for i, robot in enumerate(robots_info):
        rows = array_allocation[i]
        universe_slice = universes[flat_index:flat_index + rows]
        flat_index += rows

        # Flatten the 2D slice into 1D
        flat_data = [int(cell) for row in universe_slice for cell in row]

        # Create UniverseArray message
        msg = Parameters()
        msg.universe = flat_data
        msg.rows = rows
        msg.cols = num_cols

        msg.status = robot_status_array
        msg.tasks = task_array_msg

        pub = publishers[robot.name]

        # Wait for at least one subscriber
        while pub.get_num_connections() == 0:
            rospy.logwarn(f"Waiting for subscriber on {robot.name}/parameters...")
            rospy.sleep(0.05)

        rospy.loginfo(f"Publishing {rows}x{num_cols} universe and tasks and status to {robot.name}")
        
        msg.header = Header(stamp=rospy.Time.now())
        pub.publish(msg)
        

    

   

def init_enivornment(robots_info: List[RobotStatusInfo]):
    # Use actual coordinates from robots_info instead of hardcoded coords
    print("initializing environment ...")
    coords = np.array([r.position for r in robots_info])

    S = 1  # number of Charge stations
    Energy_Harvesting_Rate = 3
    MAX_CHARGE = 20
    CHARGING_RATE = 10 / 3600  # w/s
    DECHARGING_RATE = 5 / 3600  # w/s
    DECHARGING_Time = 4 * 3600  # s
    CHARGING_TIME = 0.5 * 3600  # s
    iterationstop = 15
    Charging_station = (-1.6, 7.2)
    SWARM_SIZE = 50
    POP_SIZE = 10
    MAX_ITERATIONS = 50




    N = len(robots_info)  # number of robots
    M = len(cached_tasks)




    # Instead of random charge duration, calculate from battery percentage
    # Assuming battery is in percentage (0 to 100), scale it to seconds of charge (0 to 20*3600)
    robot_charge_duration = [(r.battery / 100.0) * 20 * 3600 for r in robots_info]
    robots_coord = coords


    

    return MAX_ITERATIONS, SWARM_SIZE, N,robot_charge_duration, robots_coord, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate

def compute_message_size(num_candidates, num_arrays):
    """Returns total message size in bytes"""
    return num_candidates * num_arrays * BYTES_PER_INT + 2 * BYTES_PER_INT  # +2 for rows/cols

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def energy_to_transmit(bits, distance):
    return bits * (E_ELEC + E_AMP * distance ** PATH_LOSS_EXPONENT)

def allocate_exploration(central_position, robot_positions, num_candidates, total_arrays):
    # Compute distances
    distances = [compute_distance(central_position, rp) for rp in robot_positions]

    # Assume 1 array per energy cost for now (we scale up later)
    bits_per_array = compute_message_size(num_candidates, 1) * BITS_PER_BYTE

    # Compute energy cost per array per robot
    energy_costs = [energy_to_transmit(bits_per_array, d) for d in distances]

    # Inverse energy cost weighting
    inv_costs = [1 / e for e in energy_costs]
    total_inv = sum(inv_costs)
    weights = [ic / total_inv for ic in inv_costs]

    # Allocate arrays per robot
    array_allocations = [round(total_arrays * w) for w in weights]

    # Compute message sizes for the allocated arrays
    message_sizes_bytes = [compute_message_size(num_candidates, a) for a in array_allocations]

    """
    # Output results
    for i, (d, e, w, a, sz) in enumerate(zip(distances, energy_costs, weights, array_allocations, message_sizes_bytes)):
        print(f"Robot {i+1}:")
        print(f"  Distance: {d:.2f} m")
        print(f"  Energy per array: {e*1e6:.2f} µJ")
        print(f"  Allocation weight: {w:.2f}")
        print(f"  Arrays assigned: {a}")
        print(f"  Message size: {sz} bytes\n")
    """
    return array_allocations



def explore(MAX_ITERATIONS, swarm_size, M, N, s, robot_charge_duration, robots_coord, task, 
            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate):
    
    best_MVOEA, mvo_curve, init_swarm = mvo_exploration(
            MAX_ITERATIONS=MAX_ITERATIONS,
            SWARM_SIZE=swarm_size,
            M=M, N=N, s=s,
            robot_charge_duration=robot_charge_duration,
            robots_coord=robots_coord,
            task=task,
            Charging_station=Charging_station,
            CHARGING_TIME=CHARGING_TIME,
            Energy_Harvesting_Rate=Energy_Harvesting_Rate
        )
    

    return init_swarm



def exploitation(MAX_ITERATIONS, swarm_size, M, N, s, robot_charge_duration, robots_coord, task, 
            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, init_swarm=[]):
    

    best_PSOEA, _ = PSO_Algorithm(
            MAX_ITERATIONS=MAX_ITERATIONS,
            SWARM_SIZE=swarm_size,
            M=M, N=N, s=s,
            iterationstop=15,
            robot_charge_duration=robot_charge_duration,
            robots_coord=robots_coord,
            task=task,
            Charging_station=Charging_station,
            CHARGING_TIME=CHARGING_TIME,
            Energy_Harvesting_Rate=Energy_Harvesting_Rate,
            init_swarm=init_swarm
        )
    return best_PSOEA
    

if __name__ == '__main__':
    coordinator_node()



