#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd


from mvo import mvo_exploration 
from pso import PSO_Algorithm
from ga import genetic_algorithm


import rospy
from std_msgs.msg import Header, Int32, Time
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

best_fitness = 10000000
first_robot_start = None

fitness_count = 0
Ntest = 1

MM = np.zeros(Ntest)

# exploration time
tMVO = np.zeros(Ntest)


# simulation time
tCMVOGA = np.zeros(Ntest)
tCGA = np.zeros(Ntest)
tDMVOGA = np.zeros(Ntest)
tDGA = np.zeros(Ntest)

bestCMVOGA = np.zeros(Ntest)
bestCGA = np.zeros(Ntest)
bestDMVOGA = np.zeros(Ntest)
bestDGA = np.zeros(Ntest)

bestCMVOGAmax = np.zeros(Ntest)
bestCGAmax = np.zeros(Ntest)
bestDMVOGAmax = np.zeros(Ntest)
bestDGAmax = np.zeros(Ntest)

bestCMVOGAmin = np.zeros(Ntest)
bestCGAmin = np.zeros(Ntest)
bestDMVOGAmin = np.zeros(Ntest)
bestDGAmin = np.zeros(Ntest)


# simulation time
tCMVOPSO = np.zeros(Ntest)
tCPSO    = np.zeros(Ntest)
tDMVOPSO = np.zeros(Ntest)
tDPSO    = np.zeros(Ntest)

bestCMVOPSO    = np.zeros(Ntest)
bestCPSO       = np.zeros(Ntest)
bestDMVOPSO    = np.zeros(Ntest)
bestDPSO       = np.zeros(Ntest)

bestCMVOPSOmax = np.zeros(Ntest)
bestCPSOmax    = np.zeros(Ntest)
bestDMVOPSOmax = np.zeros(Ntest)
bestDPSOmax    = np.zeros(Ntest)

bestCMVOPSOmin = np.zeros(Ntest)
bestCPSOmin    = np.zeros(Ntest)
bestDMVOPSOmin = np.zeros(Ntest)
bestDPSOmin    = np.zeros(Ntest)


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
        #rospy.loginfo(f"Got status from {robot_name}: Position({resp.position.x}, {resp.position.y}), Battery: {resp.battery}%")
        return resp
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed for {robot_name}: {e}")
        return None


def get_robot_info(N):
    robot_names = ['robot1', 'robot2', 'robot3', 'robot4', 'robot5', 'robot6', 'robot7', 
                   'robot8', 'robot9', 'robot10', 'robot11', 'robot12', 'robot13', 'robot14'
                   , 'robot15', 'robot16', 'robot17', 'robot18', 'robot19', 'robot20']  # Example robot names

    robot_names = robot_names[0:N]

    robots_info: List[RobotStatusInfo] = []

    for name in robot_names:
        status = call_robot_status_service(name)
        if status:
            pos = status.position
            robots_info.append(RobotStatusInfo(position=(pos.x, pos.y), battery=status.battery, name=name))
    
    return robots_info

def task_callback(msg):
    #rospy.loginfo("Received task array")
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



    global fitness_count
    global best_fitness
    global solve_time
    global first_robot_start

    m = msg.m
    current_jj = msg.j
    current_ii = msg.i
    if fitness_count == 0:
        first_robot_start = msg.start_time

    recv_time = rospy.Time.now()
    current_solve_time = (recv_time - first_robot_start).to_sec()
    current_fitness = msg.fitness
    fitness_count += 1
    #print(f"this is the current fitness {current_fitness} and jj {current_jj} and m = {m} and fitness count is {fitness_count}")
    if current_fitness < best_fitness:
        best_fitness = current_fitness
        solve_time = current_solve_time 


    
    if fitness_count >= 5:
        fitness_count = 0
        send_time = msg.communication
        communication_time = (recv_time - send_time).to_sec()

        #rospy.loginfo(f"getting robot status in {communication_time}s")

        m = msg.m
        #print(f"this is the current fitness {current_fitness} and jj {current_jj} and {msg.m}" )
        if m == 2:

            if current_ii!=0:
                bestDMVOGA[current_jj]=max(bestDMVOGAmax[current_jj] , best_fitness)
                bestDMVOGAmin[current_jj]=min(bestDMVOGAmin[current_jj] , best_fitness)
                tDMVOGA[current_jj]=(solve_time+tDMVOGA[current_jj]* current_ii) / (current_ii + 1)
                bestDMVOGA[current_jj]=(best_fitness+bestDMVOGA[current_jj]* current_ii) / (current_ii + 1)

            else:
                tDMVOGA[current_jj]=solve_time
                bestDMVOGA[current_jj]=best_fitness
                bestDMVOGAmax[current_jj]=best_fitness
                bestDMVOGAmin[current_jj]=best_fitness

        if m == 3:
            if current_ii!=0:
                bestDGA[current_jj]=max(bestDGAmax[current_jj] , best_fitness)
                bestDGAmin[current_jj]=min(bestDGAmin[current_jj] , best_fitness)
                tDGA[current_jj]=(solve_time+tDGA[current_jj]* current_ii) / (current_ii + 1)
                bestDGA[current_jj]=(best_fitness+bestDGA[current_jj]* current_ii) / (current_ii + 1)

            else:
                tDGA[current_jj]=solve_time
                bestDGA[current_jj]=best_fitness
                bestDGAmax[current_jj]=best_fitness
                bestDGAmin[current_jj]=best_fitness
        
        if m == 6:

            if current_ii!=0:
                bestDMVOPSO[current_jj]=max(bestDMVOPSOmax[current_jj] , best_fitness)
                bestDMVOPSOmin[current_jj]=min(bestDMVOPSOmin[current_jj] , best_fitness)
                tDMVOPSO[current_jj]=(solve_time+tDMVOPSO[current_jj]* current_ii) / (current_ii + 1)
                bestDMVOPSO[current_jj]=(best_fitness+bestDMVOPSO[current_jj]* current_ii) / (current_ii + 1)

            else:
                tDMVOPSO[current_jj]=solve_time
                bestDMVOPSO[current_jj]=best_fitness
                bestDMVOPSOmax[current_jj]=best_fitness
                bestDMVOPSOmin[current_jj]=best_fitness

        if m == 7:
            if current_ii!=0:
                bestDPSO[current_jj]=max(bestDPSOmax[current_jj] , best_fitness)
                bestDPSOmin[current_jj]=min(bestDPSOmin[current_jj] , best_fitness)
                tDPSO[current_jj]=(solve_time+tDPSO[current_jj]* current_ii) / (current_ii + 1)
                bestDPSO[current_jj]=(best_fitness+bestDPSO[current_jj]* current_ii) / (current_ii + 1)

            else:
                tDPSO[current_jj]=solve_time
                bestDPSO[current_jj]=best_fitness
                bestDPSOmax[current_jj]=best_fitness
                bestDPSOmin[current_jj]=best_fitness

        #rospy.signal_shutdown("All fitness values received.")



def coordinator_node():
    rospy.init_node('coordinator_node')
    global current_m, current_jj, current_ii
    global fitness_count, best_fitness, solve_time
    
    m_pub = rospy.Publisher('/task_count', Int32, queue_size=1)

    rospy.Subscriber('/robot_tasks', DropTaskArray, task_callback)
    rospy.sleep(1)


    rospy.Subscriber('/fitness_value', FitnessValue, fitness_value_callback)
    
    M = 100
    N = 5
    for jj in range(Ntest):
        MM[jj] = M
        current_jj = jj
        
        # getting the tasks
        m_pub.publish(Int32(data=M))

        rospy.sleep(1)
        
        # getting the robots location and battery level
        robots_info = get_robot_info(N)
        
        MAX_ITERATIONS, SWARM_SIZE, \
                robot_charge_duration, robots_coord, \
                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate,\
                = init_enivornment(robots_info)
 
    
        for m in range(8):
            current_m = m
            #fitness_count = 0
            best_fitness  = float('inf')
            solve_time    = 0.0
            for ii in range(1):
                current_ii = ii
                #rospy.loginfo(f"we are at test {jj}, m is {m}")

                # mvo + ga centralised
                if m == 0:
                    if len(cached_tasks) > 0:
                        universes, tmvo = explore(MAX_ITERATIONS, SWARM_SIZE, M, N, 
                                robot_charge_duration, robots_coord, cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
                
            
                    tmvoga1 = time.time()
                    best_CMVOGA = exploitation(MAX_ITERATIONS, len(universes), M, N, robot_charge_duration, robots_coord, cached_tasks, 
                            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "ga", universes)
                    tmvoga2 = time.time()
                    tmvoga = tmvoga2 - tmvoga1

                    if ii!=0:
                        bestCMVOGA[jj]=max(bestCMVOGAmax[jj] , best_CMVOGA)
                        bestCMVOGAmin[jj]=min(bestCMVOGAmin[jj] , best_CMVOGA)
                        
                        # tPSOEA[jj] = (tPSOEA[jj] * ii + t_psoea) / (ii + 1)
                        
                        tCMVOGA[jj]=(tmvoga + tCMVOGA[jj] * ii) / (ii + 1)
                        bestCMVOGA[jj]=(best_CMVOGA+bestCMVOGA[jj]* ii) / (ii + 1)

                    else:
                        tCMVOGA[jj]=tmvoga
                        bestCMVOGA[jj]=best_CMVOGA
                        bestCMVOGAmax[jj]=best_CMVOGA
                        bestCMVOGAmin[jj]=best_CMVOGA
                # ga centralised
                if m == 1:
                    tga1 = time.time()
                    best_CGA = exploitation(MAX_ITERATIONS, SWARM_SIZE, M, N, robot_charge_duration, robots_coord, cached_tasks, 
                            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "ga")
                    tga2 = time.time()
                    tga = tga2 - tga1

                    if ii!=0:
                        bestCGA[jj]=max(bestCGAmax[jj] , best_CGA)
                        bestCGAmin[jj]=min(bestCGAmin[jj] , best_CGA)
                        tCGA[jj]=(tga+tCGA[jj] * ii) / (ii + 1)
                        bestCGA[jj]=(best_CGA+bestCGA[jj]* ii) / (ii + 1)

                    else:
                        tCGA[jj]=tga
                        bestCGA[jj]=best_CGA
                        bestCGAmax[jj]=best_CGA
                        bestCGAmin[jj]=best_CGA

                # mvo + ga distributed  
                if m == 2:
                    if len(cached_tasks) > 0:
                        universes, tmvo = explore(MAX_ITERATIONS, SWARM_SIZE, M, N, 
                                robot_charge_duration, robots_coord, cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
                

    
                    
                    array_allocation = allocate_exploration(COORDS, robots_coord, SWARM_SIZE, M)

                    send_parameters(robots_info, cached_tasks, universes, array_allocation, num_cols=M)
                
                # ga distributed
                if m == 3:

                    
                    array_allocation = allocate_exploration(COORDS, robots_coord, SWARM_SIZE, M)

                    send_parameters(robots_info, cached_tasks, universes, array_allocation, num_cols=M)
                    



                # mvo + pso centralised
                if m == 4:
                    if len(cached_tasks) > 0:
                        universes, tmvo = explore(MAX_ITERATIONS, SWARM_SIZE, M, N, 
                                robot_charge_duration, robots_coord, cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
                
            
                    tmvopso1 = time.time()
                    best_CMVOPSO = exploitation(MAX_ITERATIONS, len(universes), M, N, robot_charge_duration, robots_coord, cached_tasks, 
                            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "pso", universes)
                    tmvopso2 = time.time()
                    tmvopso = tmvopso2 - tmvopso1

                    if ii!=0:
                        bestCMVOPSO[jj]=max(bestCMVOPSOmax[jj] , best_CMVOPSO)
                        bestCMVOPSOmin[jj]=min(bestCMVOPSOmin[jj] , best_CMVOPSO)
                        tCMVOPSO[jj]=(tmvopso+tCMVOPSO[jj]* ii) / (ii + 1)
                        bestCMVOPSO[jj]=(best_CMVOPSO+bestCMVOPSO[jj]* ii) / (ii + 1)

                    else:
                        tCMVOPSO[jj]=tmvopso
                        bestCMVOPSO[jj]=best_CMVOPSO
                        bestCMVOPSOmax[jj]=best_CMVOPSO
                        bestCMVOPSOmin[jj]=best_CMVOPSO
                # pso centralised
                if m == 5:
                    tpso1 = time.time()
                    best_CPSO = exploitation(MAX_ITERATIONS, SWARM_SIZE, M, N, robot_charge_duration, robots_coord, cached_tasks, 
                            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "pso")
                    tpso2 = time.time()
                    tpso = tpso2 - tpso1

                    if ii!=0:
                        bestCPSO[jj]=max(bestCPSOmax[jj] , best_CPSO)
                        bestCPSOmin[jj]=min(bestCPSOmin[jj] , best_CPSO)
                        tCPSO[jj]=(tpso+tCPSO[jj]* ii) / (ii + 1)
                        bestCPSO[jj]=(best_CPSO+bestCPSO[jj]* ii) / (ii + 1)

                    else:
                        tCPSO[jj]=tpso
                        bestCPSO[jj]=best_CPSO
                        bestCPSOmax[jj]=best_CPSO
                        bestCPSOmin[jj]=best_CPSO

                # mvo + pso distributed  
                if m == 6:
                    if len(cached_tasks) > 0:
                        universes, tmvo = explore(MAX_ITERATIONS, SWARM_SIZE, M, N, 
                                robot_charge_duration, robots_coord, cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
                
        



                    array_allocation = allocate_exploration(COORDS, robots_coord, SWARM_SIZE, len(cached_tasks))

                    send_parameters(robots_info, cached_tasks, universes, array_allocation, num_cols=M)
                # pso distributed
                if m == 7:

                    
                    array_allocation = allocate_exploration(COORDS, robots_coord, SWARM_SIZE, len(cached_tasks))

                    send_parameters(robots_info, cached_tasks, universes, array_allocation, num_cols=M)
                    
                rospy.sleep(3)
                #CSV_FILE = 'data/coord_timing_central.csv'
                #log_coordinator_timing(pso_time=pso_c, CSV_FILE=CSV_FILE)

        N = N + 1


    
    directory = "data/plots"
    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(MM, tCMVOGA, marker="o", linestyle="-", color="red", label="tCMVOGA")
    plt.plot(MM, tCGA, marker="*", linestyle="-", color="green", label="tCGA")

    #plt.plot(MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
    #plt.plot(MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

    plt.plot(MM, tDMVOGA, marker="o", linestyle="-", color="blue", label="mtDMVOGA")
    plt.plot(MM, tDGA, marker="*", linestyle="-", color="olive", label="tDGA")

    plt.xlabel("M")
    plt.ylabel("Values")
    plt.title("tGA against M")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directory, 'tGA'))


    plt.figure(figsize=(10, 6))
    plt.plot(MM, bestCMVOGA, marker="o", linestyle="-", color="red", label="best_CMVOGA")
    plt.plot(MM, bestCGA, marker="*", linestyle="-", color="green", label="best_CGA")

    #plt.plot(MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
    #plt.plot(MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

    plt.plot(MM, bestDMVOGA, marker="o", linestyle="-", color="blue", label="best_DMVOGA")
    plt.plot(MM, bestDGA, marker="*", linestyle="-", color="olive", label="best_DGA")

    plt.xlabel("M")
    plt.ylabel("Values")
    plt.title("bestGA against M")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directory, 'bestGA'))

    plt.figure(figsize=(10, 6))
    plt.plot(MM, tCMVOPSO, marker="o", linestyle="-", color="red", label="tCMVOPSO")
    plt.plot(MM, tCPSO, marker="*", linestyle="-", color="green", label="tCPSO")

    #plt.plot(MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
    #plt.plot(MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

    plt.plot(MM, tDMVOPSO, marker="o", linestyle="-", color="blue", label="mtDMVOPSO")
    plt.plot(MM, tDPSO, marker="*", linestyle="-", color="olive", label="tDPSO")

    plt.xlabel("M")
    plt.ylabel("Values")
    plt.title("tPSO against M")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directory, 'tPSO'))


    plt.figure(figsize=(10, 6))
    plt.plot(MM, bestCMVOPSO, marker="o", linestyle="-", color="red", label="best_CMVOPSO")
    plt.plot(MM, bestCPSO, marker="*", linestyle="-", color="green", label="best_CPSO")

    #plt.plot(MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
    #plt.plot(MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

    plt.plot(MM, bestDMVOPSO, marker="o", linestyle="-", color="blue", label="best_DMVOPSO")
    plt.plot(MM, bestDPSO, marker="*", linestyle="-", color="olive", label="best_DPSO")

    plt.xlabel("M")
    plt.ylabel("Values")
    plt.title("bestPSO against M")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directory, 'bestPSO'))
    
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
    #print(array_allocation)
    #print(len(universes))
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
        msg.m = current_m
        msg.j = current_jj
        msg.i = current_ii

        pub = publishers[robot.name]

        # Wait for at least one subscriber
        while pub.get_num_connections() < 1:
            rospy.logwarn(f"Waiting for subscriber on {robot.name}/parameters...")
            rospy.sleep(2)


        
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


    

    return MAX_ITERATIONS, SWARM_SIZE, robot_charge_duration, robots_coord, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate

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

    # Bits per array
    bits_per_array = compute_message_size(num_candidates, total_arrays) * BITS_PER_BYTE

    # Energy cost per robot
    energy_costs = [energy_to_transmit(bits_per_array, d) for d in distances]

    # Inverse‐cost weights
    inv_costs = [1 / e for e in energy_costs]
    total_inv = sum(inv_costs)
    weights = [ic / total_inv for ic in inv_costs]

    # Floating‐point allocations
    exact = [num_candidates * w for w in weights]
    # Base = floor of each
    base = [int(e) for e in exact]
    # Remainder to distribute
    remainder = num_candidates - sum(base)

    # Distribute the remainder by largest fractional parts
    fracs = [e - b for e, b in zip(exact, base)]
    for i in sorted(range(len(fracs)), key=lambda i: fracs[i], reverse=True)[:remainder]:
        base[i] += 1

    array_allocations = base

    # (Optional) compute message sizes for each allocation
    message_sizes_bytes = [
        compute_message_size(num_candidates, a) for a in array_allocations
    ]

    

    return array_allocations




def explore(MAX_ITERATIONS, swarm_size, M, N, robot_charge_duration, robots_coord, task, 
            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate):
    
    start_exp = time.time()
    best_MVOEA, mvo_curve, init_swarm = mvo_exploration(
            MAX_ITERATIONS=MAX_ITERATIONS,
            SWARM_SIZE=swarm_size,
            M=M, N=N, 
            robot_charge_duration=robot_charge_duration,
            robots_coord=robots_coord,
            task=task,
            Charging_station=Charging_station,
            CHARGING_TIME=CHARGING_TIME,
            Energy_Harvesting_Rate=Energy_Harvesting_Rate
        )
    end_exp = time.time()
    tmvo = end_exp - start_exp

    return init_swarm, tmvo



def exploitation(MAX_ITERATIONS, swarm_size, M, N,  robot_charge_duration, robots_coord, task, 
            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, method, init_swarm=[]):
    
    if method == "pso":
        best, _ = PSO_Algorithm(
                MAX_ITERATIONS=MAX_ITERATIONS,
                SWARM_SIZE=swarm_size,
                M=M, N=N, 
                iterationstop=15,
                robot_charge_duration=robot_charge_duration,
                robots_coord=robots_coord,
                task=task,
                Charging_station=Charging_station,
                CHARGING_TIME=CHARGING_TIME,
                Energy_Harvesting_Rate=Energy_Harvesting_Rate,
                init_swarm=init_swarm
            )
    if method == "ga":
        best, _ = genetic_algorithm(
                POP_SIZE=swarm_size,
                M=M, N=N,iteration=MAX_ITERATIONS,
                iterationstop=15,
                robot_charge_duration=robot_charge_duration,
                robots_coord=robots_coord,
                task=task,
                Charging_station=Charging_station,
                CHARGING_TIME=CHARGING_TIME,
                Energy_Harvesting_Rate=Energy_Harvesting_Rate,
                init_swarm=init_swarm
        )

    return best
    

if __name__ == '__main__':
    coordinator_node()



