#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import os
# import py_make_goal_client.src.make_plan_client as make_plan_client
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from pso import PSO_Algorithm
from coordinator_pkg.msg import NumpyArray, DropTaskArray, DropTask

from mvo import mvo_exploration 

def callback(msg):
    rospy.loginfo(f"[Coordinator] Heard: {msg.data}")

def coordinator_node():
    rospy.init_node('coordinator_node')
    pub = rospy.Publisher('/robot_tasks', DropTaskArray, queue_size=10)

    rospy.sleep(1)  # Allow time for connections to establish

    task_array_msg = DropTaskArray()

    M, coords, _ = init_enivornment()
    tasks = get_task(M, coords)

    for t in tasks:
        task_msg = DropTask()
        task_msg.drop_initial_coordination = list(t['drop_initial_coordination'])
        task_msg.drop_target_coordination = list(t['drop_target_coordination'])
        task_msg.order = t['order']
        task_array_msg.tasks.append(task_msg)

    rospy.loginfo("Publishing task list once...")
    pub.publish(task_array_msg)
    
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        rospy.loginfo("Coordinator node is now idle and waiting...")
        rate.sleep()
        
    



def init_enivornment():
    # random.seed(45)
    coords = np.array(
        [
            (1.8, 3),
            (1.8, 4),
            (1.8, 5),
            (1.8, 6),
            (1.8, 7),

            (1.8, 8),
            (-0.5, -8),
            (-0.5, -9),
            (-0.5, -7),
            (-0.5, 1.8),
            (-0.5, 3),
            (-0.5, 4),
            (-0.5, 5),
            (-0.5, 6),
            (3.5, 1.5),
            (5.5, 1.5),
            (3.5, 2.5),
            (5.5, 2.5),
            (3.5, -2.5),
            (5.5, -2.5),
            (3.5, -4),
            (5.5, -4),
            (3.5, -5.6),
            (5.5, -5.6),
            (3.5, -10),
            (5.5, -10),
            (3.5, -7.8),
            (5.5, -7.8),
            (-4.5, -9),
            (-4.5, -9),
            (-4.5, -8),
            (-4.5, -7),
            (-4.5, -6),
            (-4.5, -5),
            (-4.5, -4),
            (-4.5, -3),
            (-4.5, -2),
            (-4.5, -1),
            (-4.5, 0),
            (-4.5, 1),
            (-4.5, 2),
            (-4.5, 3),
            (-4.5, 4),
            (-4.5, 5),
            (-4.5, 6),
            (-4.5, 7),
            (-4.5, 8),
            (-4.5, 9),
            (-3, 1),
            (-3, 1),
            (-3, 2),
            (-3, 3),
            (-3, 4),
            (-3, 5),
            (-3, 6),
            (-3, 7),
            (-3, 8),
            (-3, 9),
        ]
    )
    # N = 10
    # M = 200
    S = 1  # number of Charge stations
    Energy_Harvesting_Rate = 3
    MAX_CHARGE = 20
    CHARGING_RATE = 10 / 3600  # w/s
    DECHARGING_RATE = 5 / 3600  # w/s
    DECHARGING_Time = 4 * 3600  # s
    CHARGING_TIME = 0.5 * 3600  # s
    iterationstop = 15 
    # initialize the Random location and charging
    # recieved tasks order
    Charging_station = (-1.6, 7.2)
    SWARM_SIZE = 50
    POP_SIZE = 10
    MAX_ITERATIONS = 50


    def find_last_occurrence(arr, element, ind):
        last_index = -1
        for i in range(ind + 1):
            if arr[i] == element:
                last_index = i
        return last_index


    N = 8  # number of robots
    M = 70  # number of tasks
    Ntests = 10
    MM = np.zeros(Ntests)


    # robot charge and coordination initialization
    robot_charge_duration = [(random.uniform(0, 20 * 3600)) for i in range(N)]


    print(
        "Robot's Initial Charge percentage:",
        np.array((robot_charge_duration)) / (20 * 3600) * 100,
    )

    robot_charge_duration = [
        (duration * 20 * 3600) / 100 for duration in robot_charge_duration
    ]
    print(robot_charge_duration)


    robots_coord = np.zeros((N, len(coords[0])))
    for _ in range(N):
        robots_coord[_] = random.choice(coords)

    print(f"robot cooord is {robots_coord}")

    


    best_fitness = 1000000
    best_fitness2 = 1000000
    # Run MVO       
    tmvo1 = time.time()
    # universes = np.zeros((25*2, M), dtype=int)
    def initialize_population(pop_size, array_size, value_range):
        return np.random.randint(0, value_range, size=(pop_size, array_size)) 

    init_universe = initialize_population(50,M,N)

    return M, coords, init_universe

    explore(MAX_ITERATIONS, SWARM_SIZE, M, N, 1 , iterationstop,
            robot_charge_duration, robots_coord, task, 
            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, init_universe)


def get_task(M, coords):
    task = []
    for _ in range(M):
        drop_initial = random.choice(coords)
        order = "parallel"
        drop_target = random.choice(coords)
        tasks = {
            "drop_initial_coordination": drop_initial,
            "order": order,
            "drop_target_coordination": drop_target,
        }
        task.append(tasks)
    return task

def explore(MAX_ITERATIONS, swarm_size, M, N, s, iterationstop, robot_charge_duration, robots_coord, task, 
            Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, init_universe):
    

    best_MVOEA, mvo_curve, init_swarm, universe = mvo_exploration(
            MAX_ITERATIONS=MAX_ITERATIONS,
            SWARM_SIZE=swarm_size,
            M=M, N=N, s=s,
            iterationstop=iterationstop,
            robot_charge_duration=robot_charge_duration,
            robots_coord=robots_coord,
            task=task,
            Charging_station=Charging_station,
            CHARGING_TIME=CHARGING_TIME,
            Energy_Harvesting_Rate=Energy_Harvesting_Rate, init_universe=init_universe
        )
        # `universe` is shape (2, M) → two elites
    print(f"previous best fitness {best_fitness}")
    if best_MVOEA < best_fitness:
        best_fitness = best_MVOEA
        init_swarm = universe
        print(f"new best fitness {best_fitness}")
    return init_swarm



if __name__ == '__main__':
    coordinator_node()



