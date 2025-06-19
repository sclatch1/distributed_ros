#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import random

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


@dataclass
class RobotStatusInfo:
    position: Tuple[float, float]
    battery: float
    name: str




class CoordinatorNode:
    def __init__(self):
        self.coordinator_coords = (0,0)
        self.timings = {}
        self.Ntest = 15
        self.MM = np.zeros(self.Ntest)

        # parameters for functions
        self.current_jj = 0
        self.current_ii = 0
        self.current_m = 0

        # cache for tasks
        self.cached_tasks = []

        # fitness callback variables
        self.fitnesses = {}
        self.start_time = None
        self.best_fitness = None
        self.solve_time = None
        

        # ROS init
        rospy.init_node('coordinator_node')

        # number of task publisher 
        self.m_pub = rospy.Publisher('/task_count', Int32, queue_size=1)

        # subcriber to robot task
        rospy.Subscriber('/robot_tasks', DropTaskArray, self.task_callback)
        rospy.sleep(1)


        rospy.Subscriber('/fitness_value', FitnessValue, self.fitness_value_callback)

        # run() variables
        self.M = 10
        self.N = 6
        self.MM = np.zeros(self.Ntest)


        self.allocator = ExplorationAllocator(BYTES_PER_INT,BITS_PER_BYTE,E_ELEC,E_AMP,PATH_LOSS_EXPONENT)


        # exploration time
        self.tMVO = np.zeros(self.Ntest)


        # completion time
        self.tCMVOGA = np.zeros(self.Ntest)
        self.tCGA = np.zeros(self.Ntest)
        self.tDMVOGA = np.zeros(self.Ntest)
        self.tDGA = np.zeros(self.Ntest)

        # simulation time
        self.bestCMVOGA = np.zeros(self.Ntest)
        self.bestCGA = np.zeros(self.Ntest)
        self.bestDMVOGA = np.zeros(self.Ntest)
        self.bestDGA = np.zeros(self.Ntest)

        self.bestCMVOGAmax = np.zeros(self.Ntest)
        self.bestCGAmax = np.zeros(self.Ntest)
        self.bestDMVOGAmax = np.zeros(self.Ntest)
        self.bestDGAmax = np.zeros(self.Ntest)

        self.bestCMVOGAmin = np.zeros(self.Ntest)
        self.bestCGAmin = np.zeros(self.Ntest)
        self.bestDMVOGAmin = np.zeros(self.Ntest)
        self.bestDGAmin = np.zeros(self.Ntest)


        # completion time
        self.tCMVOPSO = np.zeros(self.Ntest)
        self.tCPSO    = np.zeros(self.Ntest)
        self.tDMVOPSO = np.zeros(self.Ntest)
        self.tDPSO    = np.zeros(self.Ntest)

        # simulation time
        self.bestCMVOPSO    = np.zeros(self.Ntest)
        self.bestCPSO       = np.zeros(self.Ntest)
        self.bestDMVOPSO    = np.zeros(self.Ntest)
        self.bestDPSO       = np.zeros(self.Ntest)

        self.bestCMVOPSOmax = np.zeros(self.Ntest)
        self.bestCPSOmax    = np.zeros(self.Ntest)
        self.bestDMVOPSOmax = np.zeros(self.Ntest)
        self.bestDPSOmax    = np.zeros(self.Ntest)

        self.bestCMVOPSOmin = np.zeros(self.Ntest)
        self.bestCPSOmin    = np.zeros(self.Ntest)
        self.bestDMVOPSOmin = np.zeros(self.Ntest)
        self.bestDPSOmin    = np.zeros(self.Ntest)


        self.communication_time = np.zeros(self.Ntest)
        self.communication_time1 = np.zeros(self.Ntest)

    def reset_iteration_data(self):
    # Resetting per-iteration variables
        self.fitnesses.clear()
        self.start_time = None
        self.best_fitness = None
        #rospy.loginfo(f"resetting {self.fitnesses} {self.start_time} {self.best_fitness}")


    def call_robot_status_service(self, robot_name):
        service_name = f"/get_robot_status_{robot_name}"
        #rospy.loginfo("calling robot status")
        rospy.wait_for_service(service_name)
        try:
            start = rospy.Time.now()
            get_status = rospy.ServiceProxy(service_name, RobotStatus)
            end = rospy.Time.now()

            self.timings[robot_name] = (end - start).to_sec()

            if len(self.timings) == self.N:
                self.communication_time[self.current_jj] = max(self.timings.values())
                self.timings.clear()
            
            #rospy.loginfo(f"this is the com time: {self.communication_time}")
            resp = get_status() 
            #rospy.loginfo(f"Got status from {robot_name}: Position({resp.position.x}, {resp.position.y}), Battery: {resp.battery}%")
            return resp
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for {robot_name}: {e}")
            return None


    def get_robot_info(self):
        robot_names = ['robot1', 'robot2', 'robot3', 'robot4', 'robot5', 'robot6', 'robot7', 
                    'robot8', 'robot9', 'robot10', 'robot11', 'robot12', 'robot13', 'robot14'
                    , 'robot15', 'robot16', 'robot17', 'robot18', 'robot19', 'robot20']  # Example robot names

        
        robot_names = robot_names[0:self.N]

        robots_info: List[RobotStatusInfo] = []

        for name in robot_names:
            status = self.call_robot_status_service(name)
            if status:
                pos = status.position
                robots_info.append(RobotStatusInfo(position=(pos.x, pos.y), battery=status.battery, name=name))
        random.shuffle(robots_info)

        return robots_info


    def task_callback(self,msg):
        #rospy.loginfo("Received task array")
        tasks = []
        for t in msg.tasks:
            task = {
                "drop_initial_coordination" : (t.drop_initial_coordination.x, t.drop_initial_coordination.y),
                "order" : t.order,
                "drop_target_coordination" : (t.drop_target_coordination.x, t.drop_target_coordination.y)
            }
            tasks.append(task) 
        self.cached_tasks = tasks


    def fitness_value_callback(self, msg):
        recv_time = rospy.Time.now()
        m = msg.m
        current_jj = msg.j
        current_ii = msg.i
        rospy.loginfo(f"this is jj {current_jj} and m is {m} solve {(recv_time - msg.start_time).to_sec()} " )


        self.fitnesses[recv_time] = msg.fitness

        if self.start_time is None or msg.start_time < self.start_time:
            self.start_time = msg.start_time

        if self.best_fitness is None or msg.fitness < self.best_fitness:
            self.best_fitness = msg.fitness
            self.solve_time = (recv_time - self.start_time).to_sec()


        if len(self.fitnesses) == self.N:            
            #rospy.loginfo(f"this is jj {current_jj} and m is {m} and fitness is {self.best_fitness} solve {self.solve_time} " )

            if m == 2:

                if current_ii!=0:
                    self.bestDMVOGA[current_jj]=max(self.bestDMVOGAmax[current_jj] , self.best_fitness)
                    self.bestDMVOGAmin[current_jj]=min(self.bestDMVOGAmin[current_jj] , self.best_fitness)
                    self.tDMVOGA[current_jj]=(self.solve_time+self.tDMVOGA[current_jj]* current_ii) / (current_ii + 1)
                    self.bestDMVOGA[current_jj]=(self.best_fitness+self.bestDMVOGA[current_jj]* current_ii) / (current_ii + 1)

                else:
                    self.tDMVOGA[current_jj]=self.solve_time
                    self.bestDMVOGA[current_jj]=self.best_fitness
                    self.bestDMVOGAmax[current_jj]=self.best_fitness
                    self.bestDMVOGAmin[current_jj]=self.best_fitness

            elif m == 3:
                if current_ii!=0:
                    self.bestDGA[current_jj]=max(self.bestDGAmax[current_jj] , self.best_fitness)
                    self.bestDGAmin[current_jj]=min(self.bestDGAmin[current_jj] , self.best_fitness)
                    self.tDGA[current_jj]=(self.solve_time+self.tDGA[current_jj]* current_ii) / (current_ii + 1)
                    self.bestDGA[current_jj]=(self.best_fitness+self.bestDGA[current_jj]* current_ii) / (current_ii + 1)

                else:
                    self.tDGA[current_jj]=self.solve_time
                    self.bestDGA[current_jj]=self.best_fitness
                    self.bestDGAmax[current_jj]=self.best_fitness
                    self.bestDGAmin[current_jj]=self.best_fitness
            
            elif m == 6:
                if current_ii!=0:
                    self.bestDMVOPSO[current_jj]=max(self.bestDMVOPSOmax[current_jj] , self.best_fitness)
                    self.bestDMVOPSOmin[current_jj]=min(self.bestDMVOPSOmin[current_jj] , self.best_fitness)
                    self.tDMVOPSO[current_jj]=(self.solve_time+self.tDMVOPSO[current_jj]* current_ii) / (current_ii + 1)
                    self.bestDMVOPSO[current_jj]=(self.best_fitness+self.bestDMVOPSO[current_jj]* current_ii) / (current_ii + 1)

                else:
                    self.tDMVOPSO[current_jj]=self.solve_time
                    self.bestDMVOPSO[current_jj]=self.best_fitness
                    self.bestDMVOPSOmax[current_jj]=self.best_fitness
                    self.bestDMVOPSOmin[current_jj]=self.best_fitness

            elif m == 7:
                if current_ii!=0:
                    self.bestDPSO[current_jj]=max(self.bestDPSOmax[current_jj] , self.best_fitness)
                    self.bestDPSOmin[current_jj]=min(self.bestDPSOmin[current_jj] , self.best_fitness)
                    self.tDPSO[current_jj]=(self.solve_time+self.tDPSO[current_jj]* current_ii) / (current_ii + 1)
                    self.bestDPSO[current_jj]=(self.best_fitness+self.bestDPSO[current_jj]* current_ii) / (current_ii + 1)

                else:
                    self.tDPSO[current_jj]=self.solve_time
                    self.bestDPSO[current_jj]=self.best_fitness
                    self.bestDPSOmax[current_jj]=self.best_fitness
                    self.bestDPSOmin[current_jj]=self.best_fitness
                rospy.sleep(2)
                """
                rospy.loginfo(f"BESTtDMVOPSO {self.tDMVOPSO}")
                rospy.loginfo(f"BESTtDMVOGA {self.tDMVOGA}")
                rospy.loginfo(f"BESTtDPSO {self.tDPSO}")
                rospy.loginfo(f"BESTtDGA {self.tDGA}")
                rospy.loginfo(f"BESTtCMVOPSO {self.tCMVOPSO}")
                rospy.loginfo(f"BESTtCMVOGA {self.tCMVOGA}")
                rospy.loginfo(f"BESTtCPSO {self.tCPSO}")
                rospy.loginfo(f"BESTtCGA {self.tCGA}")
                """
            self.reset_iteration_data()
            


            #rospy.signal_shutdown("All fitness values received.")


    def run(self):        
        for jj in range(self.Ntest):
            self.MM[jj] = self.M
            self.current_jj = jj
            
            # getting the tasks
            self.m_pub.publish(Int32(data=self.M))

            rospy.sleep(1)
            
            # getting the robots location and battery level
            robots_info = self.get_robot_info()

            self.timings.clear()
            
            MAX_ITERATIONS, SWARM_SIZE, \
                    robot_charge_duration, robots_coord, \
                    Charging_station, CHARGING_TIME, Energy_Harvesting_Rate,\
                    = self.init_enivornment(robots_info)
            universes, tmvo = self.explore(MAX_ITERATIONS, SWARM_SIZE, 
                                    robot_charge_duration, robots_coord, self.cached_tasks, 
                                    Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
            rospy.loginfo(f"the time of tmvo = {tmvo}")
            for m in range(8):
                self.current_m = m
                #rospy.loginfo("go to sleep...")
                rospy.sleep(2)
                for ii in range(10):

                    self.current_ii = ii
                    #rospy.loginfo(f"we are at test {jj}, m is {m}")

                    # mvo + ga centralised
                    if m == 0:
                        best_CMVOGA, _ , tmvoga = self.exploitation(MAX_ITERATIONS, len(universes), robot_charge_duration, robots_coord, self.cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "ga", universes)
     

                        if ii!=0:
                            self.bestCMVOGA[jj]=max(self.bestCMVOGAmax[jj] , best_CMVOGA)
                            self.bestCMVOGAmin[jj]=min(self.bestCMVOGAmin[jj] , best_CMVOGA)
                             
                            # tPSOEA[jj] = (tPSOEA[jj] * ii + t_psoea) / (ii + 1)
                            
                            self.tCMVOGA[jj]=(tmvoga + self.tCMVOGA[jj] * ii) / (ii + 1)
                            self.bestCMVOGA[jj]=(best_CMVOGA+self.bestCMVOGA[jj]* ii) / (ii + 1)

                        else:
                            self.tCMVOGA[jj]=tmvoga
                            self.bestCMVOGA[jj]=best_CMVOGA
                            self.bestCMVOGAmax[jj]=best_CMVOGA
                            self.bestCMVOGAmin[jj]=best_CMVOGA
                    # ga centralised
                    if m == 1:
                        best_CGA,  _ , tga = self.exploitation(MAX_ITERATIONS, SWARM_SIZE, robot_charge_duration, robots_coord, self.cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "ga")


                        if ii!=0:
                            self.bestCGA[jj]=max(self.bestCGAmax[jj] , best_CGA)
                            self.bestCGAmin[jj]=min(self.bestCGAmin[jj] , best_CGA)
                            self.tCGA[jj]=(tga+self.tCGA[jj] * ii) / (ii + 1)
                            self.bestCGA[jj]=(best_CGA+self.bestCGA[jj]* ii) / (ii + 1)

                        else:
                            self.tCGA[jj]=tga
                            self.bestCGA[jj]=best_CGA
                            self.bestCGAmax[jj]=best_CGA
                            self.bestCGAmin[jj]=best_CGA

                    # mvo + ga distributed  
                    if m == 2:
                        array_allocation = self.allocator.allocate_exploration(self.coordinator_coords, robots_coord, SWARM_SIZE, self.M)

                        self.send_parameters(robots_info, self.cached_tasks, universes, array_allocation, num_cols=self.M)
                    
                    # ga distributed
                    if m == 3:
                        array_allocation = self.allocator.allocate_exploration(self.coordinator_coords, robots_coord, SWARM_SIZE, self.M)

                        self.send_parameters(robots_info, self.cached_tasks, universes, array_allocation, num_cols=self.M)
                        



                    # mvo + pso centralised
                    if m == 4:
                        best_CMVOPSO, tmvopso , _= self.exploitation(MAX_ITERATIONS, len(universes), robot_charge_duration, robots_coord, self.cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "pso", universes)


                        if ii!=0:
                            self.bestCMVOPSO[jj]=max(self.bestCMVOPSOmax[jj] , best_CMVOPSO)
                            self.bestCMVOPSOmin[jj]=min(self.bestCMVOPSOmin[jj] , best_CMVOPSO)
                            self.tCMVOPSO[jj]=(tmvopso+self.tCMVOPSO[jj]* ii) / (ii + 1)
                            self.bestCMVOPSO[jj]=(best_CMVOPSO+self.bestCMVOPSO[jj]* ii) / (ii + 1)

                        else:
                            self.tCMVOPSO[jj]=tmvopso
                            self.bestCMVOPSO[jj]=best_CMVOPSO
                            self.bestCMVOPSOmax[jj]=best_CMVOPSO
                            self.bestCMVOPSOmin[jj]=best_CMVOPSO
                    # pso centralised
                    if m == 5:
                        best_CPSO, tpso ,_ = self.exploitation(MAX_ITERATIONS, SWARM_SIZE, robot_charge_duration, robots_coord, self.cached_tasks, 
                                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, "pso")

                        if ii!=0:
                            self.bestCPSO[jj]=max(self.bestCPSOmax[jj] , best_CPSO)
                            self.bestCPSOmin[jj]=min(self.bestCPSOmin[jj] , best_CPSO)
                            self.tCPSO[jj]=(tpso+self.tCPSO[jj]* ii) / (ii + 1)
                            self.bestCPSO[jj]=(best_CPSO+self.bestCPSO[jj]* ii) / (ii + 1)

                        else:
                            self.tCPSO[jj]=tpso
                            self.bestCPSO[jj]=best_CPSO
                            self.bestCPSOmax[jj]=best_CPSO
                            self.bestCPSOmin[jj]=best_CPSO

                    # mvo + pso distributed  
                    if m == 6:
                        array_allocation = self.allocator.allocate_exploration(self.coordinator_coords, robots_coord, SWARM_SIZE, len(self.cached_tasks))

                        self.send_parameters(robots_info, self.cached_tasks, universes, array_allocation, num_cols=self.M)
                    # pso distributed
                    if m == 7:
                        array_allocation = self.allocator.allocate_exploration(self.coordinator_coords, robots_coord, SWARM_SIZE, len(self.cached_tasks))

                        self.send_parameters(robots_info, self.cached_tasks, universes, array_allocation, num_cols=self.M)

                        
                    #CSV_FILE = 'data/coord_timing_central.csv'
                    #log_coordinator_timing(pso_time=pso_c, CSV_FILE=CSV_FILE)
            self.M += 5
        self.plot()
        rospy.spin()


    def send_parameters(self, robots_info, tasks, universes, array_allocation, num_cols):
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
            msg.m = self.current_m
            msg.j = self.current_jj
            msg.i = self.current_ii

            pub = publishers[robot.name]

            # Wait for at least one subscriber
            while pub.get_num_connections() < 1:
                rospy.logwarn(f"Waiting for subscriber on /{robot.name}/parameters...")
                rospy.sleep(2)


            
            msg.header = Header(stamp=rospy.Time.now())
            pub.publish(msg)

    
    def init_enivornment(self, robots_info: List[RobotStatusInfo]):
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
        SWARM_SIZE = 100
        POP_SIZE = 10
        MAX_ITERATIONS = 50




        N = len(robots_info)  # number of robots
        M = len(self.cached_tasks)




        # Instead of random charge duration, calculate from battery percentage
        # Assuming battery is in percentage (0 to 100), scale it to seconds of charge (0 to 20*3600)
        robot_charge_duration = [(r.battery / 100.0) * 20 * 3600 for r in robots_info]
        robots_coord = coords


        

        return MAX_ITERATIONS, SWARM_SIZE, robot_charge_duration, robots_coord, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate
    

    def explore(self, MAX_ITERATIONS, swarm_size, robot_charge_duration, robots_coord, task, 
                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate):
        
        start_exp = time.time()
        init_swarm = mvo_exploration(
                MAX_ITERATIONS=MAX_ITERATIONS,
                SWARM_SIZE=swarm_size,
                M=self.M, N=self.N, 
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


    def exploitation(self,MAX_ITERATIONS, swarm_size, robot_charge_duration, robots_coord, task, 
                Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, method, init_swarm=[]):
        
        tpso1 = rospy.Time.now()
        if method == "pso":
            best, _ = PSO_Algorithm(
                    MAX_ITERATIONS=MAX_ITERATIONS,
                    SWARM_SIZE=swarm_size,
                    M=self.M, N=self.N, 
                    iterationstop=15,
                    robot_charge_duration=robot_charge_duration,
                    robots_coord=robots_coord,
                    task=task,
                    Charging_station=Charging_station,
                    CHARGING_TIME=CHARGING_TIME,
                    Energy_Harvesting_Rate=Energy_Harvesting_Rate,
                    init_swarm=init_swarm
                )
        tpso2 = rospy.Time.now()
        tpso = tpso2 - tpso1    
        
        ga1 = rospy.Time.now()
        if method == "ga":
            best, _ = genetic_algorithm(
                    POP_SIZE=swarm_size,
                    M=self.M, N=self.N,iteration=MAX_ITERATIONS,
                    iterationstop=15,
                    robot_charge_duration=robot_charge_duration,
                    robots_coord=robots_coord,
                    task=task,
                    Charging_station=Charging_station,
                    CHARGING_TIME=CHARGING_TIME,
                    Energy_Harvesting_Rate=Energy_Harvesting_Rate,
                    init_swarm=init_swarm
            )
        ga2 = rospy.Time.now()
        ga = ga2 - ga1

        #rospy.loginfo(f"m {self.current_m} {self.current_jj} this is pso solve time {tpso.to_sec()}")
        return best , tpso.to_sec(), ga.to_sec()
        

    def plot(self):
        directory = "data/plots"
        os.makedirs(directory, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.MM, self.tCMVOGA, marker="o", linestyle="-", color="red", label="tCMVOGA")
        plt.plot(self.MM, self.tCGA, marker="*", linestyle="-", color="green", label="tCGA")

        #plt.plot(self.MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
        #plt.plot(self.MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

        plt.plot(self.MM, self.tDMVOGA, marker="o", linestyle="-", color="blue", label="mtDMVOGA")
        plt.plot(self.MM, self.tDGA, marker="*", linestyle="-", color="olive", label="tDGA")

        plt.xlabel("M")
        plt.ylabel("Values")
        plt.title("tGA against M")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'tGA'))


        plt.figure(figsize=(10, 6))
        plt.plot(self.MM, self.bestCMVOGA, marker="o", linestyle="-", color="red", label="best_CMVOGA")
        plt.plot(self.MM, self.bestCGA, marker="*", linestyle="-", color="green", label="best_CGA")

        #plt.plot(self.MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
        #plt.plot(self.MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

        plt.plot(self.MM, self.bestDMVOGA, marker="o", linestyle="-", color="blue", label="best_DMVOGA")
        plt.plot(self.MM, self.bestDGA, marker="*", linestyle="-", color="olive", label="best_DGA")

        plt.xlabel("M")
        plt.ylabel("Values")
        plt.title("bestGA against M")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'bestGA'))

        plt.figure(figsize=(10, 6))
        plt.plot(self.MM, self.tCMVOPSO, marker="o", linestyle="-", color="red", label="tCMVOPSO")
        plt.plot(self.MM, self.tCPSO, marker="*", linestyle="-", color="green", label="tCPSO")

        #plt.plot(self.MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
        #plt.plot(self.MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

        plt.plot(self.MM, self.tDMVOPSO, marker="o", linestyle="-", color="blue", label="mtDMVOPSO")
        plt.plot(self.MM, self.tDPSO, marker="*", linestyle="-", color="olive", label="tDPSO")

        plt.xlabel("M")
        plt.ylabel("Values")
        plt.title("tPSO against M")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'tPSO'))


        plt.figure(figsize=(10, 6))
        plt.plot(self.MM, self.bestCMVOPSO, marker="o", linestyle="-", color="red", label="best_CMVOPSO")
        plt.plot(self.MM, self.bestCPSO, marker="*", linestyle="-", color="green", label="best_CPSO")

        #plt.plot(self.MM, tMVOEA, marker="*", linestyle="-", color="black", label="bestMVOEA")
        #plt.plot(self.MM, tMVOEU, marker="o", linestyle="-", color="black", label="bestMVOEU")

        plt.plot(self.MM, self.bestDMVOPSO, marker="o", linestyle="-", color="blue", label="best_DMVOPSO")
        plt.plot(self.MM, self.bestDPSO, marker="*", linestyle="-", color="olive", label="best_DPSO")

        plt.xlabel("M")
        plt.ylabel("Values")
        plt.title("bestPSO against M")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'bestPSO'))
        


class ExplorationAllocator:
    def __init__(self, bytes_per_int, bits_per_byte, e_elec, e_amp, path_loss_exponent):
        self.BYTES_PER_INT = bytes_per_int
        self.BITS_PER_BYTE = bits_per_byte
        self.E_ELEC = e_elec
        self.E_AMP = e_amp
        self.PATH_LOSS_EXPONENT = path_loss_exponent

    def _compute_message_size(self, num_candidates, num_arrays):
        return num_candidates * num_arrays * self.BYTES_PER_INT + 2 * self.BYTES_PER_INT

    def _compute_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _energy_to_transmit(self, bits, distance):
        return bits * (self.E_ELEC + self.E_AMP * distance ** self.PATH_LOSS_EXPONENT)

    def allocate_exploration(self, central_position, robot_positions, num_candidates, total_arrays):
        distances = [self._compute_distance(central_position, rp) for rp in robot_positions]
        bits_per_array = self._compute_message_size(num_candidates, total_arrays) * self.BITS_PER_BYTE
        energy_costs = [self._energy_to_transmit(bits_per_array, d) for d in distances]

        inv_costs = [1 / e for e in energy_costs]
        total_inv = sum(inv_costs)
        weights = [ic / total_inv for ic in inv_costs]

        exact = [num_candidates * w for w in weights]
        base = [int(e) for e in exact]
        remainder = num_candidates - sum(base)

        fracs = [e - b for e, b in zip(exact, base)]
        for i in sorted(range(len(fracs)), key=lambda i: fracs[i], reverse=True)[:remainder]:
            base[i] += 1

        return base




if __name__ == '__main__':
    node = CoordinatorNode()
    node.run()