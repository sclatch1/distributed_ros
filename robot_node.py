#!/usr/bin/env python3
import rospy
from robot_msgs.msg import DropTaskArray, UniverseArray, RobotStatusArray, FitnessValue, Parameters
from geometry_msgs.msg import Point
from robot_msgs.srv import RobotStatus, RobotStatusResponse




import os
import numpy as np
import pandas as pd

from pso import PSO_Algorithm
from ga import genetic_algorithm
import time

from dataclasses import dataclass
from typing import Tuple

from utilities import write_np_to_file, write_to_file, write_df_as_text




class RobotNode:
    def __init__(self):
        self.name     = os.environ.get("ROBOT_NAME", "robot")
        self.x        = float(os.environ.get("ROBOT_X", "0.0"))
        self.y        = float(os.environ.get("ROBOT_Y", "0.0"))
        self.battery  = float(os.environ.get("BATTERY", "100.0"))
        self.Ntest    = 1
        self.m        = 0
        self.j       = 0 
        self.i  = 0
        self.SWARM_SIZE = 120
        self.N = 30
        # plus any cached_* lists (initialize to None or empty)
        self.cached_tasks = []
        self.cached_universe = np.empty((0,0))
        self.cached_robot_statuses = []

        rospy.init_node(self.name)

        # Advertise service
        srv_name = f"/get_robot_status_{self.name}"
        rospy.Service(srv_name, RobotStatus, self.handle_status_request)
        #rospy.loginfo(f"[{self.name}] Ready to respond on service {srv_name}")


        self.fitness_pub = rospy.Publisher('/fitness_value', FitnessValue, queue_size=10)


        topic = f"/{self.name}/parameters"
        rospy.Subscriber(topic, Parameters, self.parameters_callback)
        #rospy.loginfo(f"[{self.name}] Subscribed to {topic}")

    def cach_task(self, msg):
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
    
    def cach_universe(self, msg):
        #rospy.loginfo(f"Received universe array with universe len(msg.universe)")
        universes = np.array(msg.universe).reshape((msg.rows, msg.cols))
        self.cached_universe = universes
        return True

    def cach_robot_status(self, msg):
        self.cached_robot_statuses = msg.status

    def parameters_callback(self, msg):
        recv_time = rospy.Time.now()
        send_time = msg.start_communication
        


        communicate = (recv_time - send_time)

 

        # cached the tasks in cached_tasks global variable
        
        self.cach_task(msg)
        self.cach_robot_status(msg)
        self.cach_universe(msg)
        self.m = msg.m
        self.i = msg.i
        self.j = msg.j
        #rospy.loginfo(f"this is m {self.m} {msg.m}")
        #rospy.loginfo(f"for j {self.j}  i {self.i} m {self.m}")

        best, start_time = self.run_explotation()
        #rospy.loginfo(f"for m {self.m}, j {self.j} and fitness {best}")
        

        val = FitnessValue(fitness=best)
        val.communication_time = communicate
        
        
        while self.fitness_pub.get_num_connections() == 0:
            rospy.logwarn(f"Waiting for subscriber on /{self.name}/fitness.")
            rospy.sleep(0.05)

        rospy.loginfo(f"this is m {self.m} {msg.m}")
        val.start_time = start_time
        val.m = self.m
        val.j = self.j
        val.i = self.i
        val.start_com = rospy.Time.now()
        rospy.sleep(0.002) # communication delay

        self.fitness_pub.publish(val)
        
        #rospy.loginfo(f"[{robot_name}] /fitness_value publisher connections: {fitness_pub.get_num_connections()}")
    
        #rospy.signal_shutdown("fitness value sent. Shutting down")



    def handle_status_request(self, req):
        #rospy.loginfo(f"[{self.name}] Status requested.")
        res = RobotStatusResponse()
        res.robot_name = self.name
        res.position = Point(x=self.x, y=self.y, z=0.0)
        res.battery = self.battery

        return res


    def run_explotation(self):
        Energy_Harvesting_Rate = 3
        MAX_CHARGE = 20
        CHARGING_RATE = 10 / 3600  # w/s
        DECHARGING_RATE = 5 / 3600  # w/s
        DECHARGING_Time = 4 * 3600  # s
        CHARGING_TIME = 0.5 * 3600  # s
        iterationstop = 15
        Charging_station = (-1.6, 7.2)
        POP_SIZE = self.SWARM_SIZE
        MAX_ITERATIONS = 50

        N = len(self.cached_robot_statuses)  # number of robots
        M = len(self.cached_tasks)  # number of tasks
        

        if self.m in {0,2,4,6}:
            POP_SIZE = len(self.cached_universe)
        else:
            POP_SIZE = int(POP_SIZE/self.N)
            self.cached_universe = np.empty((0,))



        robot_charge_duration = []
        robots_coord = []
        for r in self.cached_robot_statuses:
            bat = (r.battery / 100.0) * 20 * 3600
            robot_charge_duration.append(bat)
            point = [r.position.x, r.position.y]
            robots_coord.append(point)

        robots_coord = np.array(robots_coord)
        
        if self.m in {4,5}:
            """
            if len(self.cached_universe) > 0:
                write_np_to_file(self.cached_universe, f"GA_{self.m}", f"data/universes/{self.name}")
            """
            start_time = rospy.Time.now()
            best_fitness , _ = genetic_algorithm(POP_SIZE,M,N,MAX_ITERATIONS,iterationstop,robot_charge_duration,robots_coord,self.cached_tasks,Charging_station,CHARGING_TIME, Energy_Harvesting_Rate, self.cached_universe)

        
        else:
            """
            if len(self.cached_universe) > 0:
                write_np_to_file(self.cached_universe, f"PSO_{self.m}", f"data/universes/{self.name}")
            """
            start_time = rospy.Time.now()
            best_fitness , swarm = PSO_Algorithm(MAX_ITERATIONS,POP_SIZE,M,N,iterationstop,robot_charge_duration,robots_coord,self.cached_tasks,Charging_station,CHARGING_TIME, Energy_Harvesting_Rate, self.cached_universe)
            #write_np_to_file(np.array(swarm), f"PSO_{self.m}", f"data/swarm/{self.name}")

        
        return best_fitness, start_time


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = RobotNode()
    node.run()






