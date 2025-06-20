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
        # 1) Read env‑vars into self
        self.name     = os.environ.get("ROBOT_NAME", "robot")
        self.x        = float(os.environ.get("ROBOT_X", "0.0"))
        self.y        = float(os.environ.get("ROBOT_Y", "0.0"))
        self.battery  = float(os.environ.get("BATTERY", "100.0"))
        self.Ntest    = 10
        self.communication_time = np.zeros(self.Ntest)
        self.communication_time_all = [[] for _ in range(self.Ntest)] 
        self.m        = 0 
        # plus any cached_* lists (initialize to None or empty)
        self.cached_tasks = []
        self.cached_universe = np.empty((0,0))
        self.cached_robot_statuses = []
        # 2) ROS init
        rospy.init_node(self.name)

        # 3) Advertise service
        srv_name = f"/get_robot_status_{self.name}"
        rospy.Service(srv_name, RobotStatus, self.handle_status_request)
        #rospy.loginfo(f"[{self.name}] Ready to respond on service {srv_name}")

        # 4) Publisher for fitness
        self.fitness_pub = rospy.Publisher('/fitness_value', FitnessValue, queue_size=10)

        # 5) Subscribe to parameters
        topic = f"/{self.name}/parameters"
        rospy.Subscriber(topic, Parameters, self.parameters_callback)
        #rospy.loginfo(f"[{self.name}] Subscribed to {topic}")

    # 6) Move your free functions in as methods:
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

    # 7) The big parameters handler becomes:
    def parameters_callback(self, msg):
        send_time = msg.header.stamp
        recv_time = rospy.Time.now()


        communicate = (recv_time - send_time).to_sec()

        # cached the tasks in cached_tasks global variable
        self.cach_task(msg)
        self.cach_robot_status(msg)
        self.cach_universe(msg)
        self.m = msg.m
        self.i = msg.i
        self.j = msg.j
        self.communication_time[self.j]=(communicate+self.communication_time[self.j]* self.i) / (self.i + 1)
        self.communication_time_all[self.j].append(communicate)
        if self.i == 4:
            std = np.std(self.communication_time_all[self.j])
            df = pd.DataFrame({'Std_Comm_Time': [std]})
            write_df_as_text(df,f"std_{self.name}_{self.j}" , "std_communication" )
            write_to_file(self.communication_time, "communication_time/" ,f"comm_time_{self.name}_{self.j}_{self.m}")
        #rospy.loginfo(f"got parameters for {self.name} will start explotation with m = {self.m}")
        best, start_time = self.run_explotation()
        #rospy.loginfo(f"for m {self.m}, j {self.j} and fitness {best}")

        val = FitnessValue(fitness=best)

        
        while self.fitness_pub.get_num_connections() == 0:
            rospy.logwarn(f"Waiting for subscriber on /{self.name}/fitness.")
            rospy.sleep(0.05)

        val.communication = rospy.Time.now()
        val.start_time = start_time
        val.m = self.m
        val.j = self.j
        val.i = self.i
        self.fitness_pub.publish(val)
        
        #rospy.loginfo(f"[{robot_name}] /fitness_value publisher connections: {fitness_pub.get_num_connections()}")
    
        #rospy.signal_shutdown("fitness value sent. Shutting down")

    # 8) Your service handler as a method:
    def handle_status_request(self, req):
        #rospy.loginfo(f"[{robot_name}] Status requested.")
        res = RobotStatusResponse()
        res.robot_name = self.name
        res.position = Point(x=self.x, y=self.y, z=0.0)
        res.battery = self.battery
        return res

    # 9) Exploitation runner can stay a free function or become:
    def run_explotation(self):
        Energy_Harvesting_Rate = 3
        MAX_CHARGE = 20
        CHARGING_RATE = 10 / 3600  # w/s
        DECHARGING_RATE = 5 / 3600  # w/s
        DECHARGING_Time = 4 * 3600  # s
        CHARGING_TIME = 0.5 * 3600  # s
        iterationstop = 15
        Charging_station = (-1.6, 7.2)
        POP_SIZE = 100
        MAX_ITERATIONS = 50

        N = len(self.cached_robot_statuses)  # number of robots
        M = len(self.cached_tasks)  # number of tasks
        

        if self.m in {0,2,4,6}:
            POP_SIZE = len(self.cached_universe)
        else:
            POP_SIZE = int(POP_SIZE/5)
            self.cached_universe = np.empty((0,))



        robot_charge_duration = []
        robots_coord = []
        for r in self.cached_robot_statuses:
            bat = (r.battery / 100.0) * 20 * 3600
            robot_charge_duration.append(bat)
            point = [r.position.x, r.position.y]
            robots_coord.append(point)

        robots_coord = np.array(robots_coord)
        
        if self.m <= 3:
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

    # 10) Finally a run() to spin:
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = RobotNode()
    node.run()






