#!/usr/bin/env python3
import rospy
from robot_msgs.msg import DropTaskArray, DropTask
from geometry_msgs.msg import Point

import random
import numpy as np

from std_msgs.msg import Int32

coords=np.array([(1.8,3),(1.8,4),(1.8,5),(1.8,6),(1.8,7),(1.8,8),(-0.5,-8),(-0.5,-9),(-0.5,-7),(-0.5,1.8),(-0.5,3),(-0.5,4),(-0.5,5),(-0.5,6),(3.5,1.5),(5.5,1.5),(3.5,2.5),(5.5,2.5),(3.5,-2.5),(5.5,-2.5),(3.5,-4),(5.5,-4),(3.5,-5.6),(5.5,-5.6),(3.5,-10),(5.5,-10),(3.5,-7.8),(5.5,-7.8),(-4.5,-9),(-4.5,-9),(-4.5,-8),(-4.5,-7),(-4.5,-6),(-4.5,-5),(-4.5,-4),(-4.5,-3),(-4.5,-2),(-4.5,-1),(-4.5,0),(-4.5,1),(-4.5,2),(-4.5,3),(-4.5,4),(-4.5,5),(-4.5,6),(-4.5,7),(-4.5,8),(-4.5,9),(-3,1),(-3,1),(-3,2),(-3,3),(-3,4),(-3,5),(-3,6),(-3,7),(-3,8),(-3,9)])



class TaskPublisher:
    def __init__(self):
        # 1) Node init
        rospy.init_node("task_node")
        rospy.loginfo("running task")
        
        # 2) State & constants
        self.coords = np.array([(1.8,3),(1.8,4),(1.8,5),(1.8,6),(1.8,7),(1.8,8),(-0.5,-8),(-0.5,-9),(-0.5,-7),(-0.5,1.8),(-0.5,3),(-0.5,4),(-0.5,5),(-0.5,6),(3.5,1.5),(5.5,1.5),(3.5,2.5),(5.5,2.5),(3.5,-2.5),(5.5,-2.5),(3.5,-4),(5.5,-4),(3.5,-5.6),(5.5,-5.6),(3.5,-10),(5.5,-10),(3.5,-7.8),(5.5,-7.8),(-4.5,-9),(-4.5,-9),(-4.5,-8),(-4.5,-7),(-4.5,-6),(-4.5,-5),(-4.5,-4),(-4.5,-3),(-4.5,-2),(-4.5,-1),(-4.5,0),(-4.5,1),(-4.5,2),(-4.5,3),(-4.5,4),(-4.5,5),(-4.5,6),(-4.5,7),(-4.5,8),(-4.5,9),(-3,1),(-3,1),(-3,2),(-3,3),(-3,4),(-3,5),(-3,6),(-3,7),(-3,8),(-3,9)])
        self.M = 50
        
        # 3) Publisher
        self.task_pub = rospy.Publisher('/robot_tasks', DropTaskArray, queue_size=10)
        # wait for subscriber(s) only once at startup
        while self.task_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)
        
        # 4) Subscriber
        rospy.Subscriber('/task_count', Int32, self.m_callback)
    
    def m_callback(self, msg):
        self.M = msg.data
        rospy.loginfo(f"[task_node] Received new M: {self.M} — generating tasks.")
        tasks = self.get_task(self.M)
        self.task_pub.publish(tasks)
        rospy.loginfo(f"[task_node] Published {self.M} tasks.")
    
    def get_task(self, M):
        task_array_msg = DropTaskArray()
        for _ in range(M):
            init = random.choice(self.coords)
            target = random.choice(self.coords)
            task = DropTask()
            task.drop_initial_coordination = Point(x=init[0], y=init[1], z=0.0)
            task.drop_target_coordination  = Point(x=target[0], y=target[1], z=0.0)
            task.order = "parallel"
            task_array_msg.tasks.append(task)
        return task_array_msg
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    tp = TaskPublisher()
    tp.run()
