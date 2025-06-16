#!/usr/bin/env python3
import rospy
from robot_msgs.msg import DropTaskArray, DropTask
from geometry_msgs.msg import Point

import random
import numpy as np

from std_msgs.msg import Int32

coords=np.array([(1.8,3),(1.8,4),(1.8,5),(1.8,6),(1.8,7),(1.8,8),(-0.5,-8),(-0.5,-9),(-0.5,-7),(-0.5,1.8),(-0.5,3),(-0.5,4),(-0.5,5),(-0.5,6),(3.5,1.5),(5.5,1.5),(3.5,2.5),(5.5,2.5),(3.5,-2.5),(5.5,-2.5),(3.5,-4),(5.5,-4),(3.5,-5.6),(5.5,-5.6),(3.5,-10),(5.5,-10),(3.5,-7.8),(5.5,-7.8),(-4.5,-9),(-4.5,-9),(-4.5,-8),(-4.5,-7),(-4.5,-6),(-4.5,-5),(-4.5,-4),(-4.5,-3),(-4.5,-2),(-4.5,-1),(-4.5,0),(-4.5,1),(-4.5,2),(-4.5,3),(-4.5,4),(-4.5,5),(-4.5,6),(-4.5,7),(-4.5,8),(-4.5,9),(-3,1),(-3,1),(-3,2),(-3,3),(-3,4),(-3,5),(-3,6),(-3,7),(-3,8),(-3,9)])


current_M = 50  # default


def m_callback(msg):
        
    task_pub = rospy.Publisher('/robot_tasks', DropTaskArray, queue_size=10)
    
    while task_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.05)

    M = msg.data
    rospy.loginfo(f"[task_node] Received new M: {M} — generating tasks.")
    tasks = get_task(M, coords)
    task_pub.publish(tasks)
    rospy.loginfo(f"[task_node] Published {M} tasks.")

def get_task(M, coords):
    task_array_msg = DropTaskArray()
    for _ in range(M):

        drop_initial = random.choice(coords)
        order = "parallel"
        drop_target = random.choice(coords)
         
        task_msg = DropTask()
        task_msg.drop_initial_coordination = Point(x=drop_initial[0], y=drop_initial[1], z=0.0)
        task_msg.drop_target_coordination = Point(x=drop_target[0], y=drop_target[1], z=0.0)
        task_msg.order = order
        task_array_msg.tasks.append(task_msg)
    

    return task_array_msg
    



def run_task_publisher():
    rospy.init_node("task_node")
    rospy.loginfo("running task")
    rospy.Subscriber('/task_count', Int32, m_callback)



    
    rospy.spin()

if __name__ == '__main__':
    run_task_publisher()




