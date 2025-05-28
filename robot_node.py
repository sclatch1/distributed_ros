#!/usr/bin/env python3
import rospy
from robot_pkg.msg import DropTaskArray, DropTask

def task_callback(msg):
    rospy.loginfo("Received task array")
    
    for i, task in enumerate(msg.tasks):
        initial = task.drop_initial_coordination
        target = task.drop_target_coordination
        order = task.order
        
        rospy.loginfo(f"Task {i}:")
        rospy.loginfo(f"  Initial: ({initial.x}, {initial.y})")
        rospy.loginfo(f"  Target: ({target.x}, {target.y})")
        rospy.loginfo(f"  Order: {order}")

def robot_node():
    rospy.init_node('robot_node')
    rospy.Subscriber('/robot_tasks', DropTaskArray, task_callback)
    rospy.loginfo("Robot node is running and listening to /robot_tasks")
    rospy.spin()

if __name__ == '__main__':
    robot_node()


