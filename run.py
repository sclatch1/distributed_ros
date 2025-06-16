import subprocess
import time

for i in range(10):
    print(f"Run {i+1} - Starting task publisher...")
    # Start task publisher node as a subprocess
    task_proc = subprocess.Popen(['python3', 'task_publisher_script.py'])
    
    # Wait for task publisher to finish (it shuts down after publishing tasks)
    task_proc.wait()
    print(f"Run {i+1} - Task publisher finished.")

    print(f"Run {i+1} - Starting coordinator node...")
    # Start coordinator node as a subprocess
    coord_proc = subprocess.Popen(['python3', 'coordinator_script.py'])
    
    # Wait for coordinator node to finish (it calls rospy.signal_shutdown eventually)
    coord_proc.wait()
    print(f"Run {i+1} - Coordinator node finished.")

    # Small delay between runs, if needed
    time.sleep(1)
