import numpy as np
import copy
from table import lookuptable


def fitness(particle, robot_charge_duration, robots_coord, tasks, Charging_station, CHARGING_TIME,
            Energy_Harvesting_Rate):
    
    num_jobs=4
    robot_charge = copy.deepcopy(robot_charge_duration)
    robot_location = copy.deepcopy(robots_coord)
    robot_time = np.zeros(len(robot_charge))
    N = len(robot_charge)
    M = len(tasks)
    timing_info = []
    task_order = []
    charger_usage = [0] * N
    startingtime = [0] * M
    endingtime = [0] * M
    Parallel_duration = [0] * M
    Battery_depletion = 0
    LARGE_NUMBER = 1e6

    # Divide tasks into jobs randomly
    job_indices = np.array_split(np.random.permutation(M), num_jobs)

    job_end_times = np.zeros(num_jobs)  # Store the end times of each job

    for job_index in range(num_jobs):
        tasks_in_job = job_indices[job_index]

        for task_index in tasks_in_job:
            task = tasks[task_index]
            robot = particle[task_index]

            # Calculate the duration of subtasks
            T_pick, E_pick = lookuptable(robot_location[robot], task["drop_initial_coordination"])
            T_drive, E_drive = lookuptable(task["drop_initial_coordination"], task["drop_target_coordination"])
            T_drop, E_drop = lookuptable(task["drop_target_coordination"], Charging_station)

            total_duration = T_pick + T_drive + T_drop
            total_energy = E_pick + E_drive + E_drop

            # Check if the robot needs to recharge
            if robot_charge[robot] < 1.15 * total_energy + total_duration:
                if robot_charge[robot] < 0:
                    Battery_depletion += 1
                # Robot needs to recharge
                charger_usage[robot] += 1
                startingtime[task_index] = max(robot_time[robot], job_end_times[job_index])
                robot_time[robot] += CHARGING_TIME + 2 * T_drop - (robot_charge[robot] / (20 * 3600)) * CHARGING_TIME
                endingtime[task_index] = robot_time[robot]
                robot_charge[robot] = 20 * 3600 - total_energy + Energy_Harvesting_Rate * T_drop
                robot_location[robot] = task["drop_initial_coordination"]
                timing_info.append([robot, startingtime[task_index], endingtime[task_index], task_index,
                                    robot_charge[robot] * 100 / (20 * 3600), 'Charging_task'])

                startingtime[task_index] = robot_time[robot]
                robot_time[robot] += total_duration
                endingtime[task_index] = robot_time[robot]
                robot_location[robot] = task["drop_target_coordination"]
                robot_charge[robot] = robot_charge[robot] - total_energy + Energy_Harvesting_Rate * total_duration
                timing_info.append([robot, startingtime[task_index], endingtime[task_index], task_index,
                                    robot_charge[robot] * 100 / (20 * 3600), 'Parallel_task'])
            else:
                task_order.append(task["order"])
                robot_location[robot] = task["drop_target_coordination"]
                robot_charge[robot] = robot_charge[robot] - total_energy + Energy_Harvesting_Rate * total_duration

                startingtime[task_index] = max(robot_time[robot], job_end_times[job_index])
                robot_time[robot] += total_duration
                endingtime[task_index] = robot_time[robot]

                timing_info.append([robot, startingtime[task_index], endingtime[task_index], task_index,
                                    robot_charge[robot] * 100 / (20 * 3600), 'Parallel_task'])

        job_end_times[job_index] = max([endingtime[task_index] for task_index in tasks_in_job])

    # Check if the job priority criteria is met
    for job_index in range(1, num_jobs):
        if job_end_times[job_index] < job_end_times[job_index - 1]:
            # If any job is completed before the previous job, assign a large number to total time
            return LARGE_NUMBER, Battery_depletion, robot_location, timing_info

    total_time = max(job_end_times)

    return total_time, Battery_depletion, robot_location, timing_info