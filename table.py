import os

import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
lookup_path = os.path.join(current_dir, 'lookup_table.pkl')

with open(lookup_path, 'rb') as f:
    loaded_S = pickle.load(f)
    #print (loaded_S)

# Create a lookup table dictionary
lookup_table = {}

# Populate the lookup table with data from loaded S
for row in loaded_S:
    coords_i = row[0]
    coords_j = row[1]
    estimated_travel_time = row[2]
    estimated_energy_cost = row[3]

    lookup_table[(coords_i, coords_j)] = (estimated_travel_time, estimated_energy_cost)


def Energy_distance_calculation(start_p,end_p):
        #Estimating_ Energy_Consumption and Duration Based on Distance, Weight and Speed.
        plan_instance = make_plan_client.PathEnergyEstimation()
        start_point = (start_p['x'],start_p['x'],3.14 / 3)
        #[-0.2, 0.57, 2 * 3.14 / 3]
        #start_point.append(2 * 3.14 / 3)
        end_point = (end_p['x'],end_p['x'],3.14 / 3)
        #end_point.append(2 * 3.14 / 3)
        #[-0.2, -0.57, 2 * 3.14 / 6]
        start, goal = plan_instance.make_start_goal_message(start_point, end_point)
        path_plan = plan_instance.get_plan(start, goal)
        max_attempts = 5
        current_attempt = 1
        plan_waypoints = None

        while current_attempt <= max_attempts:
            try:
                plan_waypoints = path_plan.plan.poses
                break  # If the operation is successful, exit the loop
            except Exception as e:
                print(f"An error occurred while retrieving plan waypoints (attempt {current_attempt}):", e)
                current_attempt += 1

            if current_attempt > max_attempts:
                 print("Reached maximum number of attempts to retrieve plan waypoints.")
        plan_waypoints = path_plan.plan.poses
        estimated_travel_time, estimated_energy_cost = plan_instance.get_path_energy_time_prediction(plan_waypoints, start_point)

        return estimated_travel_time, estimated_energy_cost


def lookuptable(input_coords_i, input_coords_j):
    input_coords_i = tuple(round(num, 1) for num in input_coords_i)
    input_coords_j = tuple(round(num, 1) for num in input_coords_j)

    if (input_coords_i, input_coords_j) in lookup_table:

        estimated_travel_time_i, estimated_energy_cost_i = lookup_table[(input_coords_i, input_coords_j)]

    else:
        a = {}
        b = {}
        a['x'] = input_coords_i[0]
        a['y'] = input_coords_i[1]
        b['x'] = input_coords_j[0]
        b['y'] = input_coords_j[1]
        estimated_travel_time_i, estimated_energy_cost_i = Energy_distance_calculation(a, b)
    return estimated_travel_time_i, estimated_energy_cost_i
