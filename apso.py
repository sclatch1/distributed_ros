from fitness_ea import fitness
import random
import numpy as np
import math
import copy



def initialize_particle(value_range,num_particles, array_size):
    # Initialize particles as arrays of integers within the specified range
    particles = np.random.randint(0, value_range, size=(num_particles, array_size))
    return particles


def apso_algorithm(max_iterations, num_particles,M,N, iterationstop,robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate, init_swarm=[]):
    value_range=N
    array_size=M
    fitnesses = []


    if len(init_swarm) == 0:
        particles = initialize_particle(value_range,num_particles, array_size)
    else:
        particles = copy.deepcopy(init_swarm)
        num_particles = len(particles)
    
    # Initialize velocities as arrays of zeros
    velocities = np.zeros((num_particles, array_size))
    c1 = np.ones(num_particles)
    c2 = np.ones(num_particles)
    inertia_weight = np.ones(num_particles)
    d_g = 0

    # Initialize best position and best fitness value
    best_position = particles[0].copy()  # Initialize with the first particle
    best_fitness = float('inf')
    # fitness = np.zeros(num_particles)
    mean_distance = np.zeros(num_particles)

    for iteration in range(max_iterations):

        # print ('iteration',iteration)

        for k in range(num_particles):
            for j in range(num_particles):
                if k != j:
                    mean_distance[k] = (1 / (num_particles - 1)) * np.sqrt(np.sum((particles[k] - particles[j]) ** 2))
                if k == 0:
                    d_g = (1 / (num_particles - 1)) * np.sqrt(np.sum((particles[j] - best_position) ** 2))

        # print('dg',d_g)
        d_min = np.min(mean_distance)
        # print('dmin',d_min)
        d_max = np.max(mean_distance)
        # print('dmax',d_max)
        f = abs((d_g - d_min)) / (d_max - d_min)
        # print('f',f)
        inertia_weight = 1 / (1 + 1.5 * np.exp(-2.5 * f))

        for i in range(num_particles):
            # print('i',i)
            # Evaluate fitness for each particle
            fit, _, _, _ = fitness(particles[i],robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate)


            # Update best position and best fitness if applicable
            if fit < best_fitness:
                best_position = particles[i].copy()
                best_fitness = fit
                fitnesses.append(best_fitness)
            else:
                fitnesses.append(best_fitness)

            # print('d',distance_to_best)

            if 0 < f < 0.2:
                c1[i] = c1[i] + 0.2
                c2[i] = c2[i] - 0.2
            elif 0.2 <= f < 0.3:
                c1[i] = c1[i] + 0.05
                c2[i] = c2[i] - 0.05
            elif 0.3 < f < 0.4:
                c1[i] = c1[i] + 0.05
                c2[i] = c2[i] + 0.05
            elif 0.4 < f < 1:
                c1[i] = c1[i] - 0.2
                c2[i] = c2[i] + 0.2

            # print('f','W','c1[i]','c2[i]',f,inertia_weight,c1[i],c2[i])

            # Update particle velocities and positions

            for j in range(array_size):

                r1 = random.gauss(.01, .02)
                r1 = max(0, min((r1 - 0.01 + 0.02) / (0.01 + 0.02 + 0.01), 1))
                r2 = random.gauss(0.01, 0.02)
                r2 = max(0, min((r2 - 0.01 + 0.02) / (0.01 + 0.02 + 0.01), 1))

                # print('preparticles',particles[i])
                # Update velocity
                velocities[i][j] = (
                        inertia_weight * velocities[i][j]
                        + c1[i] * r1 * (best_position[j] - particles[i][j])
                        + c2[i] * r2 * (best_position[j] - particles[i][j])
                )
                # print(velocities[i][j])
                # print(particles)
                a = 0
                zz = N + 1
                # print(velocity)
                if velocities[i][j] < N / 7:
                    a = 1
                if N / 7 < velocities[i][j] < (N) / 6:
                    a = (N / 6)
                if N / 6 < velocities[i][j] < (N) / 5:
                    a = (N / 5)
                if N / 5 < velocities[i][j] < (N) / 4:
                    a = (N / 4)
                if N / 4 < velocities[i][j] < (N) / 3:
                    a = N / 3
                if N / 3 < velocities[i][j] < (N) / 2:
                    a = N / 2
                if N / 2 < velocities[i][j] < (N) / 1:
                    a = N - 1 / 1
                if -1 < velocities[i][j] < 0:
                    a = -(N / 6)
                if -(N) / 5 < velocities[i][j] < -(N) / 6:
                    a = -(N / 5)
                if -(N) / 4 < velocities[i][j] < -(N) / 5:
                    a = -(N / 4)
                if -(N) / 3 < velocities[i][j] < -(N) / 4:
                    a = -(N / 3)
                if -(N) / 2 < velocities[i][j] < -(N) / 3:
                    a = -N / 2
                if -(N) / 1 < velocities[i][j] < -(N) / 2:
                    a = -N / 1

                # Update position with discrete values within the defined range
                # print('particles[i][j]',i , j , particles[i][j])
                new_position = particles[i][j] + a
                # print('new_poition',new_position)
                if not math.isnan(new_position):
                    particles[i][j] = round(max(0, min(value_range - 1, round(new_position, 0))))

                # print('particles[i][j] after modifying',i , j , particles[i][j])

                # print(particles)
        if len(fitnesses) >= iterationstop:
            # Get the last five elements from the fitnesses array
            last_five = fitnesses[-iterationstop:]

            # Check if all the last five elements are equal
            if all(x == last_five[0] for x in last_five):
                break

        if s == 1:
            _, battery_depletion, _, _ = fitness(best_position,robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate)
        else:
            _, battery_depletion, _, _ = fitnessEU(best_position,robot_charge_duration,robots_coord, task, Charging_station,CHARGING_TIME)

    return best_fitness, fitnesses
