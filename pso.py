from fitness_ea import fitness
import random
import copy
import numpy as np


def initialize_swarm(swarm_size: int, M, N) -> list:
    # Initialize particle swarm
    swarm = []
    for i in range(swarm_size):
        particle = [random.randint(0, N - 1) for _ in range(M)]
        swarm.append(particle)
    return swarm


def initialize_swarm(swarm_size, M, N):
    # Initialize particle swarm with random positions
    swarm = [np.random.randint(0, N, size=M) for _ in range(swarm_size)]
    velocities = [np.random.rand(M) * (N / 10) for _ in range(swarm_size)]  # Random initial velocities
    return swarm, velocities

def PSO_Algorithm(MAX_ITERATIONS, SWARM_SIZE, M, N, iterationstop, robot_charge_duration, robots_coord, task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate, init_swarm=[]):
    C1 = 2.0
    C2 = 2.0
    W = 0.9

    # Initialize swarm and velocities
    if len(init_swarm) == 0:
        swarm, velocities = initialize_swarm(SWARM_SIZE, M, N)
    else:
        swarm = copy.deepcopy(init_swarm)
        velocities = np.zeros((SWARM_SIZE, M))

    # Initialize best particle
    best_particle = swarm[0]
    best_fitness, _, _, _ = fitness(best_particle, robot_charge_duration, robots_coord, task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)

    fitnesses = []
    for iteration in range(MAX_ITERATIONS):
        for particle_idx, particle in enumerate(swarm):
            # Evaluate fitness
            particle_fitness, _, _, _ = fitness(particle, robot_charge_duration, robots_coord, task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)

            # Update best particle if necessary
            if particle_fitness < best_fitness:
                best_particle = copy.deepcopy(particle)
                best_fitness = particle_fitness

            # Update velocity and position
            r1 = np.random.rand(M)
            r2 = np.random.rand(M)
            
            cognitive_component = C1 * r1 * (best_particle - particle)
            social_component = C2 * r2 * (best_particle - particle)
            velocities[particle_idx] = W * velocities[particle_idx] + cognitive_component + social_component

            # Update position
            particle = particle.astype(float)  # Temporarily convert to float
            particle += velocities[particle_idx]
            particle = np.clip(particle, 0, N - 1).astype(int)  # Convert back to int
            swarm[particle_idx] = particle

        # Record fitness
        fitnesses.append(best_fitness)
        if len(fitnesses) >= iterationstop:
            last_five = fitnesses[-iterationstop:]
            if all(x == last_five[0] for x in last_five):
                break

    print(f"Number of iterations in PSO: {iteration + 1}")
    return best_fitness, swarm