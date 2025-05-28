from fitness_eu import fitnessEU
from fitness_ea import fitness
import random
import copy



def initialize_swarm(swarm_size: int, M, N) -> list:
    # Initialize particle swarm
    swarm = []
    for i in range(swarm_size):
        particle = [random.randint(0, N - 1) for _ in range(M)]
        swarm.append(particle)
    return swarm


def PSO_Algorithm(MAX_ITERATIONS, SWARM_SIZE, M, N, s, iterationstop,
                   robot_charge_duration, robots_coord, task,
                   Charging_station, CHARGING_TIME, Energy_Harvesting_Rate,
                   init_swarm=[]):
    C1 = 2
    C2 = 2
    W = 0.9    
    if len(init_swarm) == 0 :

        swarm = initialize_swarm(SWARM_SIZE,M,N)
    else:
        swarm = init_swarm
    # Initialize best particle
    best_particle = swarm[0]
    if s == 1:
        best_fitness, _, _, _ = fitness(best_particle,robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate)
    else:
        best_fitness, _, _, _ = fitnessEU(best_particle,robot_charge_duration,robots_coord, task, Charging_station,CHARGING_TIME)

    # PSO algorithm
    fitnesses = []
    for iteration in range(MAX_ITERATIONS):
        # print(iteration)
        # C1 = random.uniform(0,.1)
        # C2 = random.uniform(0,.1)
        # W =  random.uniform(0,.1)
        for particle in swarm:
            # Evaluate fitness
            if s == 1:
                particle_fitness, _, _, _ = fitness(particle,robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate)
            else:
                particle_fitness, _, _, _ = fitnessEU(particle,robot_charge_duration,robots_coord, task, Charging_station,CHARGING_TIME)
            # print('particle_fitness',particle_fitness)
            # print('best_fitness',particle_fitness)
            # print('particle',particle)
            # print('best_particle',best_particle)
            # Update best particle if necessary
            if particle_fitness < best_fitness:
                best_particle = copy.deepcopy(particle)
                best_fitness = copy.deepcopy(particle_fitness)

            # Update velocity and position
            for i in range(M):
                r1 = random.gauss(.01, .02)
                r1 = max(0, min((r1 - 0.01 + 0.02) / (0.01 + 0.02 + 0.01), 1))
                r2 = random.gauss(0.01, 0.02)
                r2 = max(0, min((r2 - 0.01 + 0.02) / (0.01 + 0.02 + 0.01), 1))

                # r1=1
                # r2=1
                # print('particle[i]', particle[i])
                velocity = (particle[i] + W * particle[i] + C1 * r1 * (best_particle[i] - particle[i]) + C2 * r2 * (
                            swarm[iteration % SWARM_SIZE][i] - particle[i]))
                a = 0

                if velocity < N / 7:
                    a = 1
                if N / 7 < velocity < (N) / 6:
                    a = (N / 6)
                if N / 6 < velocity < (N) / 5:
                    a = (N / 5)
                if N / 5 < velocity < (N) / 4:
                    a = (N / 4)
                if N / 4 < velocity < (N) / 3:
                    a = N / 3
                if N / 3 < velocity < (N) / 2:
                    a = N / 2
                if N / 2 < velocity < (N) / 1:
                    a = N - 1 / 1
                if -1 < velocity < 0:
                    a = -(N / 6)
                if -(N) / 5 < velocity < -(N) / 6:
                    a = -(N / 5)
                if -(N) / 4 < velocity < -(N) / 5:
                    a = -(N / 4)
                if -(N) / 3 < velocity < -(N) / 4:
                    a = -(N / 3)
                if -(N) / 2 < velocity < -(N) / 3:
                    a = -N / 2
                if -(N) / 1 < velocity < -(N) / 2:
                    a = -N / 1

                a = a + particle[i]
                # particle[i] = max(0, min(N-1, particle[i]))
                particle[i] = round(max(0, min(N - 1, a)))
                # particle[i]=random.randint(0,N)
                # print('particle[a]', particle[i])

        # Update swarm
        swarm[iteration % SWARM_SIZE] = particle

        # Record fitness
        fitnesses.append(best_fitness)
        if len(fitnesses) >= iterationstop:
            # Get the last five elements from the fitnesses array
            last_five = fitnesses[-iterationstop:]

            # Check if all the last five elements are equal
            if all(x == last_five[0] for x in last_five):
                break

        # print('fitnesses',fitnesses)
        # print('best_particle', best_particle)
        # print('particle',particle)

    return best_fitness, fitnesses