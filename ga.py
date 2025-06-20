
from fitness_ea import fitness
import numpy as np
from utilities import write_np_to_file, write_to_file
import copy


# choose a fixed seed
SEED = 42


np.random.seed(SEED)


def initialize_population(pop_size, array_size, value_range):
    return np.random.randint(0, value_range, size=(pop_size, array_size))


# Function to select parents for crossover using tournament selection
def select_parents(population, fitness):
    # Tournament selection: randomly select two individuals and choose the one with higher fitness
    indices = np.random.choice(len(population), size=2)
    if fitness[indices[0]] > fitness[indices[1]]:
        return population[indices[0]]
    else:
        return population[indices[1]]

# Function to perform crossover (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Function 
def mutate(individual,N): 
    MUTATION_RATE = 0.1
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = np.random.randint(0, N)  # Mutate gene to a new random value within the range
    return individual


def genetic_algorithm(POP_SIZE,M,N,iteration,iterationstop,robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate, init_swarm=[]):

    if len(init_swarm) == 0:
        population = initialize_population(POP_SIZE, M, N)
    else:
        population = copy.deepcopy(init_swarm)
    #write_np_to_file(population, 'ga_population', 'population_output') 
    #print(population[0])
    fitnesses=[]
    best_fitness = float('inf')
    for i in range (iteration):
        # Select parents, perform crossover and mutation to create new population
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1 = population[np.random.randint(0, POP_SIZE)]
            parent2 = population[np.random.randint(0, POP_SIZE)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1,N)
            child2 = mutate(child2,N)
            new_population.extend([child1, child2])

        # If the new population is smaller than POP_SIZE (due to odd number of individuals)
        if len(new_population) < POP_SIZE:
            # Add an additional mutated individual from the existing population
            extra = population[np.random.randint(0, POP_SIZE)]
            new_population.append(mutate(extra.copy(), N))

        # Replace old population with new population
        population = np.array(new_population)

        #write_np_to_file(population, f'population_iter_{i}', 'population_output')

        # Return the best individual found after all generations
        individual_index = np.zeros(POP_SIZE)

        for j in range(POP_SIZE):

            individual_index[j],_,_,_ = fitness(population[j],robot_charge_duration,robots_coord,task,Charging_station,CHARGING_TIME,Energy_Harvesting_Rate)
            

              # Ensure calculate_fitness returns a single value

        best_individual_index = np.argmin(individual_index)
        best_of_iteration= individual_index[best_individual_index]
        if best_fitness> best_of_iteration:
            fitnesses.append(best_of_iteration)
            best_fitness=best_of_iteration
            best_individual= population[best_individual_index]
            #write_np_to_file(population, f'current_population_{i}', 'population_output')
            #write_np_to_file(best_individual, f'best_individual_{i}', 'population_individual')
            #write_np_to_file(best_individual_index, f'best_individual_index_{i}', 'population_index')
            #print('F', fitnesses)
        else:
            fitnesses.append(best_fitness)

        if len(fitnesses) >= iterationstop:
                # Get the last five elements from the fitnesses array
            last_ten = fitnesses[-iterationstop:]

                # Check if all the last five elements are equal
            if all(x == last_ten[0] for x in last_ten):
                break

   # print(f"number of iteration in ga {i}")


    return best_fitness, fitnesses
