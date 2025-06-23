import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utilities import write_np_to_file
from fitness_ea import fitness


# choose a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)






def calculate_diversity(population):
    """Calculate average pairwise Hamming distance"""
    n = len(population)
    if n < 2:
        return 0
    
    total_distance = 0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.sum(population[i] != population[j])
            total_distance += distance
            count += 1
    
    return total_distance / (count * len(population[0]))  # Normalize by dimensions


def select_diverse_source(universes, fitness_vals, current_idx):
    """Select source universe considering both fitness and diversity"""
    n = len(universes)
    
    # Calculate distances from current universe
    distances = np.array([
        np.sum(universes[i] != universes[current_idx]) 
        for i in range(n)
    ])
    
    # Normalize fitness and distances
    norm_fitness = 1 / (fitness_vals + 1e-10)  # Invert for minimization
    norm_fitness = norm_fitness / np.sum(norm_fitness)
    
    norm_distances = distances / (np.sum(distances) + 1e-10)
    
    # Combined score (balance fitness and diversity)
    alpha = 0.7  # Weight for fitness
    scores = alpha * norm_fitness + (1 - alpha) * norm_distances
    
    # Roulette wheel selection based on scores
    cumsum = np.cumsum(scores)
    r = np.random.random() * cumsum[-1]
    return np.searchsorted(cumsum, r)








def track_mvo_diversity(universes):
    """
    Compute diversity metrics for a population of universes:
     - unique_count: number of unique solutions
     - avg_hamming: average pairwise Hamming distance
    """
    unique_count = len(np.unique(universes, axis=0))
    pairs = list(combinations(range(len(universes)), 2))
    if not pairs:
        avg_hamming = 0.0
    else:
        distances = [np.sum(universes[i] != universes[j]) for i, j in pairs]
        avg_hamming = float(np.mean(distances))
    return unique_count, avg_hamming


# In fitness evaluation, penalize individuals too close to others
def fitness_sharing(universes, fitness_vals, sigma=3):
    shared = np.zeros_like(fitness_vals)
    for i, ui in enumerate(universes):
        sh = 0
        for j, uj in enumerate(universes):
            d = np.sum(ui != uj)
            sh += max(0, 1 - (d / sigma)) if d < sigma else 0
        shared[i] = fitness_vals[i] * (1 + sh)
    return shared



def mvo_exploration(
    MAX_ITERATIONS, SWARM_SIZE, M, N, 
    robot_charge_duration, robots_coord, task,
    Charging_station, CHARGING_TIME, Energy_Harvesting_Rate,
    WEP_min=0.05, WEP_max=0.2,  # Lower WEP to reduce convergence
    p=2, stagnation_limit=20,
    grace_period=5, archive_size=10
):
    # Initialize population
    universes = np.random.randint(0, N, size=(SWARM_SIZE, M))

    # Track metrics and history
    history = []
    fitness_history = []

    best_fit_global = float("inf")
    best_u_global = None
    stagnation_counter = np.zeros(SWARM_SIZE, dtype=int)

    lb, ub = 0, N - 1

    # Archive to store diverse best solutions
    archive_universes = []
    archive_fitness = []

    for t in range(1, MAX_ITERATIONS + 1):
        history.append(universes.copy())

        # Evaluate fitness
        fitness_vals = np.array([
            fitness(u, robot_charge_duration, robots_coord, task,
                    Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)[0]
            for u in universes
        ])

        # Update best global
        best_idx = np.argmin(fitness_vals)
        best_fit = float(fitness_vals[best_idx])
        best_u = universes[best_idx].copy()
        
        improvement_mask = fitness_vals < best_fit_global
        stagnation_counter = np.where(improvement_mask, 0, stagnation_counter + 1)

        if best_fit < best_fit_global:
            best_fit_global = best_fit
            best_u_global = best_u

        # Update archive with diverse best solutions
        update_best_archive(archive_universes, archive_fitness, universes, fitness_vals, archive_size)

        # Dynamic parameters - but keep WEP low
        WEP = WEP_min + (t / MAX_ITERATIONS) * (WEP_max - WEP_min)
        TDR = 1 - (t / MAX_ITERATIONS) ** (1 / p)

        # Generate new universes
        new_universes = universes.copy()
        for i, u in enumerate(universes):
            if stagnation_counter[i] >= grace_period:
                # Instead of pure random, create in unexplored region
                if len(archive_universes) > 0:
                    # Move away from all archived solutions
                    new_u = np.random.randint(0, N, size=M)
                    for arch in archive_universes:
                        # Ensure new solution is different from archived ones
                        while np.sum(new_u != arch) < M * 0.3:
                            new_u = np.random.randint(0, N, size=M)
                else:
                    new_u = np.random.randint(0, N, size=M)
                stagnation_counter[i] = 0
            else:
                new_u = u.copy()
                for j in range(M):
                    # Use multiple attractors from archive instead of just global best
                    if random.random() < WEP and len(archive_universes) > 0:
                        # Randomly select from archive, not just best
                        selected_idx = random.randint(0, len(archive_universes) - 1)
                        selected_best = archive_universes[selected_idx]
                        
                        if random.random() < 0.5:
                            new_u[j] = selected_best[j] + TDR * (ub - lb) * random.random()
                        else:
                            new_u[j] = selected_best[j] - TDR * (ub - lb) * random.random()
                        new_u[j] = min(max(new_u[j], lb), ub)
                    
                    # Add random exploration
                    elif random.random() < 0.3:  # 30% chance of random move
                        new_u[j] = random.randint(0, N-1)
            
            new_universes[i] = np.round(new_u).astype(int)

        universes = new_universes
        fitness_history.append(best_fit_global)

    


    return archive_universes, history, archive_fitness





def update_best_archive(archive_universes, archive_fitness, universes, fitness_vals, max_size):
    best_idx = np.argmin(fitness_vals)
    best_universe = universes[best_idx].copy()
    best_fitness = fitness_vals[best_idx]

    for archived_universe in archive_universes:
        if np.sum(archived_universe != best_universe) < len(best_universe) * 0.1:
            return

    archive_universes.append(best_universe.copy())
    archive_fitness.append(best_fitness)

    # Keep only the most diverse solutions
    if len(archive_universes) > max_size:
        worst_idx = np.argmax(archive_fitness)
        archive_universes.pop(worst_idx)
        archive_fitness.pop(worst_idx)




