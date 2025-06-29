import numpy as np
from fitness_ea import fitness
import os


def multiverse_algorithm(MAX_ITERATIONS,
                        SWARM_SIZE,
                        M, N,
                        robot_charge_duration,
                        robots_coord,
                        task,
                        Charging_station,
                        CHARGING_TIME,
                        Energy_Harvesting_Rate):
    """
    Multiverse Algorithm (MVA) implementation based on multiverse theory
    
    Parameters:
    - MAX_ITERATIONS: Maximum number of iterations
    - SWARM_SIZE: Population size (number of universes)
    - M: Dimension of each solution
    - N: Upper bound for solution values
    - eps: Small positive number for convergence criteria
    - capture_steps: If True, captures visualization steps for PCA plotting
    - Other parameters: Problem-specific parameters for fitness function
    
    Returns:
    - best_solution: Best solution found
    - best_fitness: Best fitness value
    - fitness_history: History of best fitness values
    - visualization_steps: Dict containing the 6 steps for visualization (if capture_steps=True)
    """

    MAX_ITERATIONS = 50
    eps = max(1, M * 0.05)

    
    # Step 1: Initialize population (Algorithm 1)
    universes = generate_initial_population(SWARM_SIZE, M, N)
    
    # Evaluate initial fitness
    fitness_vals = np.array([
        fitness(u, robot_charge_duration, robots_coord,
                task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)[0]
        for u in universes
    ])
    
    # Track best solution
    best_idx = np.argmin(fitness_vals)
    best_fitness = float(fitness_vals[best_idx])
    best_solution = universes[best_idx].copy()
    fitness_history = [best_fitness]
    history = []
    # Main MVA loop
    for k in range(MAX_ITERATIONS):
        history.append(universes)
        #  Sort universes according to their fitness (dark energy concept)
        sorted_indices = np.argsort(fitness_vals)
        sorted_universes = universes[sorted_indices]
        sorted_fitness = fitness_vals[sorted_indices]
        
        # Step 2: Explosion of solutions (Algorithm 2)
        new_universes = explosion_of_solutions(sorted_universes, sorted_fitness, 
                                             M, N, eps, k, MAX_ITERATIONS)
        
        # Evaluate new solutions
        new_fitness_vals = np.array([
            fitness(u, robot_charge_duration, robots_coord,
                    task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)[0]
            for u in new_universes
        ])
        
        # Update population with better solutions
        universes, fitness_vals = update_population(universes, fitness_vals,
                                                  new_universes, new_fitness_vals)
        

        

        # Update best solution
        current_best_idx = np.argmin(fitness_vals)
        current_best_fitness = float(fitness_vals[current_best_idx])
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = universes[current_best_idx].copy()
        
        fitness_history.append(best_fitness)
        
        
        # Convergence check
        if k > 0 and abs(fitness_history[-1] - fitness_history[-2]) < eps:
            print(f"Converged at iteration {k}")
            break
        
        

    return universes, history ,best_fitness


def generate_initial_population(swarm_size, m, n):
    """
    Algorithm 1: Generate initial population of solutions
    
    Parameters:
    - swarm_size: Number of solutions (universes)
    - m: Dimension of each solution
    - n: Upper bound for solution values
    
    Returns:
    - universes: Initial population matrix
    """
    # Generate feasible solutions randomly
    universes = np.random.randint(0, n, size=(swarm_size, m))
    return universes


def explosion_of_solutions(sorted_universes, sorted_fitness, m, n, eps, iteration, max_iterations):
    """
    Algorithm 2: Explosion of solutions based on multiverse theory
    
    Parameters:
    - sorted_universes: Universes sorted by fitness (dark energy)
    - sorted_fitness: Corresponding fitness values
    - m: Dimension of solutions
    - n: Upper bound for solution values
    - eps: Small positive number
    - iteration: Current iteration
    - max_iterations: Maximum iterations
    
    Returns:
    - new_universes: New population after explosion
    """
    swarm_size = len(sorted_universes)
    new_universes = []
    
    # Calculate alpha (exploration coefficient that decreases over time)
    alpha = 2 * (1 - iteration / max_iterations)
    
    for i in range(swarm_size):
        current_universe = sorted_universes[i].copy()
        
        # Rank-based number of solutions to generate
        rank = i + 1
        # Better solutions (lower rank) generate more new solutions
        num_solutions = max(1, int(swarm_size / rank))
        
        for j in range(num_solutions):
            if j < len(sorted_universes):
                # Generate new solution based on movement equation: x_j = x_i + λd_ij
                prev_universe = sorted_universes[j]
                
                # Calculate distance vector
                distance_vector = current_universe - prev_universe
                
                # Apply movement with random coefficient λ (lambda)
                lambda_coeff = np.random.uniform(-alpha, alpha, m)
                
                # Generate new solution
                new_solution = current_universe + lambda_coeff * distance_vector
                
                # Ensure feasibility (clamp to valid range)
                new_solution = np.clip(new_solution, 0, n-1).astype(int)
                
                # Check if solution is feasible (within epsilon tolerance)
                if np.linalg.norm(new_solution - current_universe) <= eps or np.random.random() < 0.1:
                    new_universes.append(new_solution)
    
    # If no new solutions generated, create some random ones
    if len(new_universes) == 0:
        new_universes = [np.random.randint(0, n, size=m) for _ in range(swarm_size)]
    
    # Ensure we have enough solutions
    while len(new_universes) < swarm_size:
        new_universes.append(np.random.randint(0, n, size=m))
    
    return np.array(new_universes[:swarm_size])


def update_population(old_universes, old_fitness, new_universes, new_fitness):
    """
    Update population by selecting better solutions
    
    Parameters:
    - old_universes: Current population
    - old_fitness: Current fitness values
    - new_universes: New candidate solutions
    - new_fitness: New fitness values
    
    Returns:
    - updated_universes: Updated population
    - updated_fitness: Updated fitness values
    """
    # Combine old and new populations
    combined_universes = np.vstack([old_universes, new_universes])
    combined_fitness = np.concatenate([old_fitness, new_fitness])
    
    # Select best solutions
    swarm_size = len(old_universes)
    best_indices = np.argsort(combined_fitness)[:swarm_size]
    
    updated_universes = combined_universes[best_indices]
    updated_fitness = combined_fitness[best_indices]
    
    return updated_universes, updated_fitness


def better_solution(x1, x2, f1, f2):
    """
    Determine which solution is better based on the BS function from the paper
    For minimization problems: BS = x1 if f1 < f2, else x2
    """
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2

