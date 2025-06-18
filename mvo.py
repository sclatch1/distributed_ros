import random
import numpy as np


from fitness_ea import fitness


def roulette_wheel_selection(probs):
    """
    Selects an index i with probability proportional to probs[i].
    Assumes sum(probs) == 1.
    """
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def initialize_population(value_range,num_particles, array_size):
    # Initialize particles as arrays of integers within the specified range
    particles = np.random.randint(0, value_range, size=(num_particles, array_size))
    return particles

        

def mvo_exploration(
    MAX_ITERATIONS,
    SWARM_SIZE,
    M, N,
    robot_charge_duration,
    robots_coord,
    task,
    Charging_station,
    CHARGING_TIME,
    Energy_Harvesting_Rate,
    WEP_min=0.2,
    WEP_max=1.0,
    p=6
):
    """
    Original Multi-Verse Optimizer (Mirjalili, 2016) with improved diversity handling.

    Returns:
      best_fitness, fitness_history, universes
    """
    universes = initialize_population(N,SWARM_SIZE,M)
    lb, ub = 0, N - 1
    fitness_history = []

    for t in range(1, MAX_ITERATIONS + 1):
        # 1. Evaluate fitness values
        fitness_vals = []
        for u in universes:

            f, _ , _, _ = fitness(u, robot_charge_duration, robots_coord,
                               task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)

            fitness_vals.append(f)
        fitness_vals = np.array(fitness_vals)

        # Track best universe
        best_idx = np.argmin(fitness_vals)
        best_u = universes[best_idx]
        best_fit = fitness_vals[best_idx]
        fitness_history.append(best_fit)

        # 2. Dynamic parameters
        WEP = WEP_min + (t / MAX_ITERATIONS) * (WEP_max - WEP_min)
        TDR = 1 - (t / MAX_ITERATIONS) ** (1 / p)

        # 3. Inflation rates and normalization
        inf = 1.0 / (fitness_vals + 1e-9)
        norm_inf = inf / inf.sum()
        # print(norm_inf)
        # 4. Generate new universes
        new_universes = []
        for i, u in enumerate(universes):
            new_u = u.copy().astype(float)
            for j in range(M):
                r1 = random.random()
                # White-hole: copy from another universe
                if r1 < norm_inf[i]:
                    donor = roulette_wheel_selection(norm_inf)
                    new_u[j] = universes[donor][j]
                else:
                    # Wormhole: travel towards best universe
                    r2 = random.random()
                    if r2 < WEP:
                        r3 = random.random()
                        r4 = random.random()
                        # Correct distance formula
                        distance = TDR * (ub - lb) * r4
                        if r3 < 0.5:
                            new_u[j] = best_u[j] + distance
                        else:
                            new_u[j] = best_u[j] - distance
                        # clamp within bounds
                        new_u[j] = min(max(new_u[j], lb), ub)
                    # else: keep original u[j]
            # Convert to int after all updates
            new_universes.append(np.round(new_u).astype(int))

        universes = np.array(new_universes)
        # Debug: check diversity
        unique_count = len(np.unique(universes, axis=0))
        #print(f"Iteration {t}: Unique universes = {unique_count}")
        if unique_count < 50:
            break

    return best_fit, fitness_history, universes