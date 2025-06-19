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
    p=4,
    stagnation_limit=10,
):
    universes = initialize_population(N, SWARM_SIZE, M)
    lb, ub = 0, N - 1

    best_fit_global = float("inf")
    best_u_global = None
    stagnation_counter = 0

    for t in range(1, MAX_ITERATIONS + 1):
        # 1. Evaluate fitness values
        fitness_vals = []
        for u in universes:
            f, _, _, _ = fitness(u, robot_charge_duration, robots_coord,
                                 task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)
            fitness_vals.append(f)
        fitness_vals = np.array(fitness_vals)

        # Track best universe
        best_idx = np.argmin(fitness_vals)
        best_fit = fitness_vals[best_idx]
        best_u = universes[best_idx]

        # Stagnation check
        if best_fit < best_fit_global:
            best_fit_global = best_fit
            best_u_global = best_u.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # 2. Dynamic parameters
        WEP = WEP_min + (t / MAX_ITERATIONS) * (WEP_max - WEP_min)
        TDR = 1 - (t / MAX_ITERATIONS) ** (1 / p)

        # 3. Inflation rates
        inf = 1.0 / (fitness_vals + 1e-9)
        norm_inf = inf / inf.sum()

        # 4. Generate new universes
        new_universes = []
        for i, u in enumerate(universes):
            new_u = u.copy().astype(float)
            for j in range(M):
                r1 = random.random()
                if r1 < norm_inf[i]:  # White-hole
                    donor = roulette_wheel_selection(norm_inf)
                    new_u[j] = universes[donor][j]
                else:
                    r2 = random.random()
                    if r2 < WEP:  # Wormhole
                        r3 = random.random()
                        r4 = random.random()
                        distance = TDR * (ub - lb) * r4
                        if r3 < 0.5:
                            new_u[j] = best_u[j] + distance
                        else:
                            new_u[j] = best_u[j] - distance
                        new_u[j] = min(max(new_u[j], lb), ub)
            new_universes.append(np.round(new_u).astype(int))

        # 5. Adaptive Mutation if stagnation is too long
        if stagnation_counter >= stagnation_limit:
            mutation_strength = min(1.0, (stagnation_counter - stagnation_limit + 1) / 10.0)  # up to 100%
            num_mutations = int(mutation_strength * M)
            for u in new_universes:
                for _ in range(num_mutations):
                    idx = random.randint(0, M - 1)
                    u[idx] = random.randint(lb, ub)

        universes = np.array(new_universes)

    return universes
