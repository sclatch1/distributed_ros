from fitness_ea import fitness
import random
import copy
import numpy as np

# --- Reproducible randomness ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

V_MAX_FACTOR = 0.5  # max velocity as fraction of search range


def initialize_swarm(swarm_size: int, M: int, N: int):
    """
    Initialize swarm positions and velocities as numpy arrays.
    Positions: uniform ints in [0, N).
    Velocities: uniform floats in [-Vmax, Vmax].
    """
    # positions
    swarm = np.random.randint(0, N, size=(swarm_size, M))
    # velocity clamp based on range
    v_max = V_MAX_FACTOR * (N - 1)
    velocities = np.random.uniform(-v_max, v_max, size=(swarm_size, M))
    return swarm, velocities


def PSO_Algorithm(
    MAX_ITERATIONS: int,
    SWARM_SIZE: int,
    M: int,
    N: int,
    iterationstop: int,
    robot_charge_duration,
    robots_coord,
    task,
    Charging_station,
    CHARGING_TIME,
    Energy_Harvesting_Rate,
    init_swarm=None
):

    # --- PSO Hyperparameters ---
    C1 = 2.0  # cognitive coefficient
    C2 = 2.0  # social coefficient
    W_MAX = 0.9  # initial inertia weight
    W_MIN = 0.4  # final inertia weight


    # --- Initialization ---
    if init_swarm is None or len(init_swarm) == 0:
        swarm, velocities = initialize_swarm(SWARM_SIZE, M, N)
    else:
        # assume init_swarm is array-like
        swarm = np.array(init_swarm, dtype=int)
        _, velocities = initialize_swarm(SWARM_SIZE, M, N)

    # personal bests
    pbest_positions = swarm.copy()
    pbest_fitnesses = np.full(SWARM_SIZE, np.inf)

    # global best
    gbest_position = None
    gbest_fitness = np.inf

    # pre-calculate velocity clamp
    v_max = V_MAX_FACTOR * (N - 1)

    fitness_history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        # inertia weight decay
        W = W_MAX - (W_MAX - W_MIN) * (iteration / MAX_ITERATIONS)

        for i in range(SWARM_SIZE):
            # evaluate
            fit_val, _, _, _ = fitness(
                swarm[i], robot_charge_duration, robots_coord,
                task, Charging_station, CHARGING_TIME, Energy_Harvesting_Rate
            )

            # update personal best
            if fit_val < pbest_fitnesses[i]:
                pbest_fitnesses[i] = fit_val
                pbest_positions[i] = swarm[i].copy()

            # update global best
            if fit_val < gbest_fitness:
                gbest_fitness = fit_val
                gbest_position = swarm[i].copy()

        # record global best
        fitness_history.append(gbest_fitness)

        # stopping criterion
        if len(fitness_history) >= iterationstop:
            last_vals = fitness_history[-iterationstop:]
            if all(val == last_vals[0] for val in last_vals):
                break

        # velocity & position updates
        for i in range(SWARM_SIZE):
            r1 = np.random.rand(M)
            r2 = np.random.rand(M)
            cognitive = C1 * r1 * (pbest_positions[i] - swarm[i])
            social    = C2 * r2 * (gbest_position - swarm[i])

            # update velocity with clamp
            velocities[i] = W * velocities[i] + cognitive + social
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # update position and clip to bounds
            swarm[i] = np.clip(np.round(swarm[i] + velocities[i]), 0, N - 1).astype(int)

    return gbest_fitness, swarm
