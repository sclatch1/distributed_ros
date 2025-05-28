import random
import copy
from pso import PSO_Algorithm
from fitness_eu import fitnessEU
from fitness_ea import fitness


import math
import numpy as np

def roulette_wheel_selection(weights):
    total = sum(weights)
    pick = random.uniform(0, total) 
    current = 0
    for idx, w in enumerate(weights):
        current += w
        if current >= pick:
            return idx
    return len(weights) - 1



def mvo_exploration(MAX_ITERATIONS, SWARM_SIZE,
                             M, N, s, iterationstop,
                             robot_charge_duration, robots_coord,
                             task, Charging_station, CHARGING_TIME,
                    Energy_Harvesting_Rate, init_universe,
                             WEP_min=0.2, WEP_max=1.0, p=6):
    # --- Phase 1: MVO Exploration with dynamic WEP/TDR ---
    universes = init_universe 
    
    # write_np_to_file(universes, "universe", "etc")

    fitnesses = []
    for t in range(1, MAX_ITERATIONS+1):
        # evaluate
        fitness_vals = []
        for u in universes:
            val = (fitness(u, robot_charge_duration, robots_coord, task,
                           Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)[0]
                   if s==1 else
                   fitnessEU(u, robot_charge_duration, robots_coord, task,
                             Charging_station, CHARGING_TIME)[0])
            fitness_vals.append(val)
        best_fit = min(fitness_vals)

        # compute WEP and TDR
        # WEP = WEP_min + t*((WEP_max - WEP_min)/MAX_ITERATIONS)
        # TDR = 1 - (t**(1/p))/(MAX_ITERATIONS**(1/p))
        
        WEP = WEP_min + t * ((0.8 - WEP_min)/MAX_ITERATIONS)
        TDR = 1 - (t / MAX_ITERATIONS)**(1/p)
        
        # update via white hole (roulette) & wormhole
        inf_vals = [1.0/(fv+1e-9) for fv in fitness_vals]
        normalized_inf = [iv/sum(inf_vals) for iv in inf_vals]
        # pre-sort universes by inf for wormhole donor
        best_idx = np.argmax(inf_vals)
        best_u = universes[best_idx]

        new_univ = []
        for idx, u in enumerate(universes):
            nu = u.copy()
            for j in range(M):
                r = random.random()
                # Wormhole toward best
                if r < WEP:
                    delta = int(round(TDR * (N-1) * random.random()))
                    if random.random() < 0.5:
                        nu[j] = best_u[j] + delta
                    else:
                        nu[j] = best_u[j] - delta
                    # clamp to [0, N-1]
                    nu[j] = min(max(nu[j], 0), N-1)
                # White‑hole selection from another universe
                elif r < WEP + normalized_inf[idx]:
                    donor = roulette_wheel_selection(inf_vals)
                    nu[j] = universes[donor][j]
                # else: keep own u[j]
            new_univ.append(nu)
        # Convert to NumPy array for easier processing
        new_univ = np.array(new_univ)
        unique_count = len(np.unique(new_univ, axis=0))
        # print(f"Iteration {t}: Unique solutions = {unique_count}")

        # # Diversity injection if too homogeneous
        # if unique_count < SWARM_SIZE // 2:
        #     num_to_inject = SWARM_SIZE - unique_count
        #     print(f"Injecting {num_to_inject} new random universes to preserve diversity.")
        #     new_individuals = initialize_population(num_to_inject, M, N)
        #     indices_to_replace = np.random.choice(SWARM_SIZE, size=num_to_inject, replace=False)
        #     new_univ[indices_to_replace] = new_individuals
        #
        #
        # if unique_count < SWARM_SIZE // 2:
        #     print(f"[DIVERSITY] Before injection: {unique_count} unique")
        #     # pick indices
        #     indices_to_replace = np.random.choice(SWARM_SIZE, size=num_to_inject, replace=False)
        #     new_indiv = initialize_population(num_to_inject, M, N)
        #     # check overlap
        #     overlap = sum((new_univ[indices_to_replace] == new_indiv).all(axis=1))
        #     print(f"[DIVERSITY] Overlap between old & new injected: {overlap}/{num_to_inject}")
        #     new_univ[indices_to_replace] = new_indiv
        

        # # Apply random mutation
        # mutation_rate = 0.05
        # for i in range(SWARM_SIZE):
        #     if np.random.rand() < mutation_rate:
        #         j = np.random.randint(0, M)
        #         new_univ[i][j] = np.random.randint(0, N)

        universes = new_univ
        fitnesses.append(best_fit)
        if len(fitnesses) >= iterationstop:
            # Get the last five elements from the fitnesses array
            last_five = fitnesses[-iterationstop:]

            # Check if all the last five elements are equal
            if all(x == last_five[0] for x in last_five):
                break
    
    universes = np.array(universes)
    # write_np_to_file(universes, "final_universe", "etc")

    # --- Phase 2: Elite Seeding + Perturbation ---
    scored = [(fitness(u, robot_charge_duration, robots_coord, task,
                       Charging_station, CHARGING_TIME, Energy_Harvesting_Rate)[0]
               if s==1 else
               fitnessEU(u, robot_charge_duration, robots_coord, task,
                         Charging_station, CHARGING_TIME)[0], u)
              for u in universes]
    scored.sort(key=lambda x: x[0])
    k = min(2, SWARM_SIZE)
    elites = [u for _, u in scored[:k]]
    
    elites_array = np.array(elites, dtype=int)
    # write_np_to_file(elites_array, 'mvo_array', 'mvo_exploration')

    return best_fit, fitnesses, elites_array, universes


