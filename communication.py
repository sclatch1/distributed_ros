import numpy as np
import matplotlib.pyplot as plt

# Constants
BYTES_PER_INT = 4
BITS_PER_BYTE = 8
E_ELEC = 50e-9
E_AMP = 100e-12
PATH_LOSS_EXPONENT = 2

# Functions
def compute_message_size(num_candidates, num_arrays):
    return num_candidates * num_arrays * BYTES_PER_INT + 2 * BYTES_PER_INT

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def energy_to_transmit(bits, distance):
    return bits * (E_ELEC + E_AMP * distance ** PATH_LOSS_EXPONENT)

def generate_robot_positions(num_robots, radius):
    """Place robots evenly spaced on a circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

def simulate_energy_consumption(central_position, robot_positions, num_candidates, total_arrays):
    distances = [compute_distance(central_position, rp) for rp in robot_positions]
    bits_per_array = compute_message_size(num_candidates, 1) * BITS_PER_BYTE
    energy_costs = [energy_to_transmit(bits_per_array, d) for d in distances]

    inv_costs = [1 / e for e in energy_costs]
    total_inv = sum(inv_costs)
    weights = [ic / total_inv for ic in inv_costs]

    array_allocations = [round(total_arrays * w) for w in weights]

    total_energy = sum(
        energy_to_transmit(compute_message_size(num_candidates, a) * BITS_PER_BYTE, d)
        for a, d in zip(array_allocations, distances)
    )

    return total_energy

# Parameters
central_position = (0, 0)
num_robots = 5
num_candidates = 50
total_arrays = 30
radii = list(range(5, 55, 5))  # Vary distances: 5m, 10m, ..., 50m

# Run simulation
total_energies = []
for r in radii:
    robot_positions = generate_robot_positions(num_robots, radius=r)
    total_energy = simulate_energy_consumption(central_position, robot_positions, num_candidates, total_arrays)
    total_energies.append(total_energy * 1e6)  # Convert to µJ

# Plot
plt.figure(figsize=(8, 5))
plt.plot(radii, total_energies, marker='o', color='darkblue')
plt.xlabel("Average Distance to Central Node (m)")
plt.ylabel("Total Energy Consumed (µJ)")
plt.title("Energy Consumption vs. Distance to Central Node")
plt.grid(True)
plt.tight_layout()
plt.show()
