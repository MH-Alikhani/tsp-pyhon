import random
import numpy as np
from matplotlib import pyplot as plt

def get_distance(city1, city2):
  return np.sqrt(((city1[0] - city2[0])**2) + ((city1[1] - city2[1])**2))

def solve_tsp(n):
  # Generate random cities
  cities = []
  for _ in range(n):
    city = np.random.randint(100, size=2)
    cities.append(city)

  # Initialize tour
  tour = np.arange(n)
  random.shuffle(tour)

  # Annealing loop
  temperature = 1
  while temperature > 0.001:
    # Select two cities
    i = np.random.choice(n)
    j = np.random.choice(n)

    # Swap the cities
    new_tour = np.copy(tour)
    new_tour[i:i+2] = tour[j:j+2][::-1]

    # Calculate energy difference
    energy_diff = get_total_distance(tour) - get_total_distance(new_tour)

    # Accept or reject the new solution
    if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temperature):
      tour = np.copy(new_tour)

    # Reduce temperature
    temperature *= 0.9999

  return tour

def get_total_distance(tour):
  total_distance = 0
  for i in range(len(tour)-1):
    city1 = cities[tour[i]]
    city2 = cities[tour[i+1]]
    distance = get_distance(city1, city2)
    total_distance += distance

  return total_distance

def plot_tour(tour, cities):
  x_coords = [cities[i][0] for i in tour]
  y_coords = [cities[i][1] for i in tour]

  plt.plot(x_coords, y_coords, "xb-")
  plt.show()

# User input
n = int(input("Enter the number of cities: "))

# Solve TSP
tour = solve_tsp(n)

# Print tour
print(tour)

# Plot tour
plot_tour(tour, cities)
