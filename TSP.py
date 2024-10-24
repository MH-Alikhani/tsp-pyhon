import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import random
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class TSPSolverHybrid:
    def __init__(self, cities: np.ndarray, population_size: int = 100, seed: int = None):
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.population = self.initialize_population()
        self.best_tour = None
        self.best_distance = float('inf')

    # Initialize population with random tours
    def initialize_population(self) -> List[np.ndarray]:
        population = [np.random.permutation(self.num_cities) for _ in range(self.population_size)]
        return population

    @staticmethod
    @njit
    def get_distance(city1: np.ndarray, city2: np.ndarray) -> float:
        return np.linalg.norm(city1 - city2)

    @njit
    def get_total_distance(self, tour: np.ndarray) -> float:
        total_distance = 0.0
        for i in range(self.num_cities - 1):
            total_distance += self.get_distance(self.cities[tour[i]], self.cities[tour[i + 1]])
        total_distance += self.get_distance(self.cities[tour[-1]], self.cities[tour[0]])  # Return to start
        return total_distance

    # Selection based on tournament selection
    def selection(self, k: int = 5) -> np.ndarray:
        # Randomly select k individuals and choose the one with the best fitness (smallest distance)
        tournament = random.sample(self.population, k)
        tournament.sort(key=lambda x: self.get_total_distance(x))
        return tournament[0]

    # Ordered crossover (OX)
    def ordered_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        start, end = sorted(random.sample(range(self.num_cities), 2))
        offspring = np.full(self.num_cities, -1)
        offspring[start:end] = parent1[start:end]
        
        pos = end
        for city in parent2:
            if city not in offspring:
                if pos >= self.num_cities:
                    pos = 0
                offspring[pos] = city
                pos += 1
        return offspring

    # Mutation: Apply 2-opt swap for mutation
    @staticmethod
    @njit
    def two_opt_swap(tour: np.ndarray, i: int, j: int) -> np.ndarray:
        new_tour = np.copy(tour)
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        return new_tour

    def mutate(self, tour: np.ndarray) -> np.ndarray:
        i, j = sorted(random.sample(range(self.num_cities), 2))
        return self.two_opt_swap(tour, i, j)

    # Apply simulated annealing to refine an individual
    def simulated_annealing(self, tour: np.ndarray, initial_temp: float = 1000, min_temp: float = 1e-3, cooling_rate: float = 0.999) -> np.ndarray:
        current_distance = self.get_total_distance(tour)
        temperature = initial_temp

        while temperature > min_temp:
            i, j = sorted(random.sample(range(self.num_cities), 2))
            new_tour = self.two_opt_swap(tour, i, j)
            new_distance = self.get_total_distance(new_tour)

            if new_distance < current_distance or np.random.rand() < np.exp((current_distance - new_distance) / temperature):
                tour = new_tour
                current_distance = new_distance

            temperature *= cooling_rate

        return tour

    # Evolve population through crossover, mutation, and SA
    def evolve_population(self) -> None:
        new_population = []
        for _ in range(self.population_size // 2):
            parent1 = self.selection()
            parent2 = self.selection()
            offspring1 = self.ordered_crossover(parent1, parent2)
            offspring2 = self.ordered_crossover(parent2, parent1)
            
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # Apply simulated annealing to offspring
            offspring1 = self.simulated_annealing(offspring1)
            offspring2 = self.simulated_annealing(offspring2)

            new_population.extend([offspring1, offspring2])

        # Replace the population with the new one
        self.population = new_population

        # Update the best tour and distance found
        for tour in self.population:
            distance = self.get_total_distance(tour)
            if distance < self.best_distance:
                self.best_tour = tour
                self.best_distance = distance
                logging.info(f"New best distance: {self.best_distance:.2f}")

    # Main loop to run GA + SA
    def solve(self, generations: int = 500) -> Tuple[np.ndarray, float]:
        for generation in range(generations):
            logging.info(f"Generation {generation + 1}/{generations}")
            self.evolve_population()

        logging.info(f"Best distance found: {self.best_distance:.2f}")
        return self.best_tour, self.best_distance

    # Plot the cities and the tour
    def plot_tour(self, tour: np.ndarray) -> None:
        plt.figure(figsize=(10, 6))
        x_coords = self.cities[tour][:, 0]
        y_coords = self.cities[tour][:, 1]
        plt.plot(np.append(x_coords, x_coords[0]), np.append(y_coords, y_coords[0]), 'xb-', label='Tour Path')
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='r', label='Cities')
        plt.title(f"Tour - Total Distance: {self.best_distance:.2f}")
        plt.legend()
        plt.grid(True)
        plt.show()

# Generate random cities
def generate_cities(n: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n, 2) * 100  # Coordinates between 0 and 100

# Main execution logic
def main():
    n = int(input("Enter the number of cities: "))
    if n < 3:
        raise ValueError("The number of cities must be at least 3.")

    cities = generate_cities(n, seed=42)
    solver = TSPSolverHybrid(cities, population_size=100, seed=42)
    best_tour, best_distance = solver.solve(generations=500)

    print(f"Best tour found: {best_tour}")
    print(f"Total distance: {best_distance:.2f}")
    solver.plot_tour(best_tour)

if __name__ == "__main__":
    main()
