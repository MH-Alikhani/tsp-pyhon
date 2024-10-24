# Traveling Salesman Problem Hybrid Solver

## Overview
This repository contains an implementation of a **hybrid metaheuristic solver** for the Traveling Salesman Problem (TSP). The algorithm integrates **Genetic Algorithms (GA)** with **Simulated Annealing (SA)** to effectively explore and exploit solutions, providing a balance between global and local search strategies. The solver aims to minimize the total distance traveled by visiting a set of cities exactly once and returning to the starting point.

The approach leverages GA operators like **selection**, **ordered crossover (OX)**, and **mutation (2-opt swap)** combined with **Simulated Annealing (SA)** for local refinement of solutions.

## Key Features
- **Hybrid Algorithm**: Combines GA and SA to ensure both exploration and exploitation in solution search.
- **Efficient Distance Calculation**: Utilizes `numba` for JIT-compiled distance calculations and mutation operations, enhancing performance.
- **Customizable Parameters**: Control population size, number of generations, and cooling rates in SA.
- **Visualization**: Built-in support for visualizing the final solution (tour) using `Matplotlib`.

## How It Works
1. **Initialization**: The algorithm starts by initializing a population of random tours (permutations of city indices). Each tour represents a potential solution.
2. **Fitness Evaluation**: The total distance of each tour is calculated using the Euclidean distance between cities.
3. **Selection**: A tournament selection method is used to select parents for crossover based on the shortest tour distances.
4. **Crossover**: An Ordered Crossover (OX) method is employed to combine two parent tours into offspring while preserving the city sequence as much as possible.
5. **Mutation**: A 2-opt swap mutation is applied to further explore the solution space by reversing subsections of the tour.
6. **Simulated Annealing**: Each mutated solution undergoes a Simulated Annealing process, refining the solution locally and ensuring it avoids local minima by accepting worse solutions with a probability that decreases as the temperature lowers.
7. **Evolution**: The population evolves over several generations, combining the strengths of crossover, mutation, and simulated annealing to search for better tours iteratively.
8. **Visualization**: Once the best solution is found, the tour is visualized using `matplotlib`, showing the path between cities.

## Code Structure
- **TSPSolverHybrid**: Main class that encapsulates the hybrid GA + SA logic.
  - `__init__`: Initializes the cities, population, and parameters.
  - `initialize_population`: Generates a random initial population of tours.
  - `get_distance`: Computes the Euclidean distance between two cities.
  - `get_total_distance`: Calculates the total distance of a given tour.
  - `selection`: Performs tournament selection to choose parents.
  - `ordered_crossover`: Executes ordered crossover on two parents.
  - `mutate`: Applies 2-opt mutation to a tour.
  - `simulated_annealing`: Refines the tour using simulated annealing.
  - `evolve_population`: Evolves the population using crossover, mutation, and simulated annealing.
  - `solve`: Main loop to run the hybrid algorithm for a specified number of generations.
  - `plot_tour`: Visualizes the tour on a 2D plot.
  - `generate_cities`: Generates a random set of cities with (x, y) coordinates.
  - `main`: Entry point for running the solver, prompting the user to input the number of cities and executing the algorithm.

## Dependencies
- Python 3.8+
- **NumPy**: Efficient numerical computations.
- **Matplotlib**: Visualization of results.
- **Numba**: JIT compilation for performance improvements.
- **Logging**: Integrated logging for tracking progress and results.
