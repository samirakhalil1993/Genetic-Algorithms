import numpy as np
import random
import matplotlib.pyplot as plt


# Maze: 1 = open path, 1000 = wall
maze = np.array([
    [2,1,1,1,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1],  # Row 1
    [1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1,1,1,1000,1000],  # Row 2
    [1,1,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1000,1,1,1,1,1,1,1,1,1,1,1,1000,1,1000,1,1,1],  # Row 3
    [1,1,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1],  # Row 4
    [1,1,1,1,1,1000,1,1,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1000,1],  # Row 5
    [1000,1000,1000,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1],  # Row 6
    [1,1,1,1000,1,1,1,1000,1,1,1,1,1,1,1,1,1,1000,1,1000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # Row 7
    [1,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000],  # Row 8
    [1,1000,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1000,1,1000,1,1000,1,1,1,1,1,1,1,1000,1,1,1,1,1,1000,1,1,1],  # Row 9
    [1,1000,1000,1000,1000,1000,1000,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1,1,1,1,1000,1,1000,1000,1000,1,1,1,1000,1],  # Row 10
    [1,1000,1,1,1,1,1,1000,1,1000,1,1,1,1000,1,1000,1,1,1,1000,1,1000,1,1000,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1000,1],  # Row 11
    [1,1000,1,1000,1000,1,1,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1],  # Row 12
    [1,1000,1,1,1000,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1,1,1,1,1000,1,3,1],  # Row 13
])

# Start and goal positions
start_pos = (0, 0)  # Start at the top-left corner (blue)
goal_pos = (13, 38)   # Goal at the bottom-right corner (red)Z

import numpy as np
import random

# Directions: up, down, left, right
directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

# Fitness function: reward getting closer to the goal, penalize hitting walls
def fitness(individual, maze, goal_pos):
    pos = start_pos
    penalty = 1000  # Hög straffavgift för att slå i väggar
    total_cost = 0
    wall_hits = 0  # Räkna antalet väggträffar
    
    for move in individual:
        new_pos = (pos[0] + move[0], pos[1] + move[1])
        
        # Kolla om den nya positionen är utanför labyrinten eller träffar en vägg
        if new_pos[0] < 0 or new_pos[0] >= maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= maze.shape[1] or maze[new_pos[0], new_pos[1]] == 1000:
            total_cost += penalty
            wall_hits += 1
        else:
            pos = new_pos
            total_cost += np.linalg.norm(np.array(pos) - np.array(goal_pos))  # Avstånd till mål
            
            if pos == goal_pos:
                return 0  # Om vi når målet, returnera bästa möjliga fitness
    
    # Lägg till extra straff för många väggträffar
    return total_cost + wall_hits * 500


def initialize_population(size, max_moves):
    population = []
    for _ in range(size):
        individual = [random.choice(directions) for _ in range(max_moves)]
        population.append(individual)
    return population

# Lägg till en diversifieringsfunktion:
def diversify_population(population, new_individuals_count, max_moves):
    for _ in range(new_individuals_count):
        individual = [random.choice(directions) for _ in range(max_moves)]
        population.append(individual)


# Selection: tournament selection
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])  # Sort by fitness
    return selected[0][0]  # Return the best individual

# Crossover: single-point crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

# Mutation: random move mutation
def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(directions)
    return individual

# Genetic Algorithm
def genetic_algorithm(maze, start_pos, goal_pos, pop_size=100, max_moves=500, generations=500, mutation_rate=0.2):
    population = initialize_population(pop_size, max_moves)
    
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [fitness(individual, maze, goal_pos) for individual in population]
        
        # Check if a solution is found
        if 0 in fitnesses:
            solution = population[fitnesses.index(0)]
            print(f"Solution found in generation {generation}!")
            return solution
        
        # Create new population
        new_population = []
        for _ in range(pop_size // 2):
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            
            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population
    
    print("No solution found.")
    return None

# Run the Genetic Algorithm
solution = genetic_algorithm(maze, start_pos, goal_pos)

# Visualize the maze with the solution
if solution:
    pos = start_pos
    path = [start_pos]
    for move in solution:
        new_pos = (pos[0] + move[0], pos[1] + move[1])
        if new_pos[0] < 0 or new_pos[0] >= maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= maze.shape[1] or maze[new_pos[0], new_pos[1]] == 1000:
            break
        pos = new_pos
        path.append(pos)
    
    maze_copy = np.copy(maze)
    for p in path:
        maze_copy[p[0], p[1]] = 5  # Mark the path
    
    plt.imshow(maze_copy)
    plt.show()
def visualize_solution(individual, maze):
    pos = start_pos
    path = [start_pos]
    
    for move in individual:
        new_pos = (pos[0] + move[0], pos[1] + move[1])
        if new_pos[0] < 0 or new_pos[0] >= maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= maze.shape[1] or maze[new_pos[0], new_pos[1]] == 1000:
            break
        pos = new_pos
        path.append(pos)
    
    maze_copy = np.copy(maze)
    for p in path:
        maze_copy[p[0], p[1]] = 5  # Markera vägen
    
    plt.imshow(maze_copy)
    plt.show()

# Anropa denna funktion för att visualisera de bästa individerna under några generationer
#visualize_solution(best_individual, maze)
