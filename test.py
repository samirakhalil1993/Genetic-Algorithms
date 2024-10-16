import numpy as np
import random
import matplotlib.pyplot as plt

# Maze: 1 = open path, 1000 = wall
maze = np.array([
    [2,1,1,1,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1],  
    [1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1,1,1,1000,1000],  
    [1,1,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1000,1,1,1,1,1,1,1,1,1,1,1,1000,1,1000,1,1,1],  
    [1,1,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1],  
    [1,1,1,1,1,1000,1,1,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1000,1],  
    [1000,1000,1000,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1],  
    [1,1,1,1000,1,1,1,1000,1,1,1,1,1,1,1,1,1,1000,1,1000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  
    [1,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1,1000,1,1000,1000,1000,1,1000,1000,1000,1000,1000,1000,1000,1000,1000,1,1000,1000,1000,1000],  
    [1,1000,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1000,1,1000,1,1000,1,1,1,1,1,1,1,1000,1,1,1,1,1,1000,1,1,1],  
    [1,1000,1000,1000,1000,1000,1000,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1000,1,1,1,1,1,1000,1,1000,1000,1000,1,1,1,1000,1],  
    [1,1000,1,1,1,1,1,1000,1,1000,1,1,1,1000,1,1000,1,1,1,1000,1,1000,1,1000,1,1,1,1,1,1000,1,1,1,1000,1,1,1,1000,1],  
    [1,1000,1,1000,1000,1,1,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1000,1000,1,1000,1000,1000,1],  
    [1,1000,1,1,1000,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1,1,1000,1,1,1,1,1,1,1,1,1,1,1,1,1,1000,1,3,1],  
])

# Start and goal positions
start_pos = (0, 0)  # Start at the top-left corner (blue)
goal_pos = (12, 38)  # Goal at the bottom-right corner (red)
length = 200
# Define the directions: up, down, left, right
directions = ['up', 'down', 'left', 'right']

# Genetic Algorithm Parameters
population_size = 100
generations = 10000
mutation_rate = 0.1
length=150

def fitness_function(path):
    start_pos = (0, 0)  # Start at the top-left corner (blue)
    goal_pos = (12, 37)  # Goal at the bottom-right corner (red)
    time = 0
    for action in path:
        time += 1
        if action == 'up':
            start_pos = (start_pos[0], start_pos[1]+1)   
        elif action == 'down':
            start_pos = (start_pos[0], start_pos[1]-1)   
        elif action == 'left':
            start_pos = (start_pos[0]+1, start_pos[1])   
        elif action == 'right':
            start_pos = (start_pos[0]-1, start_pos[1])   
        if start_pos == goal_pos:
            return 1.0/time
    return 0.0



    
# Generate a random path (chromosome)
def random_path(length):
    return [random.choice(directions) for _ in range(length)]

# Crossover between two paths (parents)
def crossover(path1, path2):
    split_point = random.randint(1, len(path1) - 1)
    return path1[:split_point] + path2[split_point:]

# Mutate a path by randomly changing a move
def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, len(path) - 1)
        path[index] = random.choice(directions)
    return path

def genetic_algorithm():
    population = [random_path(length) for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness-
        fitness_scores = [fitness_function(path) for path in population]
        
        # Check if any path reaches the goal (fitness = 0)
        if 0 in fitness_scores:
            best_index = fitness_scores.index(0)
            return population[best_index], 0
        print(f"generation: {gen} fitness score: {fitness_scores[0]}")
        # Select parents based on fitness (lower fitness is better)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
        
        # Select top half as parents
        parents = sorted_population[:len(sorted_population)//2]
        
        # Create next generation through crossover and mutation
        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_population.append(child)
        
        population = next_population
    
    # Return the best solution after all generations
    fitness_scores = [fitness_function(path) for path in population]
    best_index = fitness_scores.index(min(fitness_scores))
    best_path = population[best_index]
    best_fitness = fitness_scores[best_index]
    
    return best_path, best_fitness


# Example run
best_path, fitness = genetic_algorithm()
print("Best path:", best_path)
print("Fitness:", fitness)

# Plotting

cmap = {1000: 'black', 1: 'white', 2: 'blue', 3: 'red', 'path': 'green'}

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the maze
for row in range(maze.shape[0]):
    for col in range(maze.shape[1]):
        color = cmap[maze[row, col]]
        rect = plt.Rectangle((col, maze.shape[0] - row - 1), 1, 1, facecolor=color)
        ax.add_patch(rect)

# Plot the best path
x, y = start_pos
for move in best_path:
    if move == 'up':
        x = max(0, x - 1)
    elif move == 'down':
        x = min(maze.shape[0] - 1, x + 1)
    elif move == 'left':
        y = max(0, y - 1)
    elif move == 'right':
        y = min(maze.shape[1] - 1, y + 1)
    
    # Color the path as green
    rect = plt.Rectangle((y, maze.shape[0] - x - 1), 1, 1, facecolor=cmap['path'])
    ax.add_patch(rect)

# Redraw the start and goal to keep their original colors
rect = plt.Rectangle((start_pos[1], maze.shape[0] - start_pos[0] - 1), 1, 1, facecolor=cmap[2])  # Blue start
ax.add_patch(rect)
rect = plt.Rectangle((goal_pos[1], maze.shape[0] - goal_pos[0] - 1), 1, 1, facecolor=cmap[3])  # Red goal
ax.add_patch(rect)

# Set grid and limits
ax.set_xticks(np.arange(0, maze.shape[1], 1))
ax.set_yticks(np.arange(0, maze.shape[0], 1))
ax.grid(True)
ax.set_xlim([0, maze.shape[1]])
ax.set_ylim([0, maze.shape[0]])
ax.set_aspect('equal')

plt.show()
