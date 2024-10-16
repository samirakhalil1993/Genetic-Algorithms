import numpy as np
import random
import matplotlib.pyplot as plt
from queue import Queue



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
goal_pos = (13, 37)   # Goal at the bottom-right corner (red)Z

# Define the directions: up, down, left, right
directions = ['up', 'down', 'left', 'right']

population_size=1000
generations=10000
mutation_rate=0.1
length = 200


def fitness_function(path):
    x, y = start_pos
    penalty=0
    for move in path:
        if move == 'up':
            x = max(0, x - 1)
        elif move == 'down':
            x = min(len(maze) - 1, x + 1)
        elif move == 'left':
            y = max(0, y - 1)
        elif move == 'right':
            y = min(len(maze[0]) - 1, y + 1)
        penalty+= maze[x][y]
    return penalty 
        
       
# Generate a random path (chromosome)
def random_path(length):
    return [random.choice(directions) for _ in range(length)]

# Crossover between two paths (parents)
def crossover(path1, path2):
    split_point = random.randint(1, len(path1) - 1)
    return path1[:split_point] + path2[split_point:]

# Mutate a path by randomly changing a move
def mutate(path,mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, len(path) - 1)
        path[index] = random.choice(directions)
    return path

def selction(fitness_scores,population):
    # Select parents (tournament selection, roulette wheel, etc.)
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
    # Select top half as parents
    parents = sorted_population[:len(sorted_population)//2]
    return parents
        

def genetic_algorithm():
    population = [random_path(length) for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness_function(path) for path in population]
        best=min(fitness_scores)
        print(f"generation{gen} has best best fitness {best}")
        
        # Check if any path reaches the goal (fitness = 0)
        for i in fitness_scores:
            if i <= 1000:
                return population[i], i
    
        parents= selction(fitness_scores,population)

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



# Create a color map:
cmap = {1000: 'black', 1: 'white', 2: 'blue', 3: 'red', 'path': 'green'}

# Initialize the plot
fig, ax = plt.subplots(figsize=(20, 10))

# Loop through the matrix and plot each cell
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
    
    # Color the path as a small green circle
    circle = plt.Circle((y + 0.5, maze.shape[0] - x - 0.5), 0.1, color=cmap['path'], zorder=5)
    ax.add_patch(circle)

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
