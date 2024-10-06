import numpy as np
import matplotlib.pyplot as plt
#from plotblit import PlotBlit

# Define the maze dimensions and the matrix (38x12)
maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],  # Row 1
    [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0],  # Row 2
    [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],  # Row 3
    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1],  # Row 4
    [1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],  # Row 5
    [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],  # Row 6
    [1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # Row 7
    [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0],  # Row 8
    [1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1],  # Row 9
    [1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,0,0,1,1,1,0,1],  # Row 10
    [1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1],  # Row 11
    [1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],  # Row 12
    [1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1],  # Row 13
])
# Create a color map:
# 0: Black for walls
# 1: White for paths
# 2: Blue for start
# 3: Red for end
cmap = {0: 'black', 1: 'white', 2: 'blue', 3: 'red'}

# Initialize the plot
fig, ax = plt.subplots(figsize=(20, 10))

# Loop through the matrix and plot each cell
for row in range(maze.shape[0]):
    for col in range(maze.shape[1]):
        color = cmap[maze[row, col]]
        rect = plt.Rectangle((col, maze.shape[0] - row - 1), 1, 1, facecolor=color)
        ax.add_patch(rect)

# Set grid and limits
ax.set_xticks(np.arange(0, maze.shape[1], 1))
ax.set_yticks(np.arange(0, maze.shape[0], 1))
ax.grid(True)
ax.set_xlim([0, maze.shape[1]])
ax.set_ylim([0, maze.shape[0]])
ax.set_aspect('equal')

plt.show()
