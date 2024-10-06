import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (make sure the file path is correct)
image_path = "/home/samir-akhalil/Pictures/Screenshots/Screenshot from 2024-10-05 08-53-53.png"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to binary (0 for walls, 1 for paths)
_, binary_maze = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Rescale the binary maze to fit the grid size (39x13)
binary_maze = cv2.resize(binary_maze, (39, 13), interpolation=cv2.INTER_NEAREST)

# Normalize the binary maze to get values of 0 (walls) and 1 (paths)
maze = (binary_maze / 255).astype(np.int32)

# Optionally, mark start and end points
maze[0, 0] = 2  # Start point (top-left)
maze[-1, -1] = 3  # End point (bottom-right)

# Print the resulting maze matrix
print("Maze Matrix:\n", maze)

# Plot the maze for visualization
# 0: black for walls, 1: white for paths, 2: blue for start, 3: red for end
cmap = {0: 'black', 1: 'white', 2: 'blue', 3: 'red'}

fig, ax = plt.subplots(figsize=(11, 7))  # Adjust figsize for visualization

# Loop through the maze matrix and plot each cell with respective colors
for row in range(maze.shape[0]):
    for col in range(maze.shape[1]):
        color = cmap[maze[row, col]]
        rect = plt.Rectangle((col, maze.shape[0] - row - 1), 1, 1, facecolor=color)
        ax.add_patch(rect)

# Set grid and axis limits for visualization
ax.set_xticks(np.arange(0, maze.shape[1], 1))
ax.set_yticks(np.arange(0, maze.shape[0], 1))
ax.grid(True)
ax.set_xlim([0, maze.shape[1]])
ax.set_ylim([0, maze.shape[0]])
ax.set_aspect('equal')

# Show the plot
plt.show()
