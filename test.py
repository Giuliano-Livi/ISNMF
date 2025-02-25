import matplotlib.pyplot as plt
import numpy as np

# Create some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create a figure and a grid of subplots with 3 rows and 1 column
fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # 3 rows, 1 column

# Plot the first subplot (row 1)
axs[0].plot(x, y1, color='blue')
axs[0].set_title('Sine Wave')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')

# Plot the second subplot (row 2)
axs[1].plot(x, y2, color='red')
axs[1].set_title('Cosine Wave')
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis')

# Plot the third subplot (row 3)
axs[2].plot(x, y3, color='green')
axs[2].set_title('Tangent Wave')
axs[2].set_xlabel('X-axis')
axs[2].set_ylabel('Y-axis')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()