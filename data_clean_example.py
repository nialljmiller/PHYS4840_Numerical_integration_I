import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3e6, 1000)  # Some x data
y = np.exp(-x / 1e5) * 2.5e-13  # some y data

# Suppose I only want data where y is bigger than some value
threshold_y = 0#.2e-13     
filtered_indices = y > threshold_y #find the places (indices) where the y values are bigger than the treshold
x_filtered = x[filtered_indices]   #use the locations of those places to create a new list of x
y_filtered = y[filtered_indices]   # and y

plt.scatter(x_filtered, y_filtered)
plt.axhline(y=threshold_y, color="red", linestyle="--", label=f"Cutoff: {threshold_y:.2e}")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()
