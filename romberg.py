import numpy as np

def romberg_rule(f, a, b, max_order):
    """
    Approximates the integral using Romberg's method.

    Parameters:
        f (function): The function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))  # Create a Romberg table
    N = 1  # Start with a single interval

    # First approximation using the trapezoidal rule
    h = (b - a)
    R[0, 0] = (h / 2) * (f(a) + f(b))

    for i in range(1, max_order):
        N *= 2  # Double the number of intervals
        h /= 2  # Halve the step size

        # Compute the new trapezoidal approximation
        sum_new_points = sum(f(a + k * h) for k in range(1, N, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_new_points

        # Compute extrapolated Romberg values
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]  # Return the most refined estimate


# Example usage with a function
f = lambda x: np.exp(-x**2)  # Function to integrate
a, b = 0, 1  # Integration bounds
max_order = 5  # Romberg order

integral_approx = romberg_rule(f, a, b, max_order)
print("Approximated Integral with Romberg's Rule (max_order=5):", integral_approx)


# Romberg integration for empirical data
def empirical_romberg_rule(y_values, x_values, max_order):
    """
    Approximates the integral using Romberg's method for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))
    a, b = x_values[0], x_values[-1]
    N = 1
    h = (b - a)

    # First trapezoidal estimate
    R[0, 0] = (h / 2) * (y_values[0] + y_values[-1])

    for i in range(1, max_order):
        N *= 2
        h /= 2

        sum_new_points = sum(np.interp(a + k * h, x_values, y_values) for k in range(1, N, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_new_points

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]


# Example usage with empirical data
x_data = np.array([0, 0.3, 0.6, 1])  # Given x data
y_data = np.array([1.0, 0.85, 0.55, 0.2])  # Corresponding y data

integral_empirical = empirical_romberg_rule(y_data, x_data, max_order=5)
print("Approximated Integral (Empirical Data, max_order=5, Romberg's Rule):", integral_empirical)

