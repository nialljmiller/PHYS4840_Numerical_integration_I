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


def function(x):
    return np.exp(-x**2)

a = 0  # Integration bounds
b = 1  # Integration bounds
N = # Number of trapezoids

integral_approx = romberg_rule(function, a, b, N)
print(f"Approximated Integral with N={N}: {integral_approx}")

#Hint: the integral of e^(-x**2) between 0 and 1 is 0.746824132812427

