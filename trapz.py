import numpy as np

def trapezoidal_rule(f, a, b, N):
    """
    Approximates the integral using the trapezoidal rule with a loop.

    Parameters:
        f (function or array-like): If a function, it's evaluated at N+1 points.
                                    If an array, it's treated as precomputed values.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (trapezoids).

    Returns:
        float: The approximated integral.
    """
    # Step size
    h = (b - a) / N

    integral = (1/2) * (f(a) + f(b)) * h  # Matches the first & last term in the sum

    # Loop through k=1 to N-1 to sum the middle terms
    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly (matches the formula)
        integral += f(xk) * h  # Normal weight (multiplied by h directly)

    return integral


# Example usage with a function (matching your derivation)
f = lambda x: np.exp(-x**2)  # Function to integrate
a, b = 0, 1  # Integration bounds
N = 50  # Number of trapezoids

integral_approx = trapezoidal_rule(f, a, b, N)
print("Approximated Integral with N=50:", integral_approx)





# Example usage with empirical data (handling precomputed values directly)
def empirical_trapezoidal_rule(y_values, x_values, N):
    """
    Approximates the integral using trapezoidal rule for given y_values at given x_values.
    
    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals.

    Returns:
        float: The approximated integral.
    """
    a = x_values[0]
    b = x_values[-1]
    h = x_values[1] - x_values[0]

    integral = (1/2) * (y_values[0] + y_values[-1]) * h  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk * h

    return integral


# Example usage with empirical data
x_data = np.array([0, 0.3, 0.6, 1])  # Given x data
y_data = np.array([1.0, 0.85, 0.55, 0.2])  # Corresponding y data

integral_empirical = empirical_trapezoidal_rule(y_data, x_data, N=10)  # Use N=10
print("Approximated Integral (Empirical Data, N=10):", integral_empirical)

