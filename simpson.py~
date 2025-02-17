import numpy as np

def simpsons_rule(f, a, b, N):
    """
    Approximates the integral using Simpson's rule.

    Parameters:
        f (function): The function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """
    if N % 2 == 1:
        raise ValueError("N must be even for Simpson's rule.")

    h = (b - a) / N  # Step size
    integral = f(a) + f(b)  # First and last terms

    # Loop through k=1 to N-1
    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        integral += 4 * f(xk)

    for k in range(2, N-1, 2):  # Even indices (weight 2)
        xk = a + k * h
        integral += 2 * f(xk)

    return (h / 3) * integral  # Final scaling


# Example usage with a function
f = lambda x: np.exp(-x**2)  # Function to integrate
a, b = 0, 1  # Integration bounds
N = 50  # Number of intervals (must be even)

integral_approx = simpsons_rule(f, a, b, N)
print("Approximated Integral with Simpson's Rule (N=50):", integral_approx)


# Simpson's rule for empirical data
def empirical_simpsons_rule(y_values, x_values, N):
    """
    Approximates the integral using Simpson's rule for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """
    if N % 2 == 1:
        raise ValueError("N must be even for Simpson's rule.")

    a, b = x_values[0], x_values[-1]
    h = (b - a) / N

    integral = y_values[0] + y_values[-1]  # First and last terms

    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 4 * yk

    for k in range(2, N-1, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    return (h / 3) * integral  # Final scaling


# Example usage with empirical data
x_data = np.array([0, 0.3, 0.6, 1])  # Given x data
y_data = np.array([1.0, 0.85, 0.55, 0.2])  # Corresponding y data

integral_empirical = empirical_simpsons_rule(y_data, x_data, N=10)  # Use N=10
print("Approximated Integral (Empirical Data, N=10, Simpson's Rule):", integral_empirical)

