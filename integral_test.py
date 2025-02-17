import numpy as np
import time

# Example usage with empirical data (handling precomputed values directly)
def trapezoidal(y_values, x_values, N):
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
    h = (b - a) / N  # Properly calculating h

    integral = (1/2) * (y_values[0] + y_values[-1]) * h  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk * h

    return integral


# Simpson's rule for empirical data
def simpsons(y_values, x_values, N):
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

    for k in range(2, N, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    return (h / 3) * integral  # Final scaling


# Romberg integration for empirical data
def romberg(y_values, x_values, max_order):
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


def timing_function(integration_method, x_values, y_values, steps=10, *args):
    """
    Times the execution of an integration method.

    Parameters:
        integration_method (function): The numerical integration function.
        x_values (array-like): The x values.
        y_values (array-like): The corresponding y values.
        steps (int, optional): Number of intervals to use.
        *args: Additional arguments for the integration method.

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.time()
    result = integration_method(y_values, x_values, steps, *args)
    end_time = time.time()
    
    return end_time - start_time, result


# Define function and generate sample data
f = lambda x: x * np.exp(-x)

x_data = np.linspace(0, 1, 1000000)  # Generate x values
y_data = f(x_data)  # Compute corresponding y values

# Testing the integration methods with N=10 (adjustable)
N = 10  # Must be even for Simpson's method
max_order = 5  # Romberg's accuracy level

# Compute integrals
trap_result = trapezoidal(y_data, x_data, N)
simp_result = simpsons(y_data, x_data, N)
romb_result = romberg(y_data, x_data, max_order)

# Measure timing
trap_time, _ = timing_function(trapezoidal, x_data, y_data, N)
simp_time, _ = timing_function(simpsons, x_data, y_data, N)
romb_time, _ = timing_function(romberg, x_data, y_data, max_order)

# Print results
answer = 0.26424111765711535680895245967707826510837773793646433098432639660507700851

print(f"Trapezoidal Rule: {trap_result:.6f}, Time: {trap_time:.6f} sec")
print(f"Simpson's Rule: {simp_result:.6f}, Time: {simp_time:.6f} sec")
print(f"Romberg Integration: {romb_result:.6f}, Time: {romb_time:.6f} sec")

