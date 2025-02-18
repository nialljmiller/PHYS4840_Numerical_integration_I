import numpy as np


def trapezoidal_rule(f, a, b, N):
    """
    Approximates the integral using the trapezoidal rule with a loop.

    Parameters:
        f (function): The function, evaluated at N+1 points.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (trapezoids).

    Returns:
        float: The approximated integral.
    """
    h = (b - a) / N  # Step size
    integral = (1/2) * (f(a) + f(b))  # First and last term in sum

    # Sum the middle terms
    for k in range(1, N):
        xk = a + k * h
        integral += f(xk)
    
    return integral * h  # Multiply by step size



def romberg_rule(f, a, b, max_order):
    """
    Approximates the integral using Romberg's method, leveraging the trapezoidal rule.

    Parameters:
        f (function): The function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))  # Create a Romberg table
    
    # First approximation using the trapezoidal rule
    R[0, 0] = trapezoidal_rule(f, a, b, 1)
    
    for i in range(1, max_order):
        N = 2**i  # Number of intervals (doubles each step)
        R[i, 0] = trapezoidal_rule(f, a, b, N)
        
        # Compute extrapolated Romberg values
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
    
    return R[max_order - 1, max_order - 1]  # Return the most refined estimate




def function(x):
    return np.exp(-x**2)

a = 0  # Integration bounds
b = 1  # Integration bounds
max_order =# Number of refinements

integral_approx = romberg_rule(function, a, b, max_order)
print(f"Approximated Integral with maximum order={max_order}: {integral_approx}")

#Hint: the integral of e^(-x**2) between 0 and 1 is 0.746824132812427

