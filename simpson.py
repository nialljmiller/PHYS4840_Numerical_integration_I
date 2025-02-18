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

    h = # Step size
    integral = f(a) + f(b)  # First and last terms
    
    # Loop through k=1 to N-1
    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        integral += 4 * f(xk)

    for k in range(2, N-1, 2):  # Even indices (weight 2)
        xk = a + k * h
        integral += 2 * f(xk)

    return (h / 3) * integral  # Final scaling


def function(x):
    return np.exp(-x**2)

a = 0  # Integration bounds
b = 1  # Integration bounds
N = # Number of sections (what happens if this is odd, why?)

integral_approx = simpsons_rule(function, a, b, N)
print(f"Approximated Integral with N={N}: {integral_approx}")

#Hint: the integral of e^(-x**2) between 0 and 1 is 0.746824132812427

