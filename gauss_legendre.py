import numpy as np

def f(x):
    return x**2  # Some function thats easy to integrate by hand and hence verify

# Number of points (n) for Gauss-Legendre Quadrature
n = 

# Compute the Gauss-Legendre Quadrature points (roots of the Legendre polynomial) and weights


#print the roots and weights for the points


# Compute the integral approximation manually using a for loop
#iterating through each legendre polynomial
    #grab the root for this polynomial
    #grap the weight for this polynomial
    #Evaluate function at the root
    # Apply weight
    # append to running sum




# Print final comparison
print("\nFinal Results:")
print(f"Approximated integral using Gauss-Legendre Quadrature: {integral_approx}")
print(f"Exact integral: {exact_integral}")
print(f"Error: {abs(integral_approx - exact_integral)}")
