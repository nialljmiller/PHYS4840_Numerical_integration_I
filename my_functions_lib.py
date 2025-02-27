#!/usr/bin/python3.8
#########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Integration Functions that take x and y data only -----

def trapezoidal(x, y):
    integral = 0.0
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        integral += (y[i] + y[i+1]) * dx / 2
    return integral


def simpsons(x, y):

    N = len(x) - 1

    h = (x[-1] - x[0]) / N
    integral = y[0] + y[-1]
    for i in range(1, N):
        if i % 2 == 1:
            integral += 4 * y[i]
        else:
            integral += 2 * y[i]
    return integral * h / 3


def trapezoidal(x_values, y_values):
    """
    Approximates the integral using trapezoidal rule for given y_values at given x_values.
    
    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals.

    Returns:
        float: The approximated integral.
    """
    
    N = len(x_values) - 1
    a = x_values[0]
    b = x_values[-1]
    h = (b - a) / N  # Properly calculating h

    integral = (1/2) * (y_values[0] + y_values[-1]) * h  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk * h

    return integral


def simpsons(x_values, y_values):
    """
    Approximates the integral using Simpson's rule for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """

    N = len(x_values) - 1
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



def romberg(x_values, y_values, max_order):
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

################################
#
# a function for plotting data
#
#################################


# plots can be functions too!
def show_data(x_gaia, y_gaia,x_vega, y_vega, **kwargs):

    savename = kwargs.get('output_png', 'test.png')

    plt.figure(figsize=(10, 5))
    plt.plot(x_gaia, y_gaia, label="GAIA")
    plt.plot(x_vega, y_vega, label="Vega")
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("GAIA and Vega Data")
    plt.legend()
   #plt.show()
    plt.savefig(savename)
    plt.close()

    return 