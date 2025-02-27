import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


#a very small plotting function as i suspect something is up with the data...
def show_data(x_gaia, y_gaia,x_vega, y_vega):
    plt.figure(figsize=(10, 5))
    plt.plot(x_gaia, y_gaia, label="GAIA")
    plt.plot(x_vega, y_vega, label="Vega")
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("GAIA and Vega Data")
    plt.legend()
    plt.show()



# Load the CSV files into a pandas data frame
gaia = pd.read_csv("GAIA_G.csv", header=None, names=["Wavelength", "Flux"])
vega = pd.read_csv("vega_SED.csv")

# Split the data into lists
#This can be done in many different ways but for the sake of clarity (and NOT speed) I will do this
x_gaia = np.array(gaia["Wavelength"])
y_gaia = np.array(gaia["Flux"])

x_vega = np.array(vega["WAVELENGTH"])
y_vega = np.array(vega["FLUX"])

show_data(x_gaia, y_gaia,x_vega, y_vega) #hmm yes this looks bad

#We dont need to set a seperation as we have that from the data - its already binned. 
#Romberg still needs to know how long to compute for though 
max_order = 8

# GAIA data integration
trapz_gaia   = trapezoidal(x_gaia, y_gaia)
simpson_gaia = simpsons(x_gaia, y_gaia)
romberg_gaia = romberg(x_gaia, y_gaia, max_order)

# Vega data integration
trapz_vega   = trapezoidal(x_vega, y_vega)
simpson_vega = simpsons(x_vega, y_vega)
romberg_vega = romberg(x_vega, y_vega, max_order)


#print results
print("GAIA Data Integration:")
print("Trapezoidal Rule:   ", trapz_gaia)
print("Simpson's Rule:     ", simpson_gaia)
print("Romberg Integration:", romberg_gaia)

print("\nVega Data Integration:")
print("Trapezoidal Rule:   ", trapz_vega)
print("Simpson's Rule:     ", simpson_vega) # why is this one different?
print("Romberg Integration:", romberg_vega)
print("=============================\n")





#hmmm, that plot looked bad...
#We should remove the long tail from the Vega SED
#Romberg does not work well with things like this.
#See the final two paragraphs of page 162 in Mark Newmans book.
threshold_y = 0.2e-10    #at what point does the SED basically become 0 - this is relative to the scale of the SED, not just some small number
mask = y_vega > threshold_y #create a mask to identify where the data is below the value
x_vega = x_vega[mask] #new data = masked old data 
y_vega = y_vega[mask]


show_data(x_gaia, y_gaia,x_vega, y_vega) #Better, the Vega SED has a different scale to Gaia
#Thats fine as long as we dont run into floating point uncertainty... how small can we go?


# GAIA data integration
trapz_gaia   = trapezoidal(x_gaia, y_gaia)
simpson_gaia = simpsons(x_gaia, y_gaia)
romberg_gaia = romberg(x_gaia, y_gaia, max_order)

# Vega data integration
trapz_vega   = trapezoidal(x_vega, y_vega)
simpson_vega = simpsons(x_vega, y_vega)
romberg_vega = romberg(x_vega, y_vega, max_order)


#print results
print("\n After setting a lower limit on the Vega SED...")
print("GAIA Data Integration:")
print("Trapezoidal Rule:   ", trapz_gaia)
print("Simpson's Rule:     ", simpson_gaia)
print("Romberg Integration:", romberg_gaia)

print("\nVega Data Integration:")
print("Trapezoidal Rule:   ", trapz_vega)
print("Simpson's Rule:     ", simpson_vega) # why is this one different?
print("Romberg Integration:", romberg_vega)


