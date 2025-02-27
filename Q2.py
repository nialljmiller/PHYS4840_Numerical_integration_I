import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Integration Functions that take x and y data only -----

def trapezoidal_rule_data(x, y):
    """
    Approximates the integral using the trapezoidal rule from raw data.
    """
    integral = 0.0
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        integral += (y[i] + y[i+1]) * dx / 2
    return integral


def simpsons_rule_data(x, y):
    """
    Approximates the integral using Simpson's rule from raw data.
    Assumes that the x data is evenly spaced.
    If there's an odd number of intervals, the last interval is dropped.
    """
    N = len(x) - 1

    h = (x[-1] - x[0]) / N
    integral = y[0] + y[-1]
    for i in range(1, N):
        if i % 2 == 1:
            integral += 4 * y[i]
        else:
            integral += 2 * y[i]
    return integral * h / 3

def romberg_rule_data(x, y, max_order):
    """
    Approximates the integral using Romberg's method from raw data.
    Uses linear interpolation (np.interp) internally to evaluate the function
    at arbitrary points.
    """
    a = x[0]
    b = x[-1]
    R = np.zeros((max_order, max_order))
    # First trapezoidal approximation
    f_a = np.interp(a, x, y)
    f_b = np.interp(b, x, y)
    R[0, 0] = (b - a) * (f_a + f_b) / 2
    for i in range(1, max_order):
        N = 2**i  # Number of intervals doubles with each refinement
        h = (b - a) / N
        s = 0.0
        for k in range(1, N, 2):  # Sum only over the new midpoints
            xk = a + k * h
            s += np.interp(xk, x, y)
        R[i, 0] = 0.5 * R[i - 1, 0] + h * s
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
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



# ----- Load Data -----

# Read in CSV files (make sure these files are in your working directory)
gaia = pd.read_csv("GAIA_G.csv", header=None, names=["Wavelength", "Flux"])
vega = pd.read_csv("vega_SED.csv")

x_gaia = np.array(gaia["Wavelength"])
y_gaia = np.array(gaia["Flux"])

x_vega = np.array(vega["WAVELENGTH"])
y_vega = np.array(vega["FLUX"])

show_data(x_gaia, y_gaia,x_vega, y_vega) #hmm yes this looks bad

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

#We dont need to set a seperation as we have that from the data - its already binned. 
#Romberg still needs to know how long to compute for though 
max_order = 8

# GAIA data integration
trapz_gaia   = trapezoidal_rule_data(x_gaia, y_gaia)
simpson_gaia = simpsons_rule_data(x_gaia, y_gaia)
romberg_gaia = romberg_rule_data(x_gaia, y_gaia, max_order)

# Vega data integration
trapz_vega   = trapezoidal_rule_data(x_vega, y_vega)
simpson_vega = simpsons_rule_data(x_vega, y_vega)
romberg_vega = romberg_rule_data(x_vega, y_vega, max_order)


#print results
print("GAIA Data Integration:")
print("Trapezoidal Rule:   ", trapz_gaia)
print("Simpson's Rule:     ", simpson_gaia)
print("Romberg Integration:", romberg_gaia)

print("\nVega Data Integration:")
print("Trapezoidal Rule:   ", trapz_vega)
print("Simpson's Rule:     ", simpson_vega) # why is this one different?
print("Romberg Integration:", romberg_vega)



# The Vega data is unevenly spaced, this really impacts the simpson integration...
# so we interpolate it onto an evenly spaced grid.
x_vega_even = np.linspace(x_vega[0], x_vega[-1], len(x_vega)) #create a new evenly spaced X based on the min, max and length of previous x
y_vega_even = np.interp(x_vega_even, x_vega, y_vega) #interpolate y data to x data 

# Vega data integration again
trapz_vega   = trapezoidal_rule_data(x_vega_even, y_vega_even)
simpson_vega = simpsons_rule_data(x_vega_even, y_vega_even)
romberg_vega = romberg_rule_data(x_vega_even, y_vega_even, max_order)

print("\nVega Data Integration:")
print("Trapezoidal Rule:   ", trapz_vega)
print("Simpson's Rule:     ", simpson_vega)
print("Romberg Integration:", romberg_vega)
