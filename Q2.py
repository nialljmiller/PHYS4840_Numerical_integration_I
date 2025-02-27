#!/usr/bin/python3.8
#####################################
#
# Homework Problem 2 walkthrough
# Author: Niall Miller
# modified by M Joyce
#
#####################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import sys
# sys.path.append() ##... as needed
import my_functions_lib as mfl




# Load the CSV files into a pandas data frame
gaia = pd.read_csv("GAIA_G.csv", header=None, names=["Wavelength", "Flux"])
vega = pd.read_csv("vega_SED.csv")

# Split the data into lists
#This can be done in many different ways but for the sake of clarity (and NOT speed) I will do this
x_gaia = np.array(gaia["Wavelength"])
y_gaia = np.array(gaia["Flux"])

x_vega = np.array(vega["WAVELENGTH"])
y_vega = np.array(vega["FLUX"])

mfl.show_data(x_gaia, y_gaia,x_vega, y_vega, output_png = 'explore_data.png') #hmm yes this looks bad

#We dont need to set a seperation as we have that from the data - its already binned. 
#Romberg still needs to know how long to compute for though 
max_order = 8

# GAIA data integration
trapz_gaia   = mfl.trapezoidal(x_gaia, y_gaia)
simpson_gaia = mfl.simpsons(x_gaia, y_gaia)
romberg_gaia = mfl.romberg(x_gaia, y_gaia, max_order)

# Vega data integration
trapz_vega   = mfl.trapezoidal(x_vega, y_vega)
simpson_vega = mfl.simpsons(x_vega, y_vega)
romberg_vega = mfl.romberg(x_vega, y_vega, max_order)


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

threshold_y = 0.2e-10    #at what point does the SED basically become 0?
## this is relative to the scale of the SED, not just some small number

mask = np.where(y_vega > threshold_y)
#mask = y_vega > threshold_y     #create a mask to identify where the data is below the value
x_vega = x_vega[mask]           #new data = masked old data 
y_vega = y_vega[mask]


mfl.show_data(x_gaia, y_gaia,x_vega, y_vega, output_png = 'result.png') #Better, the Vega SED has a different scale to Gaia
#Thats fine as long as we dont run into floating point uncertainty... how small can we go?


# GAIA data integration
trapz_gaia   = mfl.trapezoidal(x_gaia, y_gaia)
simpson_gaia = mfl.simpsons(x_gaia, y_gaia)
romberg_gaia = mfl.romberg(x_gaia, y_gaia, max_order)

# Vega data integration
trapz_vega   = mfl.trapezoidal(x_vega, y_vega)
simpson_vega = mfl.simpsons(x_vega, y_vega)
romberg_vega = mfl.romberg(x_vega, y_vega, max_order)


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
