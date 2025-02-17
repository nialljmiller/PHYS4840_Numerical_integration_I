import numpy as np

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

# True integral value
true_value = 0.26424111765711535680895245967707826510837773793646433098432639660507700851

# Compute errors
trap_error = abs(trap_result - true_value)
simp_error = abs(simp_result - true_value)
romb_error = abs(romb_result - true_value)

# Print results with error analysis
print("\nIntegration Method Comparison")
print("=" * 50)
print(f"{'Method':<20}{'Result':<20}{'Error':<20}{'Time (sec)':<10}")
print("-" * 50)
print(f"{'Trapezoidal Rule':<20}{trap_result:<20.8f}{trap_error:<20.8e}{trap_time:<10.6f}")
print(f"{'Simpson\'s Rule':<20}{simp_result:<20.8f}{simp_error:<20.8e}{simp_time:<10.6f}")
print(f"{'Romberg Integration':<20}{romb_result:<20.8f}{romb_error:<20.8e}{romb_time:<10.6f}")
print("=" * 50)
