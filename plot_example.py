import matplotlib.pyplot as plt

#plotting example.Sshowing how we can make a list, populate it and plot it

def function(x):
	#a function that does stuff
    return x/2, x**2

# Initialise lists
array_1 = []
array_2 = []

for N in [5, 100, 1000, 10000]:
    array_1.append(N)
    a, b = function(N)
    array_2.append(b)

plt.scatter(array_1, array_2, color='blue', marker='x', s=80, label='Data')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('N values (log scale)')
plt.ylabel('Function Output (N^2) (log scale)')
plt.title('Log-Log Scatter Plot')
plt.legend()

plt.show()
