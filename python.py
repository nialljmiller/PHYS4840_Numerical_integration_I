from numpy.polynomial.legendre import leggauss as legendre_thingy

legendre_roots, weights = legendre_thingy(3)

print("Gauss-Legendre Quadrature Points (Roots of P_n(x)):")
print(legendre_roots)

print("\nWeights for each point:")
print(weights)