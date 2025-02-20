from numpy.polynomial.legendre import leggauss as legendre_thingy

legendre_roots, weights = legendre_thingy(3)
print(legendre_roots, weights)