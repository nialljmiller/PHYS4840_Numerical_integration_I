**Week 5 - Tuesday, February 18**  
**Numerical Integration I: Trapezoidal Rule, Simpson’s Rule, Error Analysis, Romberg Integration**  
**Sections: 5.1, 5.2, 5.3, 5.4**  

### **Introduction to Numerical Integration**  
Numerical integration is a fundamental technique for approximating definite integrals when analytical solutions are difficult or impossible to obtain. We will explore several methods, including:
- The **Trapezoidal Rule**
- **Simpson’s Rule**
- **Error Analysis**
- **Romberg Integration**

---

# Numerical Integration: Trapezoidal Rule

## Step 1: Area of a Single Trapezoid

A single trapezoid between two points \( x_i \) and \( x_{i+1} \) on the function \( f(x) \) has an area given by:

\[
A = \frac{1}{2} \times \text{Base} \times (\text{Height}_1 + \text{Height}_2)
\]

where:

- The **base** is the distance between \( x_i \) and \( x_{i+1} \), which is:

  \[
  h = x_{i+1} - x_i
  \]

- The **heights** are the function values at these points: \( f(x_i) \) and \( f(x_{i+1}) \).

Thus, the area of a single trapezoid is:

\[
A_i = \frac{h}{2} \left( f(x_i) + f(x_{i+1}) \right)
\]

---

## Step 2: Summing Over Multiple Trapezoids

To approximate the integral over \( [a,b] \), we sum up the areas of all \( n \) trapezoids:

\[
\int_a^b f(x) dx \approx \sum_{i=0}^{n-1} A_i
\]

Substituting the formula for \( A_i \):

\[
\int_a^b f(x) dx \approx \sum_{i=0}^{n-1} \frac{h}{2} \left( f(x_i) + f(x_{i+1}) \right)
\]

which is exactly the formula used in numerical integration.

---

## Step 3: Compact Summation Notation

Rearranging the sum, we recognize that all interior function values appear twice (once as \( f(x_{i+1}) \) and once as \( f(x_i) \) in the next term), except for the first and last points:

\[
\int_a^b f(x) dx \approx \frac{h}{2} \left( f(x_0) + 2 \sum_{i=1}^{n-1} f(x_i) + f(x_n) \right)
\]

This is the form commonly used in textbooks and numerical integration libraries.

---

## Summary of Forms

1. **Basic Trapezoid Formula:**
   \[
   A = \frac{1}{2} h (f(a) + f(b))
   \]

2. **Sum Over Multiple Trapezoids:**
   \[
   \sum_{i=0}^{n-1} \frac{h}{2} \left( f(x_i) + f(x_{i+1}) \right)
   \]

3. **Compact Form with Summation:**
   \[
   \frac{h}{2} \left( f(x_0) + 2 \sum_{i=1}^{n-1} f(x_i) + f(x_n) \right)
   \]

---

## Python Implementation

```python
import numpy as np

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# Example usage:
f = lambda x: np.exp(-x**2)
a, b, n = 0, 1, 100
integral_approx = trapezoidal_rule(f, a, b, n)
print("Approximated Integral:", integral_approx)
```

---

### **2. Simpson’s Rule**  
Simpson’s Rule improves upon the Trapezoidal Rule by using quadratic interpolation instead of linear.

#### **Formula:**  
\[
\int_a^b f(x) dx \approx \frac{h}{3} \sum_{i=0, \text{even}}^{n-2} \left[f(x_i) + 4f(x_{i+1}) + f(x_{i+2})\right]
\]
where \( h = \frac{b-a}{n} \) and \( n \) must be even.

#### **Python Implementation:**
```python
def simpsons_rule(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's Rule")
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h/3) * (y[0] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n-1:2]) + y[-1])
```

---

### **3. Error Analysis in Numerical Integration**  
Each method introduces some level of error. The error for:
- The **Trapezoidal Rule** is \( O(h^2) \)
- **Simpson’s Rule** is \( O(h^4) \)
- More sophisticated methods (e.g., Romberg) further improve accuracy.

#### **Example: Comparing Errors**
We can compare errors numerically using an example function \( f(x) = e^{-x^2} \).
```python
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x**2)

exact = 0.746824  # Approximate exact integral for comparison
ns = np.arange(2, 50, 2)
errors_trapz = [abs(trapezoidal_rule(f, 0, 1, n) - exact) for n in ns]
errors_simpson = [abs(simpsons_rule(f, 0, 1, n) - exact) for n in ns]

plt.loglog(ns, errors_trapz, label='Trapezoidal Error')
plt.loglog(ns, errors_simpson, label='Simpson’s Error')
plt.legend()
plt.xlabel('Number of Subintervals (n)')
plt.ylabel('Absolute Error')
plt.show()
```

---

### **4. Romberg Integration**  
Romberg Integration extends the Trapezoidal Rule by using Richardson Extrapolation to improve accuracy.

#### **Algorithm Outline:**  
1. Compute trapezoidal approximations for various step sizes.
2. Use extrapolation to estimate the integral with higher accuracy.

#### **Python Implementation:**
```python
from scipy.integrate import romberg

result = romberg(f, 0, 1)
print("Romberg Integration Result:", result)
```

---

### **Summary**
- **Trapezoidal Rule**: First-order accurate, simple, but not highly precise.
- **Simpson’s Rule**: More accurate than Trapezoidal, but requires an even number of intervals.
- **Error Analysis**: Shows how methods compare in convergence.
- **Romberg Integration**: Uses refinement to achieve higher accuracy.

---

### **Next Class: Thursday, February 20**  
**Topic:** Numerical Integration II - Gauss-Legendre Quadrature  
**Sections:** 5.4, 5.5, 5.6, 5.7 (5.8, 5.9 TBD)


