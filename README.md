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

### **1. The Trapezoidal Rule**  
The Trapezoidal Rule approximates the integral of a function \( f(x) \) over an interval \([a, b]\) by dividing it into \(n\) subintervals and treating each segment as a trapezoid.

#### **Formula:**  
\[
\int_a^b f(x) dx \approx \sum_{i=0}^{n-1} \frac{h}{2} \left[ f(x_i) + f(x_{i+1}) \right]
\]
where \( h = \frac{b - a}{n} \) is the step size.

#### **Python Implementation:**
```python
import numpy as np

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
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
