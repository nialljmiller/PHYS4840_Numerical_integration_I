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
# Numerical Integration: Trapezoidal and Simpson's Rule

## Step 1: Understanding the Function Approximation

Numerical integration approximates the integral of a function by replacing the function with a simpler shape. The key idea is to approximate $**f(x)**$ by a piecewise-defined function, which is either:

- A **linear function** (for the Trapezoidal Rule)
- A **quadratic function** (for Simpson’s Rule)

We then integrate these simple functions instead of the original $**f(x)**$. 

---

## Step 2: Trapezoidal Rule - Approximating with a Line

The **Trapezoidal Rule** assumes that between each pair of points, the function $**f(x)**$ behaves like a straight line. The equation of a straight line is given by:

$$
y = mx + c
$$

For two points $x_i, f(x_i)$ and $\((x_{i+1}, f(x_{i+1}))\)$, the slope $\( m \)$ is:

$$
m = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i}
$$

...and...

$$
c = f(x_i)
$$

because $f(x_i)$ is the intercept in our case (the point where y is drived from the $0^{th}$ x - $x_i$, not necciserily 0 but who cares...

Thus, the equation of the line is:

$$
y = mx + c\\
L(x) = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i} (x - x_i) + f(x_i)
$$

To approximate the integral, we integrate this linear function between $x_i$ and $x_{i+1}$:

$$
\int_{x_i}^{x_{i+1}} L(x) dx = \frac{h}{2} ( f(x_i) + f(x_{i+1}) )
$$

where $h = x_{i+1} - x_i$. Summing over all intervals from $a$ to $b$:

$$
\int_a^b f(x) dx \approx \sum_{i=0}^{n-1} \frac{h}{2} ( f(x_i) + f(x_{i+1}) )
$$

Rewriting in a simpler form:

$$
\int_a^b f(x) dx \approx \frac{h}{2} ( f(x_0) + 2 \sum f(x_i) + f(x_n) )
$$

This shows that the Trapezoidal Rule replaces **f(x)** with a piecewise **linear function**.

---

## Step 3: Simpson’s Rule - Approximating with a Quadratic

Simpson’s Rule assumes that between three consecutive points, **f(x)** behaves like a quadratic function. The general quadratic equation is:

$$
y = ax^2 + bx + c
$$

Given three points \((x_i, f(x_i))\), \((x_{i+1}, f(x_{i+1}))\), and \((x_{i+2}, f(x_{i+2}))\), we fit a parabola through these points.

We then integrate this quadratic approximation over small intervals and sum them up to approximate the total integral. The result is:

$$
\int_a^b f(x) dx \approx \frac{h}{3} \sum ( f(x_i) + 4 f(x_{i+1}) + f(x_{i+2}) )
$$

where \(h\) is the step size. This highlights that Simpson’s Rule replaces **f(x)** with a piecewise **quadratic function**.

---

## Summary of Differences

| Method | Approximation | Formula |
|--------|--------------|---------|
| **Trapezoidal Rule** | Uses a linear approximation | \( \frac{h}{2} ( f(x_0) + 2 \sum f(x_i) + f(x_n) ) \) |
| **Simpson’s Rule** | Uses a quadratic approximation | \( \frac{h}{3} \sum ( f(x_i) + 4 f(x_{i+1}) + f(x_{i+2}) ) \) |

---

## Step 4: Python Implementations

### **Trapezoidal Rule Implementation**
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

### **Simpson’s Rule Implementation**
```python

def simpsons_rule(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's Rule")
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h/3) * (y[0] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n-1:2]) + y[-1])
```

This explanation makes it clear that the difference between the methods is **how we approximate the function**: a line for Trapezoidal and a parabola for Simpson’s.


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


