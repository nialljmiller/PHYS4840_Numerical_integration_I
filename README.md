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

# Trapezoidal Rule - Approximating with a Line

The **Trapezoidal Rule** approximates the area under a curve by dividing it into **N** trapezoidal slices of equal width. Given a function $f(x)$ over the interval $[a, b]$, we define:

$$
h = \frac{b - a}{N}
$$

where **h** is the width of each subinterval. We approximate $f(x)$ as a straight line between consecutive points. (h is width here because this method rotates the trapezium.
...N seems important here...

## Linear Approximation Between Points

The equation of a straight line between two points $(x_k, f(x_k))$ and $(x_{k+1}, f(x_{k+1}))$ is given by:

$$
 f(x) = m x + c
$$

where the slope **m** is:

$$
m = \frac{f(x_{k+1}) - f(x_k)}{x_{k+1} - x_k}
$$

and the intercept would be:

$$
c = f(x_k)
$$

as this is the y value at the initial x - $x_k$

Since we define the function at $x_k$ to be the starting point of the interval, the equation simplifies to:

$$
G(x) = \frac{f(x_{k+1}) - f(x_k)}{h} (x - x_k) + f(x_k)
$$


## Area Approximation

Instead of integrating explicitly, we use the **trapezoidal formula** directly to approximate the area under $f(x)$ for each interval. The area of each individual trapezoid, denoted as $A_k$, is given by:

$$
A_k = \frac{1}{2} h \big[ f(a + h(k-1)) + f(a + hk) \big]
$$

This is because the area of a trapezium is:

$$
Area = \frac{1}{2} height \big[width of two paralell lines added togther]
$$

Summing over all subintervals from $k = 1$ to $N$:

$$
A \approx  \frac{1}{2} h  \sum_{k=1}^{N}\big[ f(a + h(k-1)) + f(a + hk) \big]
$$

Rewriting in a more compact form:

$$
A \approx \frac{h}{2} \Bigg[ f(a) + f(b) + 2 \sum_{k=1}^{N-1} f(a + hk) \Bigg]
$$

...or as Mark prefers it:

$$
A \approx h \Bigg[ \frac{1}{2}f(a) + \frac{1}{2}f(b) + \sum_{k=1}^{N-1} f(a + hk) \Bigg]
$$

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


