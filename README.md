**Week 5 - Tuesday, February 18**  
**Numerical Integration I: Trapezoidal Rule, Simpson’s Rule, Error Analysis, Romberg Integration**  
**Sections: 5.1, 5.2, 5.3, 5.4 -- COMPUTATIONAL PHYSICS - Mark Newman**  

### **Introduction to Numerical Integration**  
Numerical integration is a fundamental technique for approximating definite integrals when analytical solutions are difficult or impossible to obtain. We will explore several methods, including:
- The **Trapezoidal Rule**
- **Simpson’s Rule**
- **Romberg Integration**
- **Error Analysis**

---
# Numerical Integration: Trapezoidal, Simpson's Rule and Romberg

## Understanding the Function Approximation

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

where **h** is the width of each subinterval. We approximate $f(x)$ as a straight line between consecutive points. (h is width here because this method `rotates' the trapezium).
...N seems important here...


## Area Approximation

Instead of integrating explicitly, we use the **trapezoidal formula** directly to approximate the area under $f(x)$ for each interval. The area of each individual trapezoid, denoted as $A_k$, is given by:

$$
A_k = \frac{1}{2} h \big[ f(a + h(k-1)) + f(a + hk) \big]
$$

This is because the area of a trapezium is:

$$
Area = \frac{1}{2} \text{ height } \big[\text{length of two parallel lines added together}\big]
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


```python

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

```

---

## Simpson’s Rule - Approximating with a Quadratic

The **Simpson’s Rule** method improves upon the **Trapezoidal Rule** by approximating the function using a **quadratic function** instead of a straight line. Instead of approximating \( f(x) \) between two points, we assume that over a small interval, the function behaves like a **parabola**.

---

## Quadratic Approximation Between Points
We approximate \( f(x) \) using a quadratic polynomial of the form:

$$
f(x) = Ax^2 + Bx + C
$$

Given three equally spaced points \( x = -h, 0, h \), we have:

$$
f(-h) = A h^2 - B h + C
$$

$$
f(0) = C
$$

$$
f(h) = A h^2 + B h + C
$$

Solving this system simultaneously for \( A, B, C \), we find:

$$
A = \frac{1}{2h^2}(f(h) - 2f(0) + f(-h))
$$

$$
B = \frac{1}{2h}(f(h) - f(-h))
$$

$$
C = f(0)
$$

Since integration of a quadratic function is straightforward, we integrate \( f(x) \) over the interval \( [-h, h] \):

$$
\int_{-h}^{h}(Ax^2 + Bx + C)dx = \frac{2}{3}Ah^3 + 2Ch
$$

Substituting \( A \) and \( C \):

$$
\int_{-h}^{h} f(x) \, dx = \frac{h}{3} \left[ f(-h) + 4f(0) + f(h) \right]
$$

which forms the basis of Simpson's Rule.

## Area Approximation Over an Interval

Extending this idea to an interval \([a, b]\), we divide it into \( n \) subintervals of equal width \( h \), where:

$$
h = \frac{b-a}{n}
$$

The points are:

$$
x_0 = a, \quad x_1 = a + h, \quad x_2 = a + 2h, \quad \dots, \quad x_n = b
$$

Applying the quadratic approximation iteratively across these subintervals, the integral is approximated as:

$$
\int_{a}^{b} f(x) \, dx \approx \frac{h}{3} \left[ f(a) + f(b) + 4 \sum_{\text{odd } k}^{1...N-1} f(a + kh) + 2 \sum_{\text{even } k}^{2...N-2} f(a+kh)\right]
$$

$$
\int_{a}^{b} f(x) \, dx \approx \frac{h}{3} \left[ f(a) + f(b) + 4 \sum_{\text{odd } k}^{1...N-1} f(x_N) + 2 \sum_{\text{even } k}^{2...N-2} f(x_N)\right]
$$


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

## **Romberg Integration - Refining the Trapezoidal Rule with Extrapolation**  

The **Romberg Integration** method builds upon the **Trapezoidal Rule** by applying **Richardson Extrapolation**, which systematically removes error terms to produce a more accurate result. Instead of computing a single approximation, Romberg Integration refines the integral estimation step by step using a sequence of trapezoidal approximations.

---

## **Error Reduction by Richardson Extrapolation**  

The error in the **Trapezoidal Rule** scales as:

$$
E_T = C h^2 + \mathcal{O}(h^4)
$$

where \( C \) is some constant and \( h \) is the step size. The key idea of **Romberg Integration** is to combine multiple **Trapezoidal Rule** estimates at different step sizes to systematically eliminate the leading error term.

To do this, we define:

$$
R_{m,0} = T_m
$$

where \( T_m \) is the **Trapezoidal Rule** approximation using \( 2^m \) intervals:

$$
T_m = \frac{h_m}{2} \left[ f(a) + f(b) + 2 \sum_{k=1}^{2^m-1} f(a + k h_m) \right]
$$

where the step size is:

$$
h_m = \frac{b-a}{2^m}
$$

We then apply Richardson Extrapolation recursively:

$$
R_{m, n} = \frac{4^n R_{m, n-1} - R_{m-1, n-1}}{4^n - 1}
$$

where \( $R_{m, n}$ \) is the improved estimate using the results of lower-order approximations.
The index nn in \( $R_{m, n}$ \) represents the level of extrapolation applied to refine the integral approximation.
\( $R_{m, n}$ \) is the n-th level extrapolated value derived from previous approximations to remove higher-order error terms.


This results in a **table of values**, where each row refines the previous row’s estimates.

---

## **Romberg Integration Table**  

The Romberg method fills in a triangular table as follows:

| \( $m$ \) | \( $R_{m,0}$ \) | \( $R_{m,1}$ \) | \( $R_{m,2}$ \) | \( $R_{m,3}$ \) | ... |
|---|---|---|---|---|---|
| 0 | \( $T_0$ \) | - | - | - | - |
| 1 | \( $T_1$ \) | \( $R_{1,1}$ \) | - | - | - |
| 2 | \( $T_2$ \) | \( $R_{2,1}$ \) | \( $R_{2,2}$ \) | - | - |
| 3 | \( $T_3$ \) | \( $R_{3,1}$ \) | \( $R_{3,2}$ \) | \( $R_{3,3}$ \) | - |
| ... | ... | ... | ... | ... | ... |

Each new column refines the previous estimates, improving the accuracy of the integral.

---

## **Final Approximation**  

The **best estimate** for the integral is found at the **bottom-right** of the table:

$$
I \approx R_{m, m}
$$

where \( R_{m, m} \) is the highest-order refinement available.

---

## **Romberg Integration Implementation in Python**  

The implementation follows the recursive structure described above, filling the **Romberg table** iteratively.

```python
import numpy as np

def romberg_integration(f, a, b, m):
    R = np.zeros((m+1, m+1))  
    for i in range(m+1):
        n = 2**i  
        R[i, 0] = trapezoidal_rule(f, a, b, n)  

        for j in range(1, i+1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    return R[m, m]
```

This function:  
- Computes the **Trapezoidal Rule** estimates for increasing numbers of intervals.
- Applies **Richardson Extrapolation** iteratively to refine the estimates.
- Returns the **most accurate** estimate at \( R_{m,m} \).

---

## **Comparison with Trapezoidal and Simpson’s Rule**  

- The **Trapezoidal Rule** is simple but has **\( $\mathcal{O}(h^2)$ \) error**.
- **Simpson’s Rule** improves upon it with **\( $\mathcal{O}(h^4)$ \) accuracy**, requiring an even number of subintervals.
- **Romberg Integration** systematically eliminates error terms, achieving **very high accuracy** with relatively few function evaluations.











---

### **3. Error Analysis in Numerical Integration**  
Each numerical integration method introduces some level of **error**, which depends on the function’s smoothness and the choice of subinterval width \( h \).  

The leading-order error for:  
- The **Trapezoidal Rule** is \( O(h^2) \)  
- **Simpson’s Rule** is \( O(h^4) \)  
- More sophisticated methods (e.g., **Romberg**) systematically eliminate these errors for even greater accuracy.  

---

### **Euler-Maclaurin Formula and Error Terms**  

The **Euler-Maclaurin formula** provides a way to express the error in numerical integration methods in terms of derivatives of the function.  

For the **Trapezoidal Rule**, applying Euler-Maclaurin leads to the **error term**:  

$$
E_T \approx -\frac{(b-a)h^2}{12} f''(\xi), \quad \xi \in [a, b]
$$  

This shows that the **error scales as \( O(h^2) \)**, meaning halving \( h \) reduces the error by a factor of **4**.  

For **Simpson’s Rule**, a similar derivation gives:  

$$
E_S \approx -\frac{(b-a)h^4}{180} f''''(\xi), \quad \xi \in [a, b]
$$  

which demonstrates the **\( O(h^4) \) scaling**—halving \( h \) now reduces the error by a factor of **16**.  

These results justify why **Simpson’s Rule is significantly more accurate than the Trapezoidal Rule** when the function is sufficiently smooth.  

---

### **Example: Comparing Errors Numerically**  

To illustrate the differences in error behavior, we approximate the integral:  

$$
I = \int_0^1 e^{-x^2} dx
$$  

which does not have a simple closed-form solution, but can be accurately estimated using high-precision numerical techniques.  

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return np.exp(-x**2)

# Compute "exact" integral using high-precision quadrature
exact, _ = quad(f, 0, 1)

# Define subintervals
ns = np.arange(2, 50, 2)

# Compute errors for Trapezoidal and Simpson’s Rule
errors_trapz = [abs(trapezoidal_rule(f, 0, 1, n) - exact) for n in ns]
errors_simpson = [abs(simpsons_rule(f, 0, 1, n) - exact) for n in ns]

# Plot error behavior on a log-log scale
plt.figure(figsize=(8, 5))
plt.loglog(ns, errors_trapz, label='Trapezoidal Error', marker='o')
plt.loglog(ns, errors_simpson, label='Simpson’s Error', marker='s')
plt.legend()
plt.xlabel('Number of Subintervals (n)')
plt.ylabel('Absolute Error')
plt.title('Error Comparison of Trapezoidal and Simpson’s Rule')
plt.grid(True)
plt.show()
```

---

### **Key Observations**  

1. **Trapezoidal Rule** exhibits an **\( O(h^2) \)** error decay, meaning a linear trend in a log-log plot with slope **-2**.  
2. **Simpson’s Rule** follows an **\( O(h^4) \)** error decay, meaning its error decreases much faster with increasing \( n \).  
3. **Romberg Integration** (not shown here) further refines the trapezoidal estimates, achieving even better accuracy.  



---



### **Next Class: Thursday, February 20**  
**Topic:** Numerical Integration II - Gauss-Legendre Quadrature  
**Sections:** 5.4, 5.5, 5.6, 5.7 (5.8, 5.9 TBD)


