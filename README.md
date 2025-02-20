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

Numerical integration approximates the integral of a function by replacing the function with a simpler shape. The key idea is to approximate **$f(x)$** by a piecewise-defined function, which is either:

- A **linear function** (for the Trapezoidal Rule)
- A **quadratic function** (for Simpson’s Rule)

We then integrate these simple functions instead of the original **$f(x)$**. 

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

The **Simpson’s Rule** method improves upon the **Trapezoidal Rule** by approximating the function using a **quadratic function** instead of a straight line. Instead of approximating $$f(x)$$ between two points, we assume that over a small interval, the function behaves like a **parabola**.

---

## Quadratic Approximation Between Points
We approximate $$f(x)$$ using a quadratic polynomial of the form:

$$
f(x) = Ax^2 + Bx + C
$$

Given three equally spaced points $$x = -h, 0, h$$, we have:

$$
f(-h) = A h^2 - B h + C
$$

$$
f(0) = C
$$

$$
f(h) = A h^2 + B h + C
$$

Solving this system simultaneously for $A, B, C$, we find:

$$
A = \frac{1}{2h^2}(f(h) - 2f(0) + f(-h))
$$

$$
B = \frac{1}{2h}(f(h) - f(-h))
$$

$$
C = f(0)
$$

Since integration of a quadratic function is straightforward, we integrate $$f(x)$$ over the interval $$[-h, h]$$:

$$
\int_{-h}^{h}(Ax^2 + Bx + C)dx = \frac{2}{3}Ah^3 + 2Ch
$$

Substituting $A$ and $C$:

$$
\int_{-h}^{h} f(x) \, dx = \frac{h}{3} \left[ f(-h) + 4f(0) + f(h) \right]
$$

which forms the basis of Simpson's Rule.

## Area Approximation Over an Interval

Extending this idea to an interval \([a, b]\), we divide it into $n$ subintervals of equal width $h$, where:

$$
h = \frac{b-a}{n}
$$

The points are:

$$
x_0 = a, \quad x_1 = a + h, \quad x_2 = a + 2h, \quad x_2 = a + 3h, \quad \dots, \quad x_n = b
$$

Applying the quadratic approximation iteratively across these subintervals, the integral is approximated as:

$$
\int_{a}^{b} f(x) \, dx \approx \frac{h}{3} \left[ f(a) + f(b) + 4 \sum_{\text{odd } k}^{1...N-1} f(a + kh) + 2 \sum_{\text{even } k}^{2...N-2} f(a+kh)\right]
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
## **Romberg Integration: Refining the Trapezoidal Rule with Extrapolation**  

**Romberg Integration** improves the **Trapezoidal Rule** by applying **Richardson Extrapolation**, which systematically reduces error terms for greater accuracy. Instead of computing a single integral approximation, Romberg Integration refines estimates step by step using a sequence of trapezoidal approximations at progressively smaller step sizes.

---

### **Error Analysis and Richardson Extrapolation**  
If we approximate an integral using step size $$h$$, the result can be expressed as:

$$
A(h) = A + C h^p + O(h^{p+2})
$$

where:
- $$A$$ is the **true integral value**.
- $$C h^p$$ is the **leading error term**.
- $$p$$ is the **order of accuracy** (for the Trapezoidal Rule, $$p = 2$$).
- $$O(h^{p+2})$$ is all the smaller order terms.

If we refine the approximation with a smaller step size $$h/2$$, we get:

$$
A(h/2) = A + C (h/2)^p + O(h^{p+2})
$$

Subtracting these two equations:

$$
A(h/2) - A(h) = C h^p \left( \frac{1}{2^p} - 1 \right)
$$

Solving for $$A$$:

$$
A \approx A(h/2) + \frac{A(h/2) - A(h)}{2^p - 1}
$$

This process, known as **Richardson Extrapolation**, removes the leading error term, significantly improving accuracy.

For the **Trapezoidal Rule** (where $$p = 2$$), this simplifies to:

$$
A \approx \frac{4 A(h/2) - A(h)}{3}
$$

---

### **Romberg Integration Procedure**  
Romberg Integration iteratively applies the **Trapezoidal Rule** and Richardson Extrapolation to construct a table of increasingly accurate integral approximations.

$$
R_{m,0} = T_m
$$

where $$T_m$$ is the Trapezoidal Rule approximation using $$2^m$$ intervals:

$$
T_m = \frac{h_m}{2} \left[ f(a) + f(b) + 2 \sum_{k=1}^{2^m-1} f(a + k h_m) \right]
$$

with step size:

$$
h_m = \frac{b-a}{2^m}
$$

$$
R_{m, n} = \frac{4^n R_{m, n-1} - R_{m-1, n-1}}{4^n - 1}
$$


The Romberg method fills in a triangular table as follows:

| $$m$$ | $$R_{m,0}$$ | $$R_{m,1}$$ | $$R_{m,2}$$ | $$R_{m,3}$$ | ... |
|---|---|---|---|---|---|
| 0 | $$T_0$$ | - | - | - | - |
| 1 | $$T_1$$ | $$R_{1,1}$$ | - | - | - |
| 2 | $$T_2$$ | $$R_{2,1}$$ | $$R_{2,2}$$ | - | - |
| 3 | $$T_3$$ | $$R_{3,1}$$ | $$R_{3,2}$$ | $$R_{3,3}$$ | - |
| ... | ... | ... | ... | ... | ... |


---

The **best estimate** for the integral is found at the **bottom-right** of the table:

$$
I \approx R_{m, m}
$$

where $R_{m, m}$ is the highest-order refinement available.


```python

def romberg_integration(f, a, b, m):
    R = np.zeros((m+1, m+1))  
    for i in range(m+1):
        n = 2**i  
        R[i, 0] = trapezoidal_rule(f, a, b, n)  

        for j in range(1, i+1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    return R[m, m]
```



---

### **Error Analysis in Numerical Integration**  
Each numerical integration method introduces some level of **error**, which depends on the function’s smoothness and the choice of subinterval width $h$.  

The leading-order error for:  
- The **Trapezoidal Rule** is $O(h^2)$
- **Simpson’s Rule** is $O(h^4)$
- More sophisticated methods (e.g., **Romberg**) systematically eliminate these errors for even greater accuracy.  

---

### **Euler-Maclaurin Formula and Error Terms**  

The **Euler-Maclaurin formula** provides a way to express the error in numerical integration methods in terms of derivatives of the function.  

For the **Trapezoidal Rule**, applying Euler-Maclaurin leads to the **error term**:  

$$
E_T \approx -\frac{(b-a)h^2}{12} f''(\xi), \quad \xi \in [a, b]
$$  

This shows that the **error scales as $O(h^2)$**, meaning halving $h$ reduces the error by a factor of **4**.  

For **Simpson’s Rule**, a similar derivation gives:  

$$
E_S \approx -\frac{(b-a)h^4}{180} f''''(\xi), \quad \xi \in [a, b]
$$  

which demonstrates the **$O(h^4)$ scaling**—halving $h$ now reduces the error by a factor of **16**.  

These results justify why **Simpson’s Rule is significantly more accurate than the Trapezoidal Rule** when the function is sufficiently smooth.  

---






## **Gauss-Legendre Quadrature: Higher-Order Accuracy with Fewer Points**  

Gauss-Legendre quadrature is a powerful numerical integration technique that achieves high accuracy using a small number of points. Unlike the Trapezoidal Rule and Simpson’s Rule, which approximate the integral by evaluating the function at **equally spaced points**, Gauss quadrature optimally selects points (**nodes**) and weights to maximize accuracy for a given number of function evaluations.  

Instead of using equally spaced intervals, Gauss-Legendre quadrature approximates an integral:  

$$
I = \int_{-1}^{1} f(x) dx
$$

as a weighted sum of function values at specific points:

$$
I \approx \sum_{i=1}^{n} w_i f(x_i)
$$

where:
- $x_i$ are the **Legendre nodes** (roots of the Legendre polynomial $P_n(x)$ — i.e., values of $x$ where $P_n(x) = 0$).
- $w_i$ are the **quadrature weights**, chosen to maximize accuracy.  

For **$n$ points**, Gauss-Legendre quadrature is exact for all polynomials up to degree **$2n-1$**.

Unlike the Trapezoidal and Simpson’s Rules, which rely on equally spaced points, Gauss quadrature **optimally** chooses points such that it can integrate higher-degree polynomials exactly.  

This means **Gauss quadrature achieves higher accuracy with fewer function evaluations**, making it very efficient for **smooth** functions.

---

## **Gauss-Legendre Nodes and Weights**
The quadrature points **$x_i$** are the roots of the **Legendre polynomial $P_n(x)$**:

$$
P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} (x^2 - 1)^n
$$

The weights $w_i$ are computed using:

$$
w_i = \frac{2}{(1 - x_i^2) [P_n'(x_i)]^2}
$$


```python
legendre_roots, weights = np.polynomial.legendre.leggauss(n)
```

---

## **Gauss-Legendre Quadrature for General Intervals - Transform**  
Since the standard Gauss-Legendre quadrature is defined for $[-1,1]$, we need to **map** an integral over $[a, b]$ into this interval.  

If we want to approximate:

$$
I = \int_a^b f(x) dx
$$

we perform a **change of variables**:

$$
x = \frac{b-a}{2} t + \frac{a+b}{2}
$$

which transforms the integral into:

$$
I = \frac{b-a}{2} \int_{-1}^{1} f\left(\frac{b-a}{2} t + \frac{a+b}{2} \right) dt
$$

Applying Gauss quadrature:

$$
I \approx \frac{b-a}{2} \sum_{i=1}^{n} w_i f\left(\frac{b-a}{2} x_i + \frac{a+b}{2}\right)
$$

However, in the **specific case**, that we are integrating over **$[-1,1]$**, the transformation is not needed. The Gauss-Legendre Quadrature simplifies to:

$$
I \approx \sum_{i=1}^{n} w_i f(x_i)
$$

---

## **Gauss-Legendre Quadrature Code Implementation**


### **Python Code**
```python
#!/usr/bin/env python3
import numpy as np

legendre_roots, weights = np.polynomial.legendre.leggauss(n)
for i in range(n):
    point = legendre_roots[i]
    weight = weights[i]
    function_value = f(point)  # Evaluate function at the Gauss point
    weighted_value = weight * function_value  # Apply weight
    integral_approx += weighted_value  # Add to running sum
```

---

## **Comparison with Other Methods**

| Method | Error Order $O(h^p)$ | Number of Function Evaluations (for n=4) | Accuracy |
|---|---|---|---|
| **Trapezoidal Rule** | $O(h^2)$ | 5 | Low |
| **Simpson’s Rule** | $O(h^4)$ | 5 | Moderate |
| **Romberg Integration** | $O(h^{2m})$ | Varies | High |
| **Gauss-Legendre Quadrature (n=4)** | Exact for $O(x^7)$ | 4 | Very High |

---

### **Advantages of Gauss-Legendre Quadrature**
- **Higher accuracy** for smooth functions.
- **Requires fewer function evaluations** than Simpson’s Rule or Trapezoidal Rule.
- **Exact for polynomials up to $2n-1$**.
- **Scales efficiently with increasing $n$**.

### **Limitations**
- More computationally expensive to determine nodes and weights.
- Less effective for functions with singularities or discontinuities.
- Not as intuitive as the Trapezoidal Rule for basic understanding.

---
