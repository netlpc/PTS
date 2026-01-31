# Integration Calculator (GUI)

This project is a Python-based graphical integration calculator built with Tkinter.  
It supports symbolic integration, multiple numerical integration methods, improper integrals, real-time function plotting, and a multilingual user interface.

The application is designed for educational and exploratory use in calculus and numerical analysis.

---

## Features

### Symbolic Integration
- Indefinite integrals
- Definite integrals with exact results
- Automatic simplification using SymPy
- Supports implicit multiplication (e.g. `2x`, `2pi`, `2(x+1)`)

### Numerical Integration Methods
- Rectangle Rule
- Trapezoidal Rule
- Simpson’s Rule
- Simpson 3/8 Rule
- Romberg Integration
- Gaussian Quadrature (Gauss–Legendre)
- Adaptive Simpson (via `scipy.integrate.quad`)
- Monte Carlo Integration (stratified sampling)

### Improper Integrals
- Supports infinite limits using `-inf` and `inf`
- Symbolic evaluation for convergence

### Graphical User Interface
- Built with Tkinter
- Embedded Matplotlib plotting
- Interactive navigation toolbar
- Progress bar for long numerical computations
- Scrollable calculation history

### Multilingual Interface
Supported languages include:
- English
- Simplified Chinese
- Traditional Chinese
- Japanese
- Korean
- Spanish
- French
- Arabic
- Hindi

---

## Supported Mathematical Input

### Functions and Constants
- Trigonometric: `sin(x)`, `cos(x)`, `tan(x)`
- Logarithmic: `ln(x)`, `log(x)`, `log10(x)`
- Exponential: `exp(x)`, `e^x`
- Roots and powers: `sqrt(x)`, `x^n`
- Constants: `pi`, `e`
- Infinity: `inf`, `-inf`

### Implicit Multiplication
The parser supports implicit multiplication using SymPy’s official parser:
- `2x` → `2*x`
- `2pi` → `2*pi`
- `2(x+1)` → `2*(x+1)`

---

## Requirements

- Python 3.9 or later

### Python Dependencies

Install required packages using pip:

```bash
pip install numpy sympy scipy matplotlib
