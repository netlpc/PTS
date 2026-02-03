# Integration Calculator (GUI)

A Python-based graphical integration calculator built with **Tkinter**, designed for educational and exploratory use in calculus and numerical analysis.

The application supports **symbolic and numerical integration**, **improper integrals**, **real-time plotting**, and a **multilingual user interface**, with a strong emphasis on **correct mathematical semantics** and **readable result presentation**.

---

## Features

### Symbolic Integration
- Indefinite integrals
- Definite integrals with exact (closed-form) results when available
- Automatic simplification using SymPy
- Graceful fallback to numerical evaluation when no closed-form exists
- Supports implicit multiplication (e.g. `2x`, `2pi`, `2(x+1)`)

### Numerical Integration
Supported numerical methods include:
- Rectangle Rule
- Trapezoidal Rule
- Simpson’s Rule
- Simpson 3/8 Rule
- Romberg Integration
- Gaussian Quadrature (Gauss–Legendre)
- Adaptive Simpson (`scipy.integrate.quad`)
- Monte Carlo Integration (stratified sampling)

For supported methods, **numerical error estimates** are computed and stored internally.

### Improper Integrals
- Supports infinite limits using `-inf` and `inf`
- Symbolic evaluation for convergence when possible
- Clear handling of unevaluated or divergent cases

---

## Result Display Policy

The calculator follows a unified and transparent result display strategy:

- Short, readable **exact symbolic results** are shown directly (e.g. `π^2 / 2`)
- Long or complex exact expressions are displayed as **numerical approximations** using the `≈` symbol
- If no closed-form result exists, numerical computation is used automatically
- Exact symbolic expressions are always preserved internally and can be accessed via **“View Exact Result”**

This design separates **computation precision** from **UI readability**.

---

## Graphical User Interface
- Built with Tkinter
- Embedded Matplotlib plotting
- Interactive navigation toolbar
- Progress bar for long-running numerical computations
- Scrollable calculation history

---

## History & Reproducibility
- History entries are stored as structured records (not plain text)
- Single-click: refill input fields from history
- Double-click: refill inputs and automatically recompute
- Enables reproducible and exploratory workflows

---

## Multilingual Interface

Supported languages:
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
