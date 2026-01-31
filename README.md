# 🧮 Integration Calculator (GUI)

A Python-based graphical integration calculator built with **Tkinter**, supporting:

- Symbolic integration (via SymPy)
- Multiple numerical integration methods
- Improper integrals (±∞)
- Real-time function plotting
- Multi-language UI (English / 中文 / 日本語 / 한국어)

---

## ✨ Features

- **Symbolic Integration**
  - Indefinite & definite integrals
  - Exact results (fractions, π, etc.)

- **Numerical Integration Methods**
  - Trapezoidal Rule
  - Simpson’s Rule
  - Simpson 3/8 Rule
  - Romberg Integration
  - Gaussian Quadrature (Gauss–Legendre)
  - Adaptive Simpson (via `scipy.integrate.quad`)
  - Monte Carlo Integration (stratified sampling)

- **Improper Integrals**
  - Supports `-inf` and `inf`

- **GUI Features**
  - Embedded Matplotlib plot
  - Progress bar for long computations
  - Calculation history
  - Multi-language interface

---

## 🛠 Requirements

- **Python 3.9 or higher**

### Required Python packages

Install dependencies using pip:

```bash
pip install numpy sympy scipy matplotlib
