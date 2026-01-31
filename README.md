# Integration Calculator (GUI)
A Python-based graphical integration calculator built with **Tkinter**, supporting:

- Symbolic integration (via SymPy)
- Multiple numerical integration methods
- Improper integrals (±∞)
- Real-time function plotting
- Multi-language UI :English / 中文 / 日本語 / 한국어(I don't speak Korean, so all the Korean text was written using a translator. If anyone knows Korean, I would greatly appreciate it if you could proofread it. Thank you very much.)


## Features

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


## Requirements

- **Python 3.9 or higher**

### Required Python packages

Install dependencies using pip:

```bash
pip install numpy sympy scipy matplotlib
(If that one is not functioning well, try this)
/usr/local/bin/python3 -m pip install numpy scipy sympy matplotlib
