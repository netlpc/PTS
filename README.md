Integration Calculator (GUI)

A Python-based graphical integration calculator built with Tkinter.
This project provides a user-friendly interface for computing symbolic and numerical integrals, visualizing functions, and working with improper integrals.

The calculator is designed to accept natural mathematical input (e.g. 2pi, 2(x+1), e^x) while maintaining robust and accurate parsing.

Features

Symbolic Integration
	•	Indefinite and definite integrals
	•	Exact symbolic results using SymPy
	•	Supports constants such as π and e

Numerical Integration

The following numerical methods are available:
	•	Trapezoidal Rule
	•	Simpson’s Rule
	•	Simpson’s 3/8 Rule
	•	Romberg Integration
	•	Gaussian Quadrature (Gauss–Legendre)
	•	Adaptive Simpson (via scipy.integrate.quad)
	•	Monte Carlo Integration (stratified sampling)

Improper Integrals
	•	Supports infinite bounds using -inf and inf
	•	Automatic handling without plotting infinite intervals

Expression Parsing
	•	Supports implicit multiplication (2pi, 3x, 2(x+1))
	•	Supports both exp(x) and e^x
	•	Correct handling of standard functions (sin, cos, log, etc.)
	•	Uses SymPy’s official implicit multiplication parser for reliability

GUI
	•	Built with Tkinter
	•	Embedded Matplotlib function plot
	•	Progress bar for long numerical computations
	•	Calculation history panel
	•	Multi-language interface:
	•	English
	•	中文（简体）
	•	國語（繁体）
	•	日本語
	•	한국어
	•	Español
	•	Français
	•	العربية
	•	हिन्दी

Note: I do not speak Korean. All Korean text was written using a translator.
If you are fluent in Korean and notice any issues, I would greatly appreciate corrections.

Requirements
	•	Python 3.9 or higher

Required Python Packages

Install dependencies using pip:
pip install numpy sympy scipy matplotlib

If the above command does not work correctly, try:
/usr/local/bin/python3 -m pip install numpy sympy scipy matplotlib

Running the Program

Run the main script directly:
python Integral_Calculator.py


The GUI will launch automatically.

Notes
	•	The calculator is intended for single-variable functions in x
	•	Symbolic integration may fail for functions without closed-form antiderivatives
	•	Numerical methods require finite bounds unless otherwise stated
