"""
Example 0: the minimal recipe
=============================

Get the thermal model's matrices A and b, add two lines to your own ODE
right-hand side, and solve. That's all there is to it.
"""

# %% Setup: find the coresurf package, even if it isn't installed
import sys
from pathlib import Path

try:
    import coresurf
except ImportError:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(here.parent / "src"))
    import coresurf

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# %% The minimal recipe
# 1. The thermal model's matrices, from the cell's properties
params = coresurf.ThermalParameters(L=0.0113, k=0.916, h0=150, hL=150, rho=2408, cp=1030)
A, b = coresurf.eigenmode_steadystate_model(params).get_model_matrices()
T_inf = 25.0  # ambient temperature [degC]

# 2. Your own model. Here: a 5 Ah cell discharged at 10 A, with a 50 mOhm
#    resistance; its only state is SOC.
current, resistance, capacity = 10.0, 0.05, 5.0 * 3600  # [A], [Ohm], [A s]
volume = 0.113 * 0.04 * 0.0113  # [m3]


def q(t):
    """Heat generation per unit volume [W/m3]."""
    return current**2 * resistance / volume


# 3. Append the thermal model to your right-hand side
def rhs(t, x):
    soc, T = x[0], x[1:]  # T = [T_mean, T_0]
    d_soc_dt = -current / capacity  # your model's equations
    d_thermals_dt = A @ (T - T_inf) + b * q(t)  # @ is matrix multiplication
    return np.r_[d_soc_dt, d_thermals_dt]


# 4. Solve, starting full and at ambient temperature
soln = solve_ivp(rhs, [0, 1500], [1.0, T_inf, T_inf], rtol=1e-6, atol=1e-6)

# %% Results
soc, T_mean, T_surface = soln.y
print(f"After 25 minutes: SOC = {soc[-1]:.2f}, T_mean = {T_mean[-1]:.1f} degC, "
      f"T_surface = {T_surface[-1]:.1f} degC")

plt.plot(soln.t, T_mean, label="Volume average")
plt.plot(soln.t, T_surface, label="Surface")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.show()
