"""
Example 5: using the model equations directly
=============================================

Every model is two linear differential equations,

    d/dt [T_mean, T_0] = A @ [T_mean - T_amb, T_0 - T_amb] + b * q(t)

where A is a 2x2 matrix and b a 2-vector. This example shows how to get
A and b, what they tell you, and how to solve the equations with a
standard ODE solver, for example as part of a larger model of your own.
"""

# %% Setup: let Python find the coresurf package
# If you have installed coresurf (see docs/guide.md), this just imports it.
# Otherwise it uses the copy in this folder, so the example still runs.
import sys
from pathlib import Path

try:
    here = Path(__file__).resolve().parent  # the examples folder
except NameError:  # some editors do not set __file__ when running cells
    here = Path.cwd()

try:
    import coresurf
except ImportError:
    sys.path.insert(0, str(here.parent / "src"))
    import coresurf

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

# %% 1. Get the matrices
params = coresurf.kokam_pouch(150)
model = coresurf.eigenmode_steadystate_model(params)
A, b = model.A, model.b
print("A =\n", A)
print("b =", b)
# In the paper's notation (Section 2.5):
#   A = [[-1/(C1 R1), -1/(C1 R2)],      b = [w1/C1,
#        [ 1/(C2 R3), -1/(C2 R4)]]           w2/C2]

# %% 2. What the matrices tell you
# The model's two time constants [s] come from the eigenvalues of A. The
# slow one equals the cell's first modal timescale tau_1, the time for
# the whole cell to heat up. The fast one sets how quickly the shape of
# the temperature profile settles.
print("Model time constants [s]:", np.sort(-1 / np.linalg.eigvals(A).real)[::-1])
print("Cell modal timescales [s]:", coresurf.modal_timescales(params, 2))

# The steady state under constant heating q is where d/dt = 0:
#   0 = A (x - T_amb) + b q   =>   x = T_amb - A^-1 b q
T_amb = 25.0
heat = 20 / (0.113 * 0.04 * 0.0113)  # 20 W, as W/m3
steady = T_amb - np.linalg.solve(A, b * heat)
print("Steady state [T_mean, T_0] [degC]:", steady)

# %% 3. Solving the equations with a standard ODE solver
# Write the equations as a function f(t, x) that returns dx/dt, and give
# it to an ODE solver. This is all model.simulate does; writing it out
# yourself is useful when the thermal model is part of a larger model
# (see Example 0 for the minimal recipe, Example 8 for a module of
# several cells, and Example 7 for coupling to an electrical model).


def heat_generation(t):
    """20 W for the first 15 minutes, then off [W/m3]."""
    return heat if t < 900 else 0.0


def thermal_rhs(t, x):
    return A @ (x - T_amb) + b * heat_generation(t)


times = np.linspace(0, 1800, 361)
result = scipy.integrate.solve_ivp(
    thermal_rhs,
    [times[0], times[-1]],
    [T_amb, T_amb],  # start at ambient
    t_eval=times,
    max_step=1.0,  # do not step over the heating switching off
    rtol=1e-7,
    atol=1e-7,
)

# The built-in method gives the same answer
soln = model.simulate(heat_generation, times, T_ambient=T_amb, max_step=1.0)
print("Largest difference from model.simulate:", np.max(np.abs(result.y - soln.y)), "degC")

# model.rhs builds the same f(t, x) for you:
f = model.rhs(heat_generation, T_amb)

fig, ax = plt.subplots()
ax.plot(times, result.y[0], label="T_mean (volume average)")
ax.plot(times, result.y[1], label="T_0 (surface)")
ax.axhline(steady[0], color="C0", ls=":", label="Steady state, T_mean")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [°C]")
ax.legend()
plt.show()

# %% 4. Using the model in other tools
# Written in terms of the temperature rise above ambient, y = x - T_amb,
# the model is a standard linear state-space model with one input, q:
#     dy/dt = A y + b q
# so A and b can be used directly in Simulink (State-Space block),
# Modelica, or a Kalman filter; add T_amb back to the outputs. Solve them
# with a proper adaptive ODE solver (for example Simulink's variable-step
# ode45 or ode15s), rather than a hand-written fixed-step loop, so that
# accuracy is controlled automatically.
