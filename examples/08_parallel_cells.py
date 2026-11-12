"""
Example 8: a module of parallel cells with uneven cooling
=========================================================

Four cells are connected in parallel, and sit at different places on a
cold plate, so some are cooled better than others. Their resistance
falls as they warm up, and parallel cells share current in inverse
proportion to their resistance. So a poorly cooled cell runs hotter,
takes more of the current, generates more heat, and runs hotter still.

To capture this, each cell gets its own electrical model and its own
thermal model, and all of them are solved together. The recipe is the
same as Example 0: one A and b per cell, appended to the electrical
equations. With several cells, it helps to stack them into one big
matrix.

The electrical parameters are made up, for illustration.
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
import scipy.linalg
from scipy.integrate import solve_ivp

# %% 1. Thermal models: one per cell
# All cells are the same, but cooled differently [W/m2/K].
cooling = np.array([1000, 300, 150, 50])
n_cells = len(cooling)
cell = coresurf.ThermalParameters(L=0.0113, k=0.916, h0=150, hL=150, rho=2408, cp=1030)
volume = 0.113 * 0.04 * 0.0113  # [m3]
T_inf = 25.0  # ambient temperature [degC]

# Each cell's thermal model has states [T_mean, T_core]. Stacking the
# matrices block-diagonally gives one model for the whole module, with
# states [T_mean_1, T_core_1, T_mean_2, T_core_2, ...].
matrices = [
    coresurf.symmetric_core_surface_model(cell.with_cooling(h)).core.get_model_matrices()
    for h in cooling
]
A = scipy.linalg.block_diag(*[A_i for A_i, b_i in matrices])  # (2n x 2n)
b = np.concatenate([b_i for A_i, b_i in matrices])  # (2n)

# %% 2. Electrical model: one per cell
# Each cell is an open-circuit voltage, a series resistance R0, and one
# RC pair, with SOC and the RC voltage as states. The resistances fall
# with temperature (an Arrhenius law).
capacity = 5.0 * 3600  # [A s] (5 Ah)
R0_ref, R1_ref, C1 = 0.03, 0.015, 2000.0  # [Ohm], [Ohm], [F], at 25 degC


def ocv(soc):
    return 3.2 + 0.9 * soc  # [V]


def resistance_factor(T):
    """How much the resistances change from their 25 degC values."""
    return np.exp(3000 * (1 / (T + 273.15) - 1 / 298.15))


def module_current(t):
    """Current drawn from the whole module: 60 A pulses, 60 s on, 60 s off."""
    return 60.0 if (t % 120) < 60 else 0.0


def cell_currents(t, soc, v_rc, R0):
    """
    Split the module current between the parallel cells. All cells have
    the same terminal voltage V = ocv - I R0 - v_rc, and their currents
    add up to the module current, which gives V and then each I.
    """
    V = (np.sum((ocv(soc) - v_rc) / R0) - module_current(t)) / np.sum(1 / R0)
    return (ocv(soc) - v_rc - V) / R0


# %% 3. The whole module as one ODE system
# State vector: [SOC (n), V_rc (n), temperatures (2n)]


def split(x):
    soc, v_rc, temps = x[:n_cells], x[n_cells : 2 * n_cells], x[2 * n_cells :]
    T_mean = temps[0::2]  # every other entry: the volume averages
    return soc, v_rc, temps, T_mean


def rhs(t, x):
    soc, v_rc, temps, T_mean = split(x)

    # Electrical dynamics, with resistances at each cell's temperature
    R0 = R0_ref * resistance_factor(T_mean)
    R1 = R1_ref * resistance_factor(T_mean)
    I = cell_currents(t, soc, v_rc, R0)
    dsoc = -I / capacity
    dv_rc = (I - v_rc / R1) / C1

    # Thermal dynamics: each cell's losses heat that cell
    heat = I**2 * R0 + v_rc**2 / R1  # [W], one per cell
    q = np.repeat(heat / volume, 2)  # [W/m3], once for each of a cell's two states
    dtemps = A @ (temps - T_inf) + b * q

    return np.r_[dsoc, dv_rc, dtemps]


# %% 4. Solve
x0 = np.r_[np.ones(n_cells), np.zeros(n_cells), np.full(2 * n_cells, T_inf)]
times = np.linspace(0, 1800, 1801)
soln = solve_ivp(rhs, [times[0], times[-1]], x0, t_eval=times,
                 max_step=1.0, rtol=1e-7, atol=1e-7)  # max_step: don't miss a pulse
soc, v_rc, temps, T_mean = split(soln.y)
T_core = temps[1::2]

# Each cell's current, worked out again from the solution
R0 = R0_ref * resistance_factor(T_mean)
I = np.array([cell_currents(t, *args) for t, *args in zip(times, soc.T, v_rc.T, R0.T)]).T

# %% 5. Plot
labels = [f"h = {h}" for h in cooling]
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
for i in range(n_cells):
    # Average current over the last 2-minute pulse cycle, to smooth out the pulses
    axes[0].plot(times[119:], np.convolve(I[i], np.ones(120) / 120, "valid"), f"C{i}",
                 label=labels[i])
    axes[1].plot(times, T_core[i], f"C{i}")
    axes[2].plot(times, soc[i], f"C{i}")
axes[0].set_ylabel("Cell current\n(2-min average) [A]")
axes[0].legend(ncol=4, fontsize="small", loc="lower center", bbox_to_anchor=(0.5, 1.0))
axes[1].set_ylabel("Core temperature [°C]")
axes[2].set_ylabel("SOC [-]")
axes[2].set_xlabel("Time [s]")
fig.tight_layout()
plt.show()

# The poorly cooled cell (h = 50) heats up, its resistance falls, and it
# takes more of the current, so it also discharges faster. Later, its
# lower SOC (and so lower open-circuit voltage) pushes current back to
# the other cells, partly evening things out.
print("Cell     Peak core temp   Final SOC")
for i in range(n_cells):
    print(f"{labels[i]:8} {T_core[i].max():10.1f} degC {soc[i, -1]:10.3f}")
