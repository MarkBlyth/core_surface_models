"""
Example 2: modelling your own cell
==================================

How to enter your own cell's properties, check what kind of thermal
behaviour to expect, and choose a model.
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

# %% 1. Enter your cell's thermal properties (all SI units)
# The model describes heat flow through the thickness of the cell: across
# the layers of a pouch or prismatic cell.
mass = 0.123  # cell mass [kg]
thickness = 0.0113  # [m]
width, height = 0.04, 0.113  # face dimensions [m]
volume = thickness * width * height  # [m3]

params = coresurf.ThermalParameters(
    L=thickness,  # thickness, from face x=0 to face x=L [m]
    k=0.916,  # through-plane thermal conductivity [W/m/K]
    h0=150,  # heat transfer coefficient on face x=0 [W/m2/K]
    hL=150,  # heat transfer coefficient on face x=L [W/m2/K]
    rho=mass / volume,  # density [kg/m3]
    cp=1030,  # specific heat capacity [J/kg/K]
)

# %% 2. What kind of behaviour should we expect?
# describe() prints the Biot number (how large the core-surface temperature
# difference will be), the modal timescales (how quickly the cell responds),
# and a suggested model.
coresurf.describe(params)

# %% 3. Simulate with the suggested model, and the other options
heat = 15 / volume  # 15 W [W/m3]
times = np.linspace(0, 1800, 400)  # [s]
T_ambient = 20  # [degC]

recommended = coresurf.symmetric_core_surface_model(params).simulate(heat, times, T_ambient)

# The general models work for any cooling (equal or not). Their two
# results are the volume-averaged temperature T_mean, and the
# temperature T_0 of face x=0 (here, the surface).
phi_psi = coresurf.eigenmode_steadystate_model(params).simulate(heat, times, T_ambient)
two_eig = coresurf.two_eigenmode_model(params).simulate(heat, times, T_ambient)
small_bi = coresurf.small_biot_model(params).simulate(heat, times, T_ambient)

# The traditional literature core-surface model, for comparison
literature = coresurf.literature_model(params).simulate(heat, times, T_ambient)

# %% 4. Compare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax1.plot(times, recommended["T_core"], "k", label="Core, recommended model")
ax1.plot(times, literature["T_core"], "C4--", label="Core, literature model")
ax1.plot(times, recommended["T_surf"], "k:", label="Surface, recommended model")
ax1.plot(times, literature["T_surf"], "C4:", label="Surface, literature model")
ax1.set_title("Recommended vs literature model")

ax2.plot(times, phi_psi["T_mean"], "C2", label=r"$\{\phi_1, \Psi\}$")
ax2.plot(times, two_eig["T_mean"], "C1--", label=r"$\{\phi_1, \phi_2\}$")
ax2.plot(times, small_bi["T_mean"], "C0:", label="small Biot")
ax2.set_title("Volume-averaged temperature")

for ax in (ax1, ax2):
    ax.set_xlabel("Time [s]")
    ax.legend()
ax1.set_ylabel("Temperature [°C]")
plt.show()

# The literature model overpredicts core temperature, because it assumes
# all heat is made at the centre of the cell. See Section 2 of the paper.
