#!/usr/bin/env python3
"""
Minimal example: core and surface temperature of a symmetrically cooled
pouch cell, using your own cell parameters and heating profile.
"""

import matplotlib.pyplot as plt
import numpy as np

import coresurf

# Describe the cell through its thickness. All SI units.
params = coresurf.ThermalParameters(
    L=0.0113,  # thickness [m]
    k=0.916,  # through-plane conductivity [W/m/K]
    h0=150,  # heat transfer coefficient, x=0 face [W/m2/K]
    hL=150,  # heat transfer coefficient, x=L face [W/m2/K]
    rho=2400,  # density [kg/m3]
    cp=1030,  # specific heat capacity [J/kg/K]
)
print(f"Biot number: {params.biot:.3f}")

# Heat generation [W/m3]: a constant, a function of time, or a
# (times, values) table. Here: a 10-minute pulse.
t = np.linspace(0, 3600, 500)
heating = (np.array([0, 600, 601, 3600]), np.array([3e5, 3e5, 0, 0]))

# Recommended model for symmetric cooling: gives core and surface temperature.
model = coresurf.symmetric_core_surface_model(params)
soln = model.simulate(heating, t, T_ambient=25, max_step=1.0)

# The standard literature model, for comparison.
lit = coresurf.literature_model(params).simulate(heating, t, T_ambient=25, max_step=1.0)

fig, ax = plt.subplots()
ax.plot(t, soln["T_core"], "C2", label="Core (this model)")
ax.plot(t, soln["T_surf"], "C2--", label="Surface (this model)")
ax.plot(t, lit["T_core"], "C4", label="Core (literature model)")
ax.plot(t, lit["T_surf"], "C4--", label="Surface (literature model)")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Temperature [$^\circ$C]")
ax.legend()
plt.show()
