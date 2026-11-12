"""
Example 4: unequal cooling on the two faces
===========================================

A cell sits on a cold plate on one face, with the other face nearly
insulated. Now the temperature is not symmetric, so we use the general
model, and choose which face is x = 0.
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

# %% 1. Set up the cooling
# Face x=0 is on the cold plate; face x=L is nearly insulated.
params = coresurf.kokam_pouch(h0=500, hL=5)
coresurf.describe(params)

# %% 2. Simulate
# The general model gives the volume-averaged temperature and the
# temperature of face x=0 (here, the cold-plate face).
model = coresurf.eigenmode_steadystate_model(params)
heat = 20 / (0.113 * 0.04 * 0.0113)  # 20 W [W/m3]
times = np.linspace(0, 1500, 300)
soln = model.simulate(heat, times, T_ambient=25)

# %% 3. Temperature of the other face, and the hottest point
# Rebuild the temperature profile through the cell, then read off the
# temperature at x = L and the maximum.
x = np.linspace(0, params.L, 201)
profile = model.temperature_profile(soln, x, T_ambient=25)
far_face = profile[-1]
hotspot = profile.max(axis=0)

# Alternatively, simulate the mirror-image cell, which puts the
# insulated face at x=0. Its T_0 is the far-face temperature directly.
far_face_check = coresurf.eigenmode_steadystate_model(params.mirrored()).simulate(
    heat, times, T_ambient=25
)["T_0"]
print(f"Far face at end: {far_face[-1]:.2f} degC (from profile), "
      f"{far_face_check[-1]:.2f} degC (from mirrored model)")

# %% 4. Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(times, hotspot, "C3", label="Hottest point")
ax1.plot(times, far_face, "C1--", label="Insulated face (x = L)")
ax1.plot(times, soln["T_mean"], "C2", label="Volume average")
ax1.plot(times, soln["T_0"], "C0", label="Cold-plate face (x = 0)")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Temperature [°C]")
ax1.legend()

for i in [20, 60, 299]:
    ax2.plot(x * 1000, profile[:, i], label=f"t = {times[i]:.0f} s")
ax2.set_xlabel("Distance from cold plate [mm]")
ax2.set_ylabel("Temperature [°C]")
ax2.legend()
plt.show()
