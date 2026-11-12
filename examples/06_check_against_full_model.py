"""
Example 6: checking a model against the full heat equation
==========================================================

Compares the two-state models with a detailed finite-volume simulation of
heat flow through the cell. Needs PyBaMM: pip install pybamm
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

from coresurf.reference import symmetric_core_surface

# %% 1. Simulate the full model and the two-state models
params = coresurf.kokam_pouch(1000)  # strong cooling: large temperature gradients
heat = 20 / (0.113 * 0.04 * 0.0113)
times = np.linspace(0, 400, 200)

full = symmetric_core_surface(params, heat, times)  # temperature rise above ambient
ours = coresurf.symmetric_core_surface_model(params).simulate(heat, times)
literature = coresurf.literature_model(params).simulate(heat, times)

# %% 2. Plot the errors
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 6))
for col, state, name in [(0, "T_core", "Core"), (1, "T_surf", "Surface")]:
    axes[0, col].plot(times, full[state], "k:", label="Full 1D model")
    axes[0, col].plot(times, ours[state], "C2", label="Recommended model")
    axes[0, col].plot(times, literature[state], "C4", label="Literature model")
    axes[0, col].set_title(f"{name} temperature rise [°C]")
    axes[1, col].plot(times, ours[state] - full[state], "C2")
    axes[1, col].plot(times, literature[state] - full[state], "C4")
    axes[1, col].set_title(f"{name} error [°C]")
    axes[1, col].set_xlabel("Time [s]")
axes[0, 0].legend()
plt.show()
