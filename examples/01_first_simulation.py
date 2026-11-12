"""
Example 1: your first simulation
================================

A pouch cell is cooled equally on both faces and generates 20 W of heat.
How hot do its centre (core) and surface get?

How to run: open this file in your editor and run it (F5 in Spyder, the
Run button in VS Code), or type `python examples/01_first_simulation.py`
in a terminal. The file is split into cells, the blocks starting with
"# %%"; in Spyder (Ctrl+Enter), VS Code ("Run Cell") and other editors
that support them, you can run one cell at a time. See Section 5 of
docs/guide.md for details.
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

# %% 1. Describe the cell
# kokam_pouch() gives the 5 Ah pouch cell used in the paper. The number
# is the heat transfer coefficient h [W/m2/K] on each face: 30 is roughly
# still air, 150 forced air, 1000 a cold plate or immersion cooling.
params = coresurf.kokam_pouch(150)
print(params)

# %% 2. Build the model
# This is the recommended model for a cell cooled equally on both faces.
model = coresurf.symmetric_core_surface_model(params)

# %% 3. Choose the heat generation and simulation times
# Heat generation is per unit volume [W/m3]: total heat [W] / cell volume [m3].
cell_volume = 0.113 * 0.04 * 0.0113  # m3
heat = 20 / cell_volume  # 20 W spread evenly through the cell

times = np.linspace(0, 1200, 300)  # 0 to 20 minutes, 300 output points [s]

# %% 4. Run the simulation
soln = model.simulate(heat, times, T_ambient=25)

# The results are arrays with one value per output time:
core = soln["T_core"]  # temperature at the centre of the cell [degC]
surface = soln["T_surf"]  # temperature at the cell surface [degC]
mean = soln["T_mean"]  # volume-averaged temperature [degC]

print(f"After {times[-1]:.0f} s: core {core[-1]:.2f} degC, "
      f"surface {surface[-1]:.2f} degC, average {mean[-1]:.2f} degC")

# %% 5. Plot the results
fig, ax = plt.subplots()
ax.plot(times, core, label="Core (centre)")
ax.plot(times, mean, label="Volume average")
ax.plot(times, surface, label="Surface")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [°C]")
ax.set_title("20 W in a pouch cell, h = 150 W/m²/K")
ax.legend()
plt.show()

# %% 6. Temperature through the thickness of the cell
# The model can also rebuild the full temperature profile across the cell.
x = np.linspace(0, params.L, 101)  # positions through the cell [m]
profile = model.temperature_profile(soln, x, T_ambient=25)  # rows: positions, columns: times

fig, ax = plt.subplots()
for i in [10, 30, 75, 299]:
    ax.plot(x * 1000, profile[:, i], label=f"t = {times[i]:.0f} s")
ax.set_xlabel("Position through cell [mm]")
ax.set_ylabel("Temperature [°C]")
ax.legend()
plt.show()
