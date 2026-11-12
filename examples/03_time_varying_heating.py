"""
Example 3: time-varying heat generation (drive cycle)
=====================================================

Heat generation usually changes over time. This example builds a heat
profile from a measured current profile, and simulates the cell
temperature.
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

# %% 1. Load a current profile
# data/wltp.csv has two columns: time [s] and current [A]. Swap in your
# own file; any two-column CSV with a header row will work.
data_file = here.parent / "data" / "wltp.csv"
data = np.loadtxt(data_file, delimiter=",", skiprows=1)
time_data = data[:, 0]  # [s]
current = 10 * data[:, 1]  # [A]; scaled up 10x for a demanding test

# Keep the busy part of the cycle (as in the paper), and restart the clock at 0
keep = (time_data > 1800) & (time_data < 4900)
time_data, current = time_data[keep] - 1800, current[keep]

# %% 2. Turn current into volumetric heat generation
# A simple estimate is Joule heating, I^2 R. Use your own heat
# generation data if you have it, for example from an electrical model.
resistance = 0.05  # cell internal resistance [Ohm]
volume = 0.113 * 0.04 * 0.0113  # [m3]
heat_data = current**2 * resistance / volume  # [W/m3]

# %% 3. Simulate
# Pass time-varying heat generation as a (times, values) pair; values in
# between are linearly interpolated.
params = coresurf.kokam_pouch(150)
model = coresurf.symmetric_core_surface_model(params)

times = np.arange(0, time_data[-1], 1.0)  # output every second
# max_step stops the solver taking steps longer than 0.5 s, so it cannot
# jump over rapid changes in heating. Use a value smaller than the
# spacing of your heat generation data.
soln = model.simulate((time_data, heat_data), times, T_ambient=25, max_step=0.5)

# %% 4. Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(time_data, heat_data / 1e3, "C5")
ax1.set_ylabel("Heat generation\n[kW/m³]")
ax2.plot(times, soln["T_core"], label="Core")
ax2.plot(times, soln["T_mean"], label="Volume average")
ax2.plot(times, soln["T_surf"], label="Surface")
ax2.set_ylabel("Temperature [°C]")
ax2.set_xlabel("Time [s]")
ax2.legend()
plt.show()
