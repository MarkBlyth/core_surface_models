"""
Example 7: coupling the thermal model with an electrical model (ECM)
====================================================================

In a real cell, heat generation and temperature affect each other:

    current --> electrical model --> heat generation --> thermal model
       ^                                                       |
       |              cell temperature (volume average)        |
       +-------------------------------------------------------+

The electrical model's resistances depend on temperature, so a warmer
cell generates less heat. This example joins the two models into one
set of differential equations, and solves them together.

The electrical model is a Thevenin equivalent-circuit model (ECM): a
series resistance R0 plus two resistor-capacitor (RC) pairs, with
parameters that depend on state of charge (SOC) and temperature. The
model and the parameter files in data/ (an MLP001 pouch cell) are from

    M. Blyth, A. Gupta and A. Hales, "How to parameterise an
    equivalent-circuit empirical battery model from time-domain data",
    MethodsX (2026), 103865.

Use your own parameter files in the same format to model your own cell.
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

from coresurf.ecm import TheveninModel  # needs pandas

data_dir = here.parent / "data"

# %% 1. The electrical model
# Two CSV lookup tables, each with SOC and Temperature_degC columns:
#   - MLP001_ocv.csv:    open-circuit voltage, OCV[V]
#   - MLP001_params.csv: R0 [Ohm], and R1 [Ohm], tau1 [s], R2 [Ohm], tau2 [s]
#     for the two RC pairs (tau = R C)
# Values in between table entries are interpolated. The tables cover
# 5-40 degC, so the cell temperature must stay in that range.
ecm = TheveninModel(
    data_dir / "MLP001_ocv.csv",
    data_dir / "MLP001_params.csv",
    capacity_Ah=2.132,
)
n_rc = ecm.n_rc  # number of RC pairs (2)

for temp in (10, 25, 40):
    print(f"At {temp} degC, 50% SOC: R0 = {ecm.resistances(temp, 0.5)[0]*1e3:.0f} mOhm")

# %% 2. The thermal model
# The MLP001 cell is 6 mm thick, 62 mm x 48 mm. We use the same material
# properties as the Kokam cell elsewhere in these examples; replace them
# with values for your own cell.
cell_volume = 0.062 * 0.048 * 0.006  # [m3]
params = coresurf.ThermalParameters(
    L=0.006,  # thickness [m]
    k=0.916,  # through-plane thermal conductivity [W/m/K]
    h0=150,  # cooling on each face [W/m2/K]
    hL=150,
    rho=2400,  # density [kg/m3]
    cp=1030,  # specific heat capacity [J/kg/K]
)
coresurf.describe(params)

# We use the {phi_1, Psi} model of the whole cell. Its two states are the
# volume-averaged temperature T_mean, which drives the electrical model,
# and the surface temperature T_0. Any of the models would work here.
thermal = coresurf.eigenmode_steadystate_model(params)
A, b = thermal.A, thermal.b
T_amb = 25.0  # ambient temperature [degC]

# %% 3. The current profile
# A section of the WLTP drive cycle, scaled up 10x for a demanding test,
# as in the paper. Positive current means discharge.
wltp = np.loadtxt(data_dir / "wltp.csv", delimiter=",", skiprows=1)
keep = (wltp[:, 0] > 1800) & (wltp[:, 0] < 8000)
current_times = wltp[keep, 0] - 1800  # [s], restarted at 0
current_values = 10 * wltp[keep, 1]  # [A]


def current(t):
    """Current [A] at time t [s], interpolated from the drive cycle."""
    return np.interp(t, current_times, current_values)


# %% 4. Join the two models
# The combined state vector is
#   x = [SOC, V_rc1, V_rc2, T_mean, T_0]
# The function below returns dx/dt; a standard ODE solver does the rest.


def heat_generation(I, soc, v_rc, T_mean):
    """Heat generated in the cell's resistances [W]."""
    R = ecm.resistances(T_mean, soc)
    # Power dissipated in R0, plus in the resistor of each RC pair
    return I**2 * R[0] + np.sum(v_rc**2 / R[1:])


def electrothermal_rhs(t, x):
    soc, v_rc, temps = x[0], x[1 : 1 + n_rc], x[1 + n_rc :]
    T_mean = temps[0]  # the electrical model sees the average temperature
    I = current(t)

    # Electrical model, with parameters at the present SOC and temperature
    R = ecm.resistances(T_mean, soc)
    tau = ecm.time_constants(T_mean, soc)
    if np.any(np.isnan(R)) or np.any(np.isnan(tau)):
        raise ValueError(
            f"At t = {t:.0f} s, SOC = {soc:.3f} and T_mean = {T_mean:.1f} degC "
            "are outside the electrical model's lookup tables."
        )
    dsoc_dt = -I / (ecm.capacity_Ah * 3600)
    dvrc_dt = (I * R[1:] - v_rc) / tau

    # Thermal model, heated by the electrical losses (in W/m3)
    q = heat_generation(I, soc, v_rc, T_mean) / cell_volume
    dT_dt = A @ (temps - T_amb) + b * q

    return np.concatenate([[dsoc_dt], dvrc_dt, dT_dt])


# %% 5. Solve
# Start fully charged at ambient temperature, with the RC pairs relaxed.
x_start = np.concatenate([[0.99], np.zeros(n_rc), [T_amb, T_amb]])
times = np.arange(0, 3100, 1.0)  # output every second

# The solver chooses its own step size to meet the tolerances. max_step
# caps it at 0.5 s, so that it cannot step over changes in current; the
# drive cycle is sampled every second. This takes a few seconds.
result = scipy.integrate.solve_ivp(
    electrothermal_rhs,
    [times[0], times[-1]],
    x_start,
    t_eval=times,
    max_step=0.5,
    rtol=1e-6,
    atol=1e-8,
)
if not result.success:
    raise RuntimeError(result.message)

soc = result.y[0]
v_rc = result.y[1 : 1 + n_rc]
T_mean, T_surf = result.y[1 + n_rc :]

# %% 6. Work out the voltage, heat and core temperature
I = current(times)
R0 = np.array([ecm.resistances(T, s)[0] for T, s in zip(T_mean, soc)])
voltage = ecm.ocv(T_mean, soc) - I * R0 - v_rc.sum(axis=0)
heat = np.array([heat_generation(*args) for args in zip(I, soc, v_rc.T, T_mean)])

# The thermal model can rebuild the temperature anywhere in the cell. The
# core is at the centre, x = L/2.
thermal_soln = coresurf.Solution(times, result.y[1 + n_rc :], thermal.state_names)
T_core = thermal.temperature_profile(thermal_soln, [params.L / 2], T_amb)[0]

print(f"Final SOC: {soc[-1]:.2f}")
print(f"Peak core temperature: {T_core.max():.1f} degC")

# %% 7. Plot
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
axes[0].plot(times, I, "C7")
axes[0].set_ylabel("Current [A]")
axes[1].plot(times, voltage, "C3")
axes[1].set_ylabel("Voltage [V]")
axes[2].plot(times, heat, "C5")
axes[2].set_ylabel("Heat generation [W]")
axes[3].plot(times, T_core, label="Core")
axes[3].plot(times, T_mean, label="Volume average")
axes[3].plot(times, T_surf, label="Surface")
axes[3].set_ylabel("Temperature [°C]")
axes[3].set_xlabel("Time [s]")
axes[3].legend()
fig.tight_layout()
plt.show()

# %% 8. Optional: how much does the coupling matter?
# Run the electrical model again with its temperature held at ambient,
# feed the resulting heat into the thermal model, and compare. Ignoring
# the feedback overestimates heat generation, because the resistances
# fall as the cell warms. The thermal model is passed the heat as a
# (times, values) table.
isothermal = ecm.simulate(
    current, initial_soc=0.99, temp_inf=T_amb, t_max=times[-1],
    t_eval=times, max_step=0.5, rtol=1e-6, atol=1e-8,
)
heat_isothermal = np.array([
    heat_generation(*args, T_amb)
    for args in zip(isothermal.current, isothermal.soc, isothermal.v_rc.T)
])
q_isothermal = heat_isothermal / cell_volume  # [W/m3]
uncoupled = thermal.simulate((times, q_isothermal), times, T_ambient=T_amb, max_step=0.5)

fig, ax = plt.subplots()
ax.plot(times, T_mean, label="Coupled")
ax.plot(times, uncoupled["T_mean"], "--", label="Electrical model held at 25 °C")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Volume-averaged temperature [°C]")
ax.legend()
plt.show()
difference = uncoupled["T_mean"] - T_mean
print(f"Ignoring the coupling overestimates T_mean by up to {difference.max():.2f} degC")
