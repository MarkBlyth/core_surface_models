# A practical guide to the coresurf thermal models

This guide is for engineers who want to predict the temperature inside a
battery cell. It explains what the models do, which one to use, and how
to run them. 

The models come from the paper *Why core-surface battery thermal models
fail, and how to fix them* (Blyth, Hall and Hales). The paper has the
full derivations; this guide covers what you need to use them.

**Contents**

1. [What problem do these models solve?](#1-what-problem-do-these-models-solve)
2. [What the models describe](#2-what-the-models-describe)
3. [The available models](#3-the-available-models)
4. [Choosing a model](#4-choosing-a-model)
5. [Getting started](#5-getting-started)
6. [Entering your cell's properties](#6-entering-your-cells-properties)
7. [Common tasks](#7-common-tasks)
8. [Limitations](#8-limitations)
9. [Troubleshooting](#9-troubleshooting)
10. [Where to find more in the paper](#10-where-to-find-more-in-the-paper)

---

## 1. What problem do these models solve?

A cell is not at one temperature. Heat is generated throughout its
interior and escapes through its surfaces, so the inside runs hotter than
the outside. With strong cooling, or a thick cell, the difference can be
large. That matters, because the centre of the cell can be close to a
safety or ageing limit while a sensor on the surface reads an acceptable
temperature.

Detailed simulations (finite-element or finite-volume models) could be used to describe
this, but they are too slow for a battery management system,
a pack-level model with hundreds of cells, or a design optimisation that
runs thousands of simulations. Engineers have therefore used
core-surface models, consisting of two coupled equations, one for a "core"
temperature and one for a "surface" temperature.

The paper shows that the core-surface model usually used in the
literature has two built-in flaws:

- **It assumes all heat is generated at the very centre of the cell.**
  Real cells generate heat throughout. The model therefore overpredicts
  the core temperature and makes the cell look slower to respond than it is.
- **It represents the temperature profile with only two points.** That
  implicitly assumes a linear temperature profile, which is too coarse
  to describe the true temperature variations through a cell as it
  heats and cools.

These errors are easy to miss. The surface temperature settles to the
right value eventually, so a model can look well validated against a
surface thermocouple while being badly wrong at the core. When the model
is fitted to test data, the fitting process tries to compensate for the
wrong structure by distorting the parameters (overfitting). The fitted
model may then fail under conditions it wasn't fitted on, such as
different cooling, a different drive cycle, or fast charging.

The models in this package fix both problems:

- They spread heat generation through the whole cell, as in reality.
- Instead of two points, they describe the temperature profile using the
  two most important *shapes* it takes (the dominant "heat-transfer
  modes"). This is far more accurate for the same amount of computation.
- They are built from **physical properties** (thermal conductivity,
  density, heat capacity, heat transfer coefficient), so they don't need
  fitting and can be used before any cells exist, in design studies.
- Their "core" state is the **volume-averaged temperature** of the cell.
  This is physically meaningful, and it is the natural temperature to
  feed into an electrical model.

The result is still two equations, so it runs just as fast as the
traditional model. In the paper's validation, the recommended model is
almost indistinguishable from a detailed simulation.

---

## 2. What the models describe

Think of the cell as a slab with thickness `L`. Heat flows through
the thickness, from the inside towards the two large faces:

```
         face x = 0                                  face x = L
             |                                            |
 cooling h0  |   heat generated evenly throughout,  q     |  cooling hL
   <-------  |   conducted with conductivity k  -->       |  -------->
             |                                            |
             |<------------------- L -------------------->|
```

You provide:

| Symbol | Meaning | Units |
|---|---|---|
| `L` | Cell thickness (the direction heat flows out through) | m |
| `k` | Through-plane thermal conductivity | W m⁻¹ K⁻¹ |
| `rho` | Density | kg m⁻³ |
| `cp` | Specific heat capacity | J kg⁻¹ K⁻¹ |
| `h0`, `hL` | Heat transfer coefficient on each face | W m⁻² K⁻¹ |
| `q` | Heat generation **per unit volume** (can vary in time) | W m⁻³ |
| `T_ambient` | Ambient temperature (constant) | °C |

The models give you temperatures over time:

| Output | Meaning |
|---|---|
| `T_mean` | Volume-averaged cell temperature (the paper's "core" state) |
| `T_0` | Temperature of face x = 0 |
| `T_core` | Temperature at the centre of the cell (symmetric cooling only) |
| `T_surf` | Temperature of the faces (symmetric cooling only) |

You can also rebuild the **full temperature profile** through the cell,
for example to find the hottest point (see [Section 7](#7-common-tasks)).

The model equations always have the same form:

```
C1 dT_mean/dt = w1 q(t) - (T_mean - T_amb)/R1 - (T_0 - T_amb)/R2
C2 dT_0/dt    = w2 q(t) + (T_mean - T_amb)/R3 - (T_0 - T_amb)/R4
```

This is a thermal network of capacitances `C` and resistances `R`, like
the traditional model. The difference is that every value is calculated
for you from the physical properties, in a way that matches the real
heat flow. In matrix form this is

```
d/dt [T_mean, T_0] = A @ [T_mean - T_amb, T_0 - T_amb] + b * q(t)
```

and the package gives you `A` and `b` directly, so the model can be
moved into any other tool (see [Example 5](../examples/05_using_the_matrices.py)).

---

## 3. The available models

All models take the same `ThermalParameters` and are used the same way.

### Recommended: `symmetric_core_surface_model`

For a cell **cooled equally on both faces** (`h0 == hL`). Gives
`T_core`, `T_surf` and `T_mean`. This is the model to use in most cases.

Internally it is the {φ₁, Ψ} model below, built on half of the cell (from
centre to surface) to give the centre and surface temperatures directly.

### `eigenmode_steadystate_model`: the {φ₁, Ψ} model

For **any cooling**, including different cooling on each face or one face
insulated. Gives `T_mean` and `T_0`.

It describes the temperature profile using two shapes:

- **φ₁**, the slowest natural cooling pattern of the cell. This captures
  how the cell as a whole heats up and cools down.
- **Ψ**, the steady-state temperature profile: the shape the temperature
  settles to under constant heating. This makes the model **exact at
  steady state**.

This is the most accurate choice for almost all battery applications.
It can be slightly less accurate on very short timescales (seconds),
just after a sudden change in heating.

### `two_eigenmode_model`: the {φ₁, φ₂} model

For any cooling. Gives `T_mean` and `T_0`. It uses the two slowest
natural cooling patterns of the cell. It handles very fast changes well,
but leaves a small error at steady state, because it ignores the faster
patterns. Use it when you care mainly about short timescales (see
[Section 4](#4-choosing-a-model)).

### `small_biot_model`

For any cooling. Gives `T_mean` and `T_0`. It is derived by assuming the
temperature difference across the cell is small (Biot number below about
0.5). Its equations are simple enough to write out by hand, which makes
it handy for analysis and teaching. It loses accuracy when temperature
differences are large.

### `literature_model`

The traditional core-surface model, set up from physical properties as
in Section 2 of the paper. It is included **for comparison only**. It
overpredicts core temperature and thermal lag.

### Custom models

If you want to try your own thermal modes, use
`GalerkinProjection` with your own `Mode` objects. Most users won't need
this.

```python
def my_shape(x):          # your trial function, of position x [m]
    ...
def my_shape_d2(x):       # its second derivative
    ...

my_mode = coresurf.Mode.normalised(mode=my_shape, second_derivative=my_shape_d2)
phi1 = coresurf.EigenmodeFamily(params).mode(1)
model = coresurf.GalerkinProjection(params, phi1, my_mode).model()
```

A trial function must meet four conditions. The package checks the
first three, and stops with an explanation if one fails.

1. **It must equal 1 at x = 0**, that is φ(0) = 1. The core-surface
   form relies on this: it is what makes the second state the
   temperature of face x = 0. `Mode.normalised` takes care of it by
   dividing your function (and its second derivative) by its value at
   x = 0. A function that is zero at x = 0 can't be used. If you build a
   `Mode` directly, it must already equal 1 there.
2. **It must satisfy the cooling boundary conditions** of your cell,
   k φ'(0) = h0 φ(0) and −k φ'(L) = hL φ(L). The method uses φ''
   directly, so shapes that ignore the cooling give wrong answers. This
   also means modes built for one set of `h` values can't be reused for
   another.
3. **The two shapes must be genuinely different**, with different
   volume averages.
4. **Both functions must accept an array of positions**, so use NumPy
   functions (`np.cos`, not `math.cos`).

The built-in modes (`EigenmodeFamily(params).mode(n)` and
`steadystate_mode(params)`) meet all of these.

---

## 4. Choosing a model

**Short answer:** use `symmetric_core_surface_model` if both faces are
cooled equally, otherwise `eigenmode_steadystate_model`.

To check, run

```python
coresurf.describe(params)
```

This prints two numbers that explain your cell's thermal behaviour, and
suggests a model:

- **Biot number**, `Bi = hL/k`. This compares how easily heat leaves the
  surface with how easily it moves through the cell. A small Biot number
  (well below 1) means the cell is nearly the same temperature
  throughout. A large one means a big difference between centre and
  surface. The traditional model gets worse as the Biot number grows.
  The small-Biot model is fine below about 0.5. The {φ₁, Ψ} model works
  at any practical Biot number. (Below about 0.0015 its two shapes become
  numerically indistinguishable; it then stops with a message suggesting
  the small-Biot model, which is very accurate there.)
- **Modal timescales** `tau_1, tau_2, tau_3` (seconds). `tau_1` is roughly
  how long the whole cell takes to heat up or cool down. `tau_2` and
  `tau_3` are how quickly the *shape* of the temperature profile settles
  after a change.

Compare these with how quickly your heat generation changes:

- If heating changes much more slowly than `tau_2`, or you care about
  behaviour over minutes to hours: **{φ₁, Ψ}** (the usual case).
- If you care mainly about very short events, quicker than `tau_1`, or
  heating keeps changing on the same timescale as `tau_1` and `tau_2`:
  consider **{φ₁, φ₂}**, and compare it with {φ₁, Ψ}.

If in doubt, run both and compare (see
[Example 2](../examples/02_your_own_cell.py)). If PyBaMM is installed, you
can also check either model against a detailed simulation (see
[Example 6](../examples/06_check_against_full_model.py)).

---

## 5. Getting started

The examples work in any Python environment. Pick the route that suits
you:

- [Quick start for experienced users](#quick-start-for-experienced-users):
  terminal, virtual environments, any editor.
- [Step by step in Spyder](#step-by-step-in-spyder): the most
  hand-holding, if you're new to Python or use Anaconda.
- [VS Code](#vs-code), or [other editors](#other-editors-and-jupyter)
  (Emacs, Vim, PyCharm, Jupyter).

### What you need

- Python 3.10 or newer, with `numpy` and `scipy` (the models) and
  `matplotlib` (the examples' plots).
- `pandas` for Example 7, and PyBaMM for Example 6 and the reference
  model. Both are optional.

[Anaconda](https://www.anaconda.com/download) includes Python, Spyder
and everything except PyBaMM, so it's the easiest way to start if you
have nothing installed.

### Get the code

Download the repository from GitHub (green **Code** button, then
**Download ZIP**) and unzip it somewhere convenient, or clone it:

```bash
git clone https://github.com/MarkBlyth/core_surface_models.git
```

You'll get a folder containing `src`, `examples`, `docs`, `data` and so
on.

### Quick start for experienced users

From the repository folder, in whichever environment you use (venv,
conda, ...):

```bash
pip install -e ".[examples]"
```

```bash
python examples/01_first_simulation.py
```

The `examples` extra adds matplotlib and pandas; use `".[dev]"` to also
get pytest and PyBaMM, and run `pytest` to check everything works. The
examples are plain scripts, so run them however you like. Each plot
opens in a window; close it to continue the script. On a machine with no
display, set `MPLBACKEND=Agg` and save figures with `plt.savefig`
instead.

The examples are divided into blocks by `# %%` lines. Many editors treat
these as runnable cells (see below), but they're just comments
otherwise. Each example also finds the package in `src/` if it isn't
installed, so they run straight from a fresh download.

### Step by step in Spyder

**Try an example (no installation needed)**

1. Open Spyder (from the Start menu, or the Anaconda Navigator).
2. Use **File → Open** to open `examples/01_first_simulation.py`.
3. Press **F5** (or the green Run button).

The temperatures print in the console and the plots appear in the
**Plots** pane (top right). The examples find the package in the `src`
folder automatically, so this works before you install anything.

Each example is split into **cells**: blocks that begin with `# %%`.
Click inside a cell and press **Ctrl+Enter** to run just that block. This
lets you go through an example one step at a time and inspect the
results in the **Variable Explorer** as you go.

**Install the package (recommended)**

Installing lets you `import coresurf` from your own scripts, in any
folder. The easiest way is to type this in Spyder's console (bottom
right), using the path to your copy of the repository:

```
%pip install -e "C:/path/to/core_surface_models"
```

Then restart the console (**Consoles → Restart kernel**). Alternatively,
open the **Anaconda Prompt** (Windows) or a terminal (Mac/Linux), `cd`
to the repository folder, and run `pip install -e .`.

### VS Code

1. Install the **Python** extension (Extensions panel, search "Python").
2. Open the repository folder (**File → Open Folder**).
3. Choose your Python: **Ctrl+Shift+P → Python: Select Interpreter**.
4. Open an example and click the **Run** button (top right).

With the **Jupyter** extension installed, a **Run Cell** link appears
above each `# %%` line; it runs that block in an interactive window,
much like Spyder. To install the package, open a terminal in VS Code
(**Terminal → New Terminal**, which uses the interpreter you selected)
and run `pip install -e .`.

### Other editors and Jupyter

- **Emacs, Vim, or any editor plus a terminal:** run the examples with
  `python examples/<name>.py`, or interactively from IPython with
  `%run examples/<name>.py`, which leaves the results available to
  explore. In Emacs, `python-mode` can send the buffer or a region to an
  inferior Python (`C-c C-c`, `C-c C-r`), and packages such as
  `code-cells` add `# %%` cell support.
- **PyCharm:** open the folder, set the project interpreter, and run an
  example with the green arrow.
- **Jupyter:** in a notebook, `%run ../examples/01_first_simulation.py`
  runs an example, or copy its cells into notebook cells. Install the
  package with `%pip install -e ..` (from a notebook in a subfolder of
  the repository), then restart the kernel.

### Installation notes

Whichever route you take:

- `pip install -e .` installs the package from the repository folder.
  The `-e` means "editable": if you update the folder (for example by
  downloading a newer version), the installed package updates too.
- Extras add optional dependencies: `".[examples]"` (matplotlib, pandas),
  `".[reference]"` (PyBaMM, for Example 6), `".[paper]"` (everything
  needed to reproduce the paper), `".[dev]"` (tests).
- Install into the same Python your editor runs. If you're unsure which
  that is, run `import sys; print(sys.executable)` in it.
- After installing, restart any Python console or kernel that was
  already open.

### Your own script

The minimum you need is:

```python
import numpy as np
import coresurf

params = coresurf.ThermalParameters(L=0.0113, k=0.916, h0=150, hL=150, rho=2408, cp=1030)
model = coresurf.symmetric_core_surface_model(params)

times = np.linspace(0, 1200, 300)                 # seconds
heat = 20 / (0.113 * 0.04 * 0.0113)               # 20 W in the cell, as W/m3
soln = model.simulate(heat, times, T_ambient=25)

print(soln["T_core"][-1], soln["T_surf"][-1], soln["T_mean"][-1])
```

### The examples

| Script | What it shows |
|---|---|
| [00_minimal_example.py](../examples/00_minimal_example.py) | The minimal recipe: get `A` and `b`, append them to your own ODE right-hand side, solve with `solve_ivp` |
| [01_first_simulation.py](../examples/01_first_simulation.py) | The basics: one cell, constant heating, core/surface/average temperature, temperature profile |
| [02_your_own_cell.py](../examples/02_your_own_cell.py) | Entering your own properties, `describe()`, comparing all the models |
| [03_time_varying_heating.py](../examples/03_time_varying_heating.py) | Heating from a drive-cycle current profile loaded from a CSV file |
| [04_asymmetric_cooling.py](../examples/04_asymmetric_cooling.py) | Cold plate on one face, other face insulated; finding the hottest point |
| [05_using_the_matrices.py](../examples/05_using_the_matrices.py) | Getting `A` and `b`, what they tell you, solving the equations with a standard ODE solver |
| [06_check_against_full_model.py](../examples/06_check_against_full_model.py) | Checking against a detailed simulation (needs PyBaMM) |
| [07_electrothermal_ecm.py](../examples/07_electrothermal_ecm.py) | Coupling the thermal model with an equivalent-circuit electrical model, so that heat generation and temperature affect each other (needs pandas) |
| [08_parallel_cells.py](../examples/08_parallel_cells.py) | A module of parallel cells with uneven cooling: one thermal model per cell, stacked into one system, with temperature-dependent current sharing |

---

## 6. Entering your cell's properties

Use SI units throughout: metres, kilograms, seconds, watts.

```python
params = coresurf.ThermalParameters(
    L=0.0113,     # thickness [m]
    k=0.916,      # through-plane thermal conductivity [W/m/K]
    h0=150,       # heat transfer coefficient, face x=0 [W/m2/K]
    hL=150,       # heat transfer coefficient, face x=L [W/m2/K]
    rho=2408,     # density [kg/m3]
    cp=1030,      # specific heat capacity [J/kg/K]
)
```

Where to find the values:

- **Thickness `L`**: from the datasheet. This is the dimension heat
  leaves through: the thickness of a pouch or prismatic cell, measured
  between the two cooled faces.
- **Density `rho`**: cell mass divided by cell volume.
- **Specific heat capacity `cp`**: usually 800–1200 J/kg/K for
  lithium-ion cells. You can measure it with calorimetry, or take it from
  the literature.
- **Thermal conductivity `k`**: use the *through-plane* value (across the
  electrode layers), not the in-plane value, which is much higher. It is
  typically 0.5–1.5 W/m/K, and can be measured with heat-flux or
  guarded-hot-plate tests, or taken from the literature.
- **Heat transfer coefficients `h0`, `hL`**: these describe your cooling.
  Rough guides: natural convection in air 5–30; forced air 30–200;
  liquid cold plate or immersion 300–2000+ W/m²/K. They should include
  any contact or interface resistance between the cell and the cooling.
  Use 0 for a perfectly insulated face. The cell cooling coefficient
  method (Hales et al.) measures an effective value for a real cell.
- **Heat generation `q`**: the models use heat per unit *volume*. Divide
  the total heat in watts by the cell volume in m³. A quick estimate of
  the heat is I²R; for better accuracy, use the output of an electrical
  model.

`kokam_pouch(h0, hL)` gives the 5 Ah pouch cell used in the paper, as a
starting point or for testing.

Useful helpers:

```python
params.biot                   # Biot number
params.mirrored()             # swap the two faces
params.with_cooling(300)      # same cell, different (symmetric) cooling
params.with_cooling(300, 10)  # same cell, different cooling on each face
coresurf.describe(params)     # summary and model suggestion
```

---

## 7. Common tasks

### Constant heating

```python
soln = model.simulate(heat, times, T_ambient=25)
```

### Heating that changes over time

Give the heat as a table of `(times, values)`. Values in between are
linearly interpolated:

```python
soln = model.simulate((time_data, heat_data), times, T_ambient=25, max_step=0.5)
```

or as a Python function of time:

```python
soln = model.simulate(lambda t: 3e5 if t < 600 else 0.0, times, T_ambient=25, max_step=1.0)
```

**Always set `max_step` when heating changes quickly.** The solver takes
large steps when it can. Without a limit, it may step straight over a
short burst of heating and miss it. Use a value smaller than the spacing
of your heat data. `max_step` is only an upper limit: the solver still
chooses its own steps to meet its accuracy tolerances. See
[Example 3](../examples/03_time_varying_heating.py).

### Starting from a temperature other than ambient

```python
soln = model.simulate(heat, times, T_ambient=20, T_initial=35)
```


### Different cooling on each face

Use `eigenmode_steadystate_model`. Decide which face is x = 0: the model
reports `T_0`, the temperature of that face. If you have a thermocouple
on one face, make that face x = 0 so you can compare directly.

To get the temperature of the other face, either rebuild the profile
(below), or simulate the mirror-image cell:

```python
other_face = coresurf.eigenmode_steadystate_model(params.mirrored()).simulate(heat, times, 25)["T_0"]
```

If one face is insulated and you choose it as x = 0, `T_0` is the hottest
point in the cell. See [Example 4](../examples/04_asymmetric_cooling.py).

### Temperature profile and hottest point

```python
x = np.linspace(0, params.L, 101)                         # positions [m]
profile = model.temperature_profile(soln, x, T_ambient=25)  # shape: (positions, times)
hottest = profile.max(axis=0)                             # hottest point at each time
```

Remember to pass the same `T_ambient` you simulated with. This works for
`symmetric_core_surface_model`, `eigenmode_steadystate_model` and
`two_eigenmode_model`.

The profile is built from just two shapes. For the first few seconds
after a sudden change in heating, it can show small ripples, of a few
hundredths of a degree for the paper's cell. These disappear as the
temperature shape settles, within a few times `tau_3`.

### Using the model in another tool

```python
A, b = model.A, model.b
```

These two arrays are the whole model. You can copy them into Simulink, a
Kalman filter, or a pack model. The model is linear, so it is well suited
to state estimation. Whatever the tool, solve the equations with a
proper adaptive ODE solver (such as `scipy.integrate.solve_ivp`, or
Simulink's variable-step `ode45` or `ode15s`), which controls the
accuracy for you. Avoid hand-written fixed-step loops, such as Euler's
method: their accuracy depends on the step size, and nothing warns you
when it is too large. [Example 5](../examples/05_using_the_matrices.py)
shows how to solve the equations yourself, and what `A` and `b` tell you
about the cell. For symmetric cooling, the recommended model is two of
these models side by side: `model.core` (for `T_core`) and
`model.surface` (for `T_surf`).

### Using your own ODE solver

```python
f = model.rhs(heat, T_ambient=25)  # f(t, x) for any solver
```

### Adding temperatures to your own model

If you already have a model written as ODEs (an electrical model, a pack
or vehicle model, ...), get `A` and `b`, append the two thermal
equations to your right-hand side, and solve as usual. `q(t)` is the
heat your model generates, in W/m³:

```python
A, b = coresurf.eigenmode_steadystate_model(params).get_model_matrices()

def rhs(t, x):
    soc, T = x[0], x[1:]                        # your states, then [T_mean, T_0]
    dynamics = -current / capacity              # your model's equations
    return np.r_[dynamics, A @ (T - T_inf) + b * q(t)]

soln = solve_ivp(rhs, [0, 1500], [1.0, T_inf, T_inf])
```

That's the whole recipe; [Example 0](../examples/00_minimal_example.py)
is exactly this, as a runnable script. Note `@` (matrix multiplication),
not `*`. With equal cooling on both faces, take the matrices from
`coresurf.symmetric_core_surface_model(params).core` instead, to get the
states `[T_mean, T_core]`.

For several cells, give each its own `A` and `b` and stack them with
`scipy.linalg.block_diag`. Because the temperatures are now states, your
model can also use them, for example for temperature-dependent
resistance. [Example 8](../examples/08_parallel_cells.py) does both, for
a module of parallel cells with uneven cooling.

### Coupling with an electrical model

Heat generation depends on the cell's resistance, and resistance depends
on temperature. To capture this, solve an electrical model and the
thermal model together, as one set of equations.
[Example 7](../examples/07_electrothermal_ecm.py) does this with the
Thevenin equivalent-circuit model (ECM) in `coresurf.ecm`, from
Blyth, Gupta and Hales, *How to parameterise an equivalent-circuit
empirical battery model from time-domain data*, MethodsX (2026), 103865.
The idea is:

1. Put the electrical states (state of charge, RC-pair voltages) and the
   thermal states (`T_mean`, `T_0`) in one state vector.
2. At each time, look up the electrical parameters at the
   **volume-averaged temperature** `T_mean`. This is the right
   temperature to use, because the electrical model describes the cell
   as a whole.
3. Compute the heat generated in the resistances, I² R0 + Σ V_rc²/R_i,
   divide by the cell volume, and use it as `q` in the thermal equations.
4. Solve everything with `scipy.integrate.solve_ivp`.

```python
def electrothermal_rhs(t, x):
    soc, v_rc, temps = x[0], x[1:3], x[3:]
    I = current(t)
    R = ecm.resistances(temps[0], soc)        # at T_mean
    tau = ecm.time_constants(temps[0], soc)
    q = (I**2 * R[0] + np.sum(v_rc**2 / R[1:])) / cell_volume
    return np.concatenate([
        [-I / (ecm.capacity_Ah * 3600)],      # d SOC/dt
        (I * R[1:] - v_rc) / tau,             # d V_rc/dt
        A @ (temps - T_amb) + b * q,          # d [T_mean, T_0]/dt
    ])
```

To use your own cell, supply your own OCV and parameter tables in the
same CSV format as `data/MLP001_ocv.csv` and `data/MLP001_params.csv`.

### Getting results out

`soln.t` holds the times and `soln.y` the temperatures (one row per
output). To save them to a CSV file:

```python
np.savetxt("results.csv", np.column_stack([soln.t, soln.y.T]), delimiter=",",
           header="time," + ",".join(soln.state_names), comments="")
```

### If you used the code listing in the paper

The paper's listing (`build_eigenmode_steadystatemode_model(thermals)`,
then `projection.get_model_matrices()`) still works. Those names are kept
for compatibility.

---

## 8. Limitations

The models make the following assumptions. Check that they suit your
application.

- **Heat flows in one direction, through the thickness.** This suits
  pouch and prismatic cells, where the faces are much larger than the
  thickness. It doesn't capture heat flowing out through tabs or edges,
  or radial heat flow in cylindrical cells (the same approach applies to
  cylinders, but those models aren't implemented here).
- **Heat is generated evenly through the cell.** Local hotspots, such as
  near tabs, aren't represented.
- **Material properties are uniform and constant.** `k`, `rho` and `cp`
  don't vary with position or temperature.
- **The ambient temperature and heat transfer coefficients are
  constant** during a simulation. `T_ambient` must be a single number.
- **Temperatures are in the same units as `T_ambient`.** °C and K both
  work, as long as you are consistent.

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'coresurf'`**
Either install the package ([Section 5](#5-getting-started)), or put
your script in the `examples` folder and copy the setup block from an
example, which finds the package automatically. If you have installed
it, check that your editor uses the same Python you installed into
(`import sys; print(sys.executable)`), and restart the console or kernel.

**`ModuleNotFoundError: No module named 'pybamm'`**
PyBaMM is only needed for the reference model (Example 6 and the
`coresurf.reference` module). Install it with `pip install pybamm`, or
skip that example.

**Plots don't appear.**
In Spyder, look in the **Plots** pane (top right); for separate windows,
go to **Tools → Preferences → IPython console → Graphics**, set the
backend to *Automatic*, and restart the console. In VS Code's
interactive window, plots appear inline. From a terminal, each plot
opens in a window, and the script pauses until you close it. On a remote
machine or server with no display, set `MPLBACKEND=Agg` and use
`plt.savefig("figure.png")` instead of `plt.show()`.

**Temperatures are far too high or too low.**
Check units. Heat generation must be in W/m³ (total watts ÷ cell volume),
not watts. Thickness must be in metres, not millimetres.

**The temperature misses short bursts of heating.**
Set `max_step` (see [Section 7](#heating-that-changes-over-time)).

**A simulation is slow.**
Long simulations with a small `max_step` take longer. Use the largest
`max_step` that still resolves your heating data (about half its sample
spacing), and ask for fewer output times.

**`ValueError: half_domain() requires symmetric cooling`** or
**`Can only build a literature model for symmetric cooling`**
`symmetric_core_surface_model` and `literature_model` need `h0 == hL`.
For unequal cooling, use `eigenmode_steadystate_model`.

---

## 10. Where to find more in the paper

| Topic | Paper section |
|---|---|
| Why the traditional core-surface model fails, and what to do about it | 2 |
| How the {φ₁, φ₂} and {φ₁, Ψ} models are built from heat-transfer modes | 3 |
| The small-Biot model | 4 |
| Summary of the models, choosing between them with timescales, validation results | 5 |
| Full mathematical details | Supplementary material |

Glossary:

- **Biot number**: compares the resistance to heat flow inside the cell
  with the resistance to heat leaving its surface. A large Biot number
  means a large internal temperature difference.
- **Heat transfer coefficient (h)**: how readily heat leaves a surface,
  in W per m² of surface per degree of temperature difference.
- **Mode**: a characteristic shape of the temperature profile through
  the cell. Any temperature profile can be built up from modes; the
  models keep the two that matter most.
- **Galerkin projection**: the mathematical method used to turn the full
  heat equation into two equations for the chosen modes.
- **Steady state**: the condition reached under constant heating, when
  temperatures stop changing.
- **Volume-averaged temperature**: the average temperature over the
  whole cell.
