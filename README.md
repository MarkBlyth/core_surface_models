# core_surface_models

Accurate two-state (core-surface) thermal models for battery cells, plus
all scripts and data for reproducing the analysis in our paper *Why
core-surface battery thermal models fail, and how to fix them*.

The `coresurf` package builds a linear two-state model

    dx/dt = A (x - T_ambient) + b q(t)

from a Galerkin projection of the 1D through-thickness heat equation. The
recommended model uses the first eigenfunction and the steady-state
temperature profile as trial functions. This makes it exact at steady
state and much more accurate than the standard literature core-surface
model during transients.

## Installation

```bash
pip install -e .              # core package: numpy, scipy
pip install -e ".[reference]" # + PyBaMM, for the full 1D reference model
pip install -e ".[paper]"     # + everything needed to reproduce the paper
```

## Quick start

```python
import numpy as np
import coresurf

params = coresurf.ThermalParameters(
    L=0.0113,   # thickness [m]
    k=0.916,    # through-plane conductivity [W/m/K]
    h0=150,     # heat transfer coefficient at x=0 [W/m2/K]
    hL=150,     # heat transfer coefficient at x=L [W/m2/K]
    rho=2400,   # density [kg/m3]
    cp=1030,    # specific heat capacity [J/kg/K]
)

model = coresurf.symmetric_core_surface_model(params)
t = np.linspace(0, 600, 200)
soln = model.simulate(heatgen=3e5, t_eval=t, T_ambient=25)   # W/m3

soln["T_core"], soln["T_surf"]
```

There is a fuller example in [examples/quickstart.py](examples/quickstart.py).

### Heat generation

`heatgen` is the volumetric heat generation [W m⁻³]. It can be:

- a constant: `3e5`
- a function of time: `lambda t: 3e5 * (t < 600)`
- a table, linearly interpolated: `(times, values)`

For heat generation that changes quickly (for example a drive cycle),
pass `max_step=...` so the ODE solver does not step over the changes.
Any other keyword arguments go to `scipy.integrate.solve_ivp`.

### Available models

| Builder | States | Cooling |
|---|---|---|
| `symmetric_core_surface_model` (recommended) | `T_core`, `T_surf` | symmetric |
| `eigenmode_steadystate_model` — {φ₁, Ψ} Galerkin | `T_mean`, `T_0` | any |
| `two_eigenmode_model` — {φ₁, φ₂} Galerkin | `T_mean`, `T_0` | any |
| `small_biot_model` | `T_mean`, `T_0` | any |
| `literature_model` | `T_core`, `T_surf` | symmetric |

`T_mean` is the volume-averaged temperature and `T_0` is the temperature
at x = 0. For asymmetric cooling, use `eigenmode_steadystate_model`;
build it on `params.mirrored()` to get the temperature at the other face.

Every model except `symmetric_core_surface_model` is a
`CoreSurfaceModel` with attributes `A` and `b`. You can use these
directly, for example in a state estimator or a pack model. `model.rhs(heatgen)`
returns the ODE right-hand side for use with your own solver.

### Custom trial functions

```python
modes = coresurf.EigenmodeFamily(params)
my_mode = coresurf.Mode(mode=lambda x: ..., second_derivative=lambda x: ...)  # needs mode(0) == 1
model = coresurf.GalerkinProjection(params, modes.mode(1), my_mode).model()
```

### Reference solution

`coresurf.reference.solve_heat_equation(params, heatgen, t_eval)` solves
the full 1D heat equation with PyBaMM.
`coresurf.reference.symmetric_core_surface(...)` returns ground-truth
`T_core`, `T_surf` and `T_mean` in the same format as the models.

## Reproducing the paper

```bash
pip install -e ".[paper]"
python paper/run_all.py                 # PGF figures in ./figures, as in the paper
python paper/run_all.py --format pdf    # or PDF/PNG for viewing
```

Each script can also be run on its own:

| Script | Output |
|---|---|
| `paper/fig_literature_comparison.py` | `literature_model_comparison` |
| `paper/fig_newmodel_comparison.py` | `newmodel_comparison` |
| `paper/fig_summary_comparison.py` | `summary_comparison` |
| `paper/fig_wltp_comparison.py` | `wltp_comparison` |
| `paper/fig_wltp_zoom.py` | `wltp_zoom` |
| `paper/calc_steadystate_error.py` | steady-state error figures quoted in the text |
| `paper/calc_wltp_error.py` | WLTP RMS errors quoted in the text |

Figure scripts accept `--outdir`, `--format` and `--show`. PGF output
needs a LaTeX installation.

[paper/common.py](paper/common.py) holds every shared setting: the cell
parameters, heat transfer coefficients, heating profiles and tolerances.

### Data

- `data/wltp.csv`: WLTP drive-cycle current profile.
- `data/MLP001_ocv.csv`, `data/MLP001_params.csv`: equivalent-circuit
  model parameters, used to turn the drive cycle into a heat generation
  profile.

Thermal properties of the Kokam pouch cell (`coresurf.kokam_pouch`) come from:

1. Hales et al., *The Cell Cooling Coefficient: A Standard to Define Heat
   Rejection from Lithium-Ion Batteries*.
2. Hales et al., *The Surface Cell Cooling Coefficient: A Standard to
   Define Heat Rejection from Lithium Ion Battery Pouch Cells*.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
