# core_surface_models

Fast, accurate core-surface thermal models for battery cells, and the
scripts and data that reproduce our paper *Why core-surface battery
thermal models fail, and how to fix them*.

Traditional core-surface models assume all heat is generated at the
centre of the cell, and describe the temperature with only two points.
As a result they overpredict core temperature (by up to 75 % in our
tests) and get the transient response wrong. The models here spread heat
generation through the cell and describe the temperature with its two
dominant heat-transfer modes. They are built from physical properties,
with no fitting, and run just as fast as the traditional model. They
also give the volume-averaged cell temperature, which is the natural
input to an electrical model.

**New users: start with the [guide](docs/guide.md).** It explains what
the models do, how to choose one, and how to get started, whether you
use Spyder, VS Code, or a terminal and your own editor. The
[examples](examples/) folder has ready-to-run scripts.

## Quick start

No installation is needed to try it: open
`examples/01_first_simulation.py` in your editor (Spyder, VS Code, ...)
and run it, or from this folder:

```bash
python examples/01_first_simulation.py
```

To use the models from your own scripts, install the package from this
folder (the `examples` extra adds matplotlib and pandas for the
examples):

```bash
pip install -e ".[examples]"
```

Then:

```python
import numpy as np
import coresurf

params = coresurf.ThermalParameters(
    L=0.0113,   # thickness [m]
    k=0.916,    # through-plane thermal conductivity [W/m/K]
    h0=150,     # heat transfer coefficient, face x=0 [W/m2/K]
    hL=150,     # heat transfer coefficient, face x=L [W/m2/K]
    rho=2408,   # density [kg/m3]
    cp=1030,    # specific heat capacity [J/kg/K]
)
coresurf.describe(params)       # Biot number, timescales, suggested model

model = coresurf.symmetric_core_surface_model(params)
times = np.linspace(0, 1200, 300)                 # [s]
heat = 20 / (0.113 * 0.04 * 0.0113)               # 20 W, as W/m3
soln = model.simulate(heat, times, T_ambient=25)

soln["T_core"], soln["T_surf"], soln["T_mean"]    # centre, surface, average [degC]
```

## Models

| Builder | Outputs | Cooling | Use |
|---|---|---|---|
| `symmetric_core_surface_model` | `T_core`, `T_surf`, `T_mean` | equal on both faces | **Recommended** |
| `eigenmode_steadystate_model` ({φ₁, Ψ}) | `T_mean`, `T_0` | any | Recommended for unequal cooling |
| `two_eigenmode_model` ({φ₁, φ₂}) | `T_mean`, `T_0` | any | Very short timescales |
| `small_biot_model` | `T_mean`, `T_0` | any | Biot < 0.5; simple closed form |
| `literature_model` | `T_core`, `T_surf` | equal on both faces | Comparison only |

`T_mean` is the volume-averaged temperature and `T_0` is the temperature
of face x = 0. All models share the same interface:

- `model.simulate(heat, times, T_ambient, T_initial=..., max_step=...)`.
  `heat` [W m⁻³] can be a constant, a function of time, or a
  `(times, values)` table. `T_ambient` is a constant ambient temperature.
- `model.temperature_profile(soln, x, T_ambient)` gives the temperature
  through the cell, for example to find the hottest point (Galerkin
  models).
- `model.A`, `model.b` hold the model as `dx/dt = A (x - T_amb) + b q(t)`,
  for use in a BMS, Simulink, or a pack model.

See the [guide](docs/guide.md) for details and worked examples. The code
listing in the paper (`build_eigenmode_steadystatemode_model`,
`get_model_matrices`) also works.

### Coupling with an electrical model

[examples/07_electrothermal_ecm.py](examples/07_electrothermal_ecm.py)
solves the thermal model together with a Thevenin equivalent-circuit
model (`coresurf.ecm.TheveninModel`), with the electrical parameters
evaluated at the volume-averaged temperature and the resistive losses
heating the cell.

### Custom thermal modes

```python
modes = coresurf.EigenmodeFamily(params)
my_mode = coresurf.Mode(mode=lambda x: ..., second_derivative=lambda x: ...)
model = coresurf.GalerkinProjection(params, modes.mode(1), my_mode).model()
```

The core-surface transformation assumes **every thermal mode satisfies
φ(0) = 1**, so that the second state is the temperature at x = 0. `Mode`
rejects functions that don't; `Mode.normalised(f, f'')` rescales f by
f(0) for you. Modes must also satisfy the cell's Robin
boundary conditions, k φ'(0) = h0 φ(0) and −k φ'(L) = hL φ(L), because
the projection uses φ'' directly; `GalerkinProjection` checks both.

### Reference solution

`coresurf.reference.solve_heat_equation(params, heat, times)` solves the
full 1D heat equation with PyBaMM (`pip install -e ".[reference]"`).
`coresurf.reference.symmetric_core_surface(...)` returns ground-truth
`T_core`, `T_surf` and `T_mean` in the same format as the models.

## Repository layout

| Folder | Contents |
|---|---|
| `src/coresurf/` | The Python package |
| `docs/` | [User guide](docs/guide.md) |
| `examples/` | Step-by-step example scripts, split into `# %%` cells |
| `paper/` | Scripts that reproduce every figure and number in the paper |
| `data/` | Drive-cycle and cell data used by the paper and the examples |
| `tests/` | Automated tests (`pytest`) |

## Reproducing the paper

```bash
pip install -e ".[paper]"
python paper/run_all.py                 # PGF figures in ./figures, as in the paper
python paper/run_all.py --format pdf    # or PDF/PNG for viewing
```

| Script | Output |
|---|---|
| `paper/fig_literature_comparison.py` | `literature_model_comparison` |
| `paper/fig_newmodel_comparison.py` | `newmodel_comparison` |
| `paper/fig_summary_comparison.py` | `summary_comparison` |
| `paper/fig_wltp_comparison.py` | `wltp_comparison` |
| `paper/fig_wltp_zoom.py` | `wltp_zoom` |
| `paper/calc_steadystate_error.py` | Steady-state errors quoted in the text |
| `paper/calc_wltp_error.py` | WLTP RMS errors quoted in the text |

Figure scripts accept `--outdir`, `--format` and `--show`. PGF output
needs a LaTeX installation. [paper/common.py](paper/common.py) holds
every shared setting: the cell parameters, heat transfer coefficients,
heating profiles and tolerances.

### Data

- `data/wltp.csv`: WLTP drive-cycle current profile.
- `data/MLP001_ocv.csv`, `data/MLP001_params.csv`: equivalent-circuit
  model parameters, used to turn the drive cycle into a heat generation
  profile. The Thevenin model and these parameters are from M. Blyth,
  A. Gupta and A. Hales, *How to parameterise an equivalent-circuit
  empirical battery model from time-domain data*, MethodsX (2026), 103865.

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
