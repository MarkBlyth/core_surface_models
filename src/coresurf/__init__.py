"""
coresurf: accurate two-state core-surface thermal models for battery cells.

Quick start::

    import numpy as np
    import coresurf

    params = coresurf.kokam_pouch(h0=150)          # symmetric cooling, h = 150 W/m2/K
    model = coresurf.symmetric_core_surface_model(params)
    t = np.linspace(0, 600, 200)
    soln = model.simulate(heatgen=3.5e5, t_eval=t)  # W/m3
    soln["T_core"], soln["T_surf"], soln["T_mean"]

New users: start with docs/guide.md and the scripts in examples/.
"""

from .builders import (
    SymmetricCoreSurfaceModel,
    build_eigenmode_steadystatemode_model,
    build_small_biot_model,
    build_two_eigenmodes_model,
    eigenmode_steadystate_model,
    literature_model,
    small_biot_model,
    symmetric_core_surface_model,
    two_eigenmode_model,
)
from .galerkin import EigenmodeFamily, GalerkinProjection, Mode, steadystate_mode
from .model import DEFAULT_TOL, CoreSurfaceModel, Solution
from .parameters import ThermalParameters, kokam_pouch
from .selection import describe, modal_timescales

__all__ = [
    "CoreSurfaceModel",
    "DEFAULT_TOL",
    "EigenmodeFamily",
    "GalerkinProjection",
    "Mode",
    "Solution",
    "SymmetricCoreSurfaceModel",
    "ThermalParameters",
    "build_eigenmode_steadystatemode_model",
    "build_small_biot_model",
    "build_two_eigenmodes_model",
    "describe",
    "modal_timescales",
    "eigenmode_steadystate_model",
    "kokam_pouch",
    "literature_model",
    "small_biot_model",
    "steadystate_mode",
    "symmetric_core_surface_model",
    "two_eigenmode_model",
]

__version__ = "0.1.0"
