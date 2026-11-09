"""
Ready-made core-surface models.

Each builder takes a `ThermalParameters` and returns a model with a
`.simulate(heatgen, t_eval, ...)` method. They differ in what their two
states represent:

====================================  ==========================  ================
Builder                               States                      Cooling
====================================  ==========================  ================
`symmetric_core_surface_model`        T_core, T_surf              symmetric
`eigenmode_steadystate_model`         T_mean, T_0                 any
`two_eigenmode_model`                 T_mean, T_0                 any
`small_biot_model`                    T_mean, T_0                 any
`literature_model`                    T_core, T_surf              symmetric
====================================  ==========================  ================

T_mean is the volume-averaged temperature, T_0 the temperature at x=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .galerkin import EigenmodeFamily, GalerkinProjection, steadystate_mode
from .model import CoreSurfaceModel, HeatGeneration, Solution, as_heat_function
from .parameters import ThermalParameters


def eigenmode_steadystate_model(params: ThermalParameters) -> CoreSurfaceModel:
    """
    Galerkin projection onto {phi_1, Psi}: the first eigenfunction plus
    the steady-state profile. Exact at steady state under uniform heating.
    States: (T_mean, T_0).
    """
    eigenmodes = EigenmodeFamily(params)
    return GalerkinProjection(params, eigenmodes.mode(1), steadystate_mode(params)).model()


def two_eigenmode_model(params: ThermalParameters) -> CoreSurfaceModel:
    """
    Galerkin projection onto {phi_1, phi_2}, the first two eigenfunctions.
    States: (T_mean, T_0).
    """
    eigenmodes = EigenmodeFamily(params)
    return GalerkinProjection(params, eigenmodes.mode(1), eigenmodes.mode(2)).model()


def small_biot_model(params: ThermalParameters) -> CoreSurfaceModel:
    """
    Small-Biot-number approximation. States: (T_mean, T_0).
    """
    h0, hL = params.h0, params.hL
    rho_cp, L, k = params.rho_cp, params.L, params.k

    k1 = 3 * hL * (h0 - hL) / (L * (2 * h0 - hL))
    k23 = 2 * (h0**2 - h0 * hL + hL**2) / (L * (2 * h0 - hL))
    k4 = (4 * h0**2 - h0 * hL + hL**2) / (L * (2 * h0 - hL))

    A = np.array([[-k1, -k23], [k23, -k4]]) / rho_cp

    w2 = 1 - L * (2 * h0 - hL) / (6 * k)
    b = np.array([1, w2]) / rho_cp

    return CoreSurfaceModel(A, b, state_names=("T_mean", "T_0"))


def literature_model(params: ThermalParameters) -> CoreSurfaceModel:
    """
    The standard core-surface model from the literature: two lumped
    thermal masses (core and surface half-shells, each a quarter of the
    slab) connected by a conduction resistance, with heat generated in
    the core. Requires symmetric cooling. States: (T_core, T_surf).
    """
    if not params.is_symmetric:
        raise ValueError("Can only build a literature model for symmetric cooling")
    L, k, h, rho_cp = params.L, params.k, params.hL, params.rho_cp
    thermal_mass = rho_cp * L / 4
    conductance = 2 * k / L
    A = np.array(
        [
            [-conductance, conductance],
            [conductance, -conductance - h],
        ]
    ) / thermal_mass
    b = np.array([0.5 * L / thermal_mass, 0.0])
    return _LiteratureModel(A, b, state_names=("T_core", "T_surf"), params=params)


@dataclass
class _LiteratureModel(CoreSurfaceModel):
    """Literature model, with the right-hand side evaluated exactly as in
    the paper's original scripts, so that results reproduce bit-for-bit.
    Mathematically identical to the (A, b) form."""

    params: ThermalParameters | None = None

    def rhs(self, heatgen: HeatGeneration, T_ambient: float = 0.0):
        p = self.params
        q = as_heat_function(heatgen)
        thermal_mass = p.rho_cp * p.L * 0.25

        def ode_rhs(t, x):
            T_i, T_s = x
            dTi_dt = (0.5 * p.L * q(t) - 2 * p.k * (T_i - T_s) / p.L) / thermal_mass
            dTs_dt = (2 * p.k * (T_i - T_s) / p.L - p.hL * (T_s - T_ambient)) / thermal_mass
            return dTi_dt, dTs_dt

        return ode_rhs


@dataclass
class SymmetricCoreSurfaceModel:
    """
    Core and surface temperatures of a symmetrically cooled slab, from
    two {phi_1, Psi} Galerkin models on the half-domain: one with the
    insulated centre at x=0 (giving T_core), and its mirror image with
    the cooled surface at x=0 (giving T_surf).

    Build with `symmetric_core_surface_model(params)`.
    """

    core: CoreSurfaceModel
    surface: CoreSurfaceModel
    state_names: tuple[str, ...] = ("T_core", "T_surf")

    def simulate(
        self,
        heatgen: HeatGeneration,
        t_eval: npt.ArrayLike,
        T_ambient: float = 0.0,
        **solver_kwargs,
    ) -> Solution:
        """Simulate from ambient; see `CoreSurfaceModel.simulate`."""
        core = self.core.simulate(heatgen, t_eval, T_ambient, **solver_kwargs)
        surf = self.surface.simulate(heatgen, t_eval, T_ambient, **solver_kwargs)
        return Solution(core.t, np.vstack([core["T_0"], surf["T_0"]]), self.state_names)


def symmetric_core_surface_model(params: ThermalParameters) -> SymmetricCoreSurfaceModel:
    """
    Recommended core-surface model for a symmetrically cooled cell.
    `params` describes the full cell, [0, L]. States: (T_core, T_surf).
    """
    half = params.half_domain()
    return SymmetricCoreSurfaceModel(
        core=eigenmode_steadystate_model(half),
        surface=eigenmode_steadystate_model(half.mirrored()),
    )
