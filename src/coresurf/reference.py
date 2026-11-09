"""
Reference (ground-truth) solution of the full 1D heat equation, using
PyBaMM's finite-volume discretisation. Requires the optional `pybamm`
dependency (`pip install coresurf[reference]`).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .model import DEFAULT_TOL, Solution
from .parameters import ThermalParameters


def solve_heat_equation(
    params: ThermalParameters,
    heatgen,
    t_eval: npt.ArrayLike,
    n_points: int = 250,
    tol: float = DEFAULT_TOL,
):
    """
    Solve the 1D heat equation on [0, L], starting from (and relative
    to) ambient temperature, so temperatures are rises above ambient.

    Parameters
    ----------
    params : thermal parameters.
    heatgen : volumetric heat generation [W m^-3]; a constant, a
        (times, values) tuple, or a pybamm expression in pybamm.t.
    t_eval : output times [s].
    n_points : number of finite-volume cells.
    tol : solver absolute and relative tolerance.

    Returns
    -------
    pybamm.Solution, with variables
        "Temperature": call as soln["Temperature"](t, x)
        "Volume averaged temperature": call as soln["Volume averaged temperature"](t)
    """
    import pybamm

    if isinstance(heatgen, tuple):
        times, values = (np.asarray(a, dtype=float) for a in heatgen)
        heatgen = pybamm.Interpolant(times, values, pybamm.t, "Heat generation")
    elif callable(heatgen) and not isinstance(heatgen, pybamm.Symbol):
        raise TypeError(
            "Pass time-varying heat generation as a (times, values) tuple "
            "or a pybamm expression, not a Python function"
        )

    parameter_values = pybamm.ParameterValues(
        {
            "Density": params.rho,
            "Specific heat capacity": params.cp,
            "Thermal conductivity": params.k,
            "Heat generation": heatgen,
            "LHS HTC": params.h0,
            "RHS HTC": params.hL,
            "Domain size": params.L,
        }
    )

    x = pybamm.SpatialVariable("x", domain="bulk", coord_sys="cartesian")
    T = pybamm.Variable("Temperature", domain="bulk")
    rho = pybamm.Parameter("Density")
    cp = pybamm.Parameter("Specific heat capacity")
    k = pybamm.Parameter("Thermal conductivity")
    Q = pybamm.Parameter("Heat generation")
    h0 = pybamm.Parameter("LHS HTC")
    hL = pybamm.Parameter("RHS HTC")
    L = pybamm.Parameter("Domain size")

    heatflux = -k * pybamm.grad(T)
    model = pybamm.BaseModel()
    model.rhs = {T: (-pybamm.div(heatflux) + Q) / (rho * cp)}
    model.boundary_conditions = {
        T: {
            "left": ((h0 / k) * pybamm.BoundaryValue(T, "left"), "Neumann"),
            "right": (-(hL / k) * pybamm.BoundaryValue(T, "right"), "Neumann"),
        }
    }
    model.initial_conditions = {T: 0}
    model.variables = {
        "Temperature": T,
        "Volume averaged temperature": pybamm.Integral(T, x) / L,
    }

    geometry = {"bulk": {x: {"min": pybamm.Scalar(0), "max": L}}}
    parameter_values.process_model(model)
    parameter_values.process_geometry(geometry)

    mesh = pybamm.Mesh(geometry, {"bulk": pybamm.Uniform1DSubMesh}, {x: n_points})
    disc = pybamm.Discretisation(mesh, {"bulk": pybamm.FiniteVolume()})
    disc.process_model(model)

    solver = pybamm.ScipySolver(atol=tol, rtol=tol)
    return solver.solve(model, t_eval)


def symmetric_core_surface(
    params: ThermalParameters,
    heatgen,
    t_eval: npt.ArrayLike,
    **kwargs,
):
    """
    Ground-truth core and surface temperature rises for a symmetrically
    cooled slab, solved on the half-domain. Returns a `Solution` with
    states (T_core, T_surf, T_mean), directly comparable with
    `coresurf.symmetric_core_surface_model(params).simulate(...)`.
    `kwargs` are passed to `solve_heat_equation`.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    half = params.half_domain()
    soln = solve_heat_equation(half, heatgen, t_eval, **kwargs)
    y = np.vstack(
        [
            soln["Temperature"](t_eval, 0),
            soln["Temperature"](t_eval, half.L),
            soln["Volume averaged temperature"](t_eval),
        ]
    )
    return Solution(t_eval, y, ("T_core", "T_surf", "T_mean"))
