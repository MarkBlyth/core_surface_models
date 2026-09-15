from typing import Callable, Tuple
import numpy as np
import numpy.typing as npt
import scipy.integrate
import pybamm
import thermalparams


"""
API for consistent definition of thermal parameters and models.
Ensures each simulation uses the same set of thermal parameters, and
is solved in the same way. Thermal properties come from two sources
(both of which cover the same cell).

[1] The Cell Cooling Coefficient: A Standard to Define Heat Rejection from Lithium-Ion Batteries (Cell A, Kokam power cell)
[2] The Surface Cell Cooling Coefficient: A Standard to Define Heat Rejection from Lithium Ion Battery Pouch Cells (cell A, the same Kokam power cell as previously)
"""


TOL = 1e-7

test_hs = [30, 150, 1000]
standard_heatgen = 20 / (0.113 * 0.04 * 0.0113)  # [2]

_thickness = 0.0113  # [1]
_density = 0.123 / (0.113 * 0.04 * 0.0113)  # [1]
_specific_heat_capacity = 1030  # [1]
_through_plane_conductivity = 0.916  # [2]


def get_asymmetric_thermals(h0: float, hL: float) -> thermalparams.ThermalParameters:
    """
    Get a standard set of thermal parameters for asymmetric cooling
    """
    return thermalparams.ThermalParameters(
        _thickness,
        _through_plane_conductivity,
        h0,
        hL,
        _density,
        _specific_heat_capacity,
    )


def get_symmetric_thermals(h: float) -> thermalparams.ThermalParameters:
    """
    Get a standard set of thermal parameters for symmetric cooling.
    A literature core-surface model will automatically apply them over
    the half-domain [0, L/2].

    """
    return thermalparams.ThermalParameters(
        _thickness, _through_plane_conductivity, h, h, _density, _specific_heat_capacity
    )


def get_half_domain_thermals(h: float) -> thermalparams.ThermalParameters:
    """
    Builds parameters over [0, L/2], with insulation at x=0, for
    modelling a half-domain with symmetric cooling.

    This puts the hottest part of the cell at x=0, so that T0 = max.
    temperature in a core-surface model.
    """
    return thermalparams.ThermalParameters(
        _thickness / 2,
        _through_plane_conductivity,
        0,
        h,
        _density,
        _specific_heat_capacity,
    )


def get_mirror_image_half_domain_thermals(h: float) -> thermalparams.ThermalParameters:
    """
    Builds parameters over [0, L/2], with insulation at x=L/2, and
    cooling at x=0. This is a mirror image of the solution to the
    half_domain thermals.

    Puts the coolest part of the cell at x=0, so that T0 = min.
    temperature in a core-surface model.
    """
    return thermalparams.ThermalParameters(
        _thickness / 2,
        _through_plane_conductivity,
        h,
        0,
        _density,
        _specific_heat_capacity,
    )


def get_pybamm_soln(
    thermals: thermalparams.ThermalParameters,
    heat_gen: float | pybamm.Interpolant,
    t_eval: npt.ArrayLike,
) -> pybamm.Solution:
    """
    Solve a 1d heat equation using PyBaMM.
    """
    params = pybamm.ParameterValues(
        {
            "Density": thermals.rho,
            "Specific heat capacity": thermals.cp,
            "Thermal conductivity": thermals.k,
            # W/m3; chosen as 1W generated in an LG M50 jelly roll, volume 2.13e-5 m3 [1]
            "Heat generation": heat_gen,  # Volumetric
            "LHS HTC": thermals.h0,
            "RHS HTC": thermals.hL,
            "Domain size": thermals.L,
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
    dTdt = (-pybamm.div(heatflux) + Q) / (rho * cp)
    model = pybamm.BaseModel()
    model.rhs = {T: dTdt}
    model.boundary_conditions = {
        T: {
            "left": (
                (h0 / k) * pybamm.BoundaryValue(T, "left"),
                "Neumann",
            ),
            "right": (
                -(hL / k) * pybamm.BoundaryValue(T, "right"),
                "Neumann",
            ),
        }
    }
    model.initial_conditions = {T: 0}
    model.variables = {
        "Temperature": T,
        "Volume averaged temperature": pybamm.Integral(T, x) / L,
    }

    geometry = {"bulk": {x: {"min": pybamm.Scalar(0), "max": L}}}
    params.process_model(model)
    params.process_geometry(geometry)

    submesh_types = {"bulk": pybamm.Uniform1DSubMesh}
    var_pts = {x: 250}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {"bulk": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)

    solver = pybamm.ScipySolver(atol=TOL, rtol=TOL)
    return solver.solve(model, t_eval)


def get_literature_core_surf_ode(
    thermals: thermalparams.ThermalParameters,
    heatgen: float | Callable[[npt.ArrayLike], npt.ArrayLike],
    T_inf: float,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate the differential equations for a literature core-surface
    model.
    """
    if thermals.h0 != thermals.hL:
        raise ValueError("Can only build a literature model for symmetric cooling")

    heat_gen: Callable[[npt.ArrayLike], npt.ArrayLike] = (
        heatgen if callable(heatgen) else lambda _: heatgen
    )

    def ode_rhs(t, x):
        T_i, T_s = x
        thermal_mass = thermals.rho_cp * thermals.L * 0.25
        dTi_dt = (
            0.5 * thermals.L * heat_gen(t) - 2 * thermals.k * (T_i - T_s) / thermals.L
        ) / thermal_mass
        dTs_dt = (
            2 * thermals.k * (T_i - T_s) / thermals.L - thermals.hL * (T_s - T_inf)
        ) / thermal_mass
        return dTi_dt, dTs_dt

    return ode_rhs


def get_core_surf_soln(
    ode: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    t_eval: npt.NDArray,
    T_inf: float,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Solve the differential equations of a literature core-surface
    model.
    """
    soln = scipy.integrate.solve_ivp(
        ode,
        [t_eval[0], t_eval[-1]],
        [T_inf, T_inf],
        t_eval=t_eval,
        atol=TOL,
        rtol=TOL,
    )
    return soln.y


def solve_thermal(
    A: npt.NDArray,
    b: npt.NDArray,
    heatgen: float | Callable[[npt.ArrayLike], npt.ArrayLike],
    T_ambient: float,
    t_eval: npt.NDArray,
    **solver_kwargs,
):
    """
    Solve the differential equations of one of our core-surface
    models, which are defined by matrix A and vector b.
    """

    def const_heat_ode(t, x):
        return np.matmul(A, (x - T_ambient)) + b * heatgen

    def functional_heat_ode(t, x):
        return np.matmul(A, (x - T_ambient)) + b * heatgen(t)

    ode = functional_heat_ode if callable(heatgen) else const_heat_ode
    soln = scipy.integrate.solve_ivp(
        ode,
        [t_eval[0], t_eval[-1]],
        [T_ambient, T_ambient],
        t_eval=t_eval,
        atol=TOL,
        rtol=TOL,
        **solver_kwargs,
    )
    return soln.y
