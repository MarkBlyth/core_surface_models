"""Help choosing a model: Biot number and modal timescales."""

from __future__ import annotations

import numpy.typing as npt

from .galerkin import EigenmodeFamily
from .parameters import ThermalParameters


def modal_timescales(params: ThermalParameters, n: int = 3) -> npt.NDArray:
    """
    Time constants tau_1, tau_2, ... [s] of the first n heat-transfer
    modes. tau_1 is roughly how long the cell takes to heat up or cool
    down; tau_2 and tau_3 show how quickly the temperature *shape* settles.
    """
    return EigenmodeFamily(params).timescales(n)


def describe(params: ThermalParameters, show: bool = True) -> str:
    """
    Print (and return) a short plain-language summary of a cell's
    thermal behaviour, with a suggested model, following Section 5 of
    the paper.
    """
    tau = modal_timescales(params, 3)
    bi = params.biot
    lines = [
        f"Cell thickness L = {params.L * 1000:.2f} mm, "
        f"cooling h0 = {params.h0:g}, hL = {params.hL:g} W/m2/K",
        f"Biot number = {bi:.3f}  "
        + (
            "(small: temperature is nearly uniform through the cell)"
            if bi < 0.1
            else "(moderate: noticeable core-surface temperature difference)"
            if bi < 1
            else "(large: big core-surface temperature difference)"
        ),
        "Modal timescales: "
        + ", ".join(f"tau_{i + 1} = {t:.1f} s" for i, t in enumerate(tau)),
        "",
        "Suggested model:",
    ]
    if params.is_symmetric:
        lines.append(
            "  symmetric_core_surface_model(params) -- {phi_1, Psi} Galerkin model,\n"
            "  giving core, surface and volume-averaged temperature."
        )
    else:
        lines.append(
            "  eigenmode_steadystate_model(params) -- {phi_1, Psi} Galerkin model,\n"
            "  giving volume-averaged temperature and temperature at x = 0."
        )
    lines.append(
        f"  If you care about events faster than about tau_1 = {tau[0]:.0f} s, or heating\n"
        f"  changes on the tau_1 to tau_2 timescale, compare with two_eigenmode_model."
    )
    if bi < 0.5:
        lines.append(
            "  Biot < 0.5, so small_biot_model is also valid, and its equations are\n"
            "  simple enough to write down by hand."
        )
    text = "\n".join(lines)
    if show:
        print(text)
    return text
