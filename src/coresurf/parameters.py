"""Thermal parameters for a 1D slab, and standard parameter sets."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ThermalParameters:
    """
    Parameters of the 1D heat equation on the slab [0, L],

        rho cp dT/dt = k d2T/dx2 + q(t),

    with Robin (convective) boundary conditions

        k dT/dx = h0 (T - T_inf)   at x = 0,
       -k dT/dx = hL (T - T_inf)   at x = L.

    All quantities are in SI units.

    Attributes
    ----------
    L : slab thickness [m]
    k : through-plane thermal conductivity [W m^-1 K^-1]
    h0 : heat transfer coefficient at x = 0 [W m^-2 K^-1]
    hL : heat transfer coefficient at x = L [W m^-2 K^-1]
    rho : density [kg m^-3]
    cp : specific heat capacity [J kg^-1 K^-1]
    """

    L: float
    k: float
    h0: float
    hL: float
    rho: float
    cp: float

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity [J m^-3 K^-1]."""
        return self.rho * self.cp

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity k / (rho cp) [m^2 s^-1]."""
        return self.k / self.rho_cp

    @property
    def biot(self) -> float:
        """Biot number, using the mean of the two heat transfer coefficients."""
        return 0.5 * (self.h0 + self.hL) * self.L / self.k

    @property
    def is_symmetric(self) -> bool:
        return self.h0 == self.hL

    def with_cooling(self, h0: float, hL: float | None = None) -> ThermalParameters:
        """Copy of these parameters with new heat transfer coefficients.
        If only h0 is given, cooling is symmetric (hL = h0)."""
        return replace(self, h0=h0, hL=h0 if hL is None else hL)

    def mirrored(self) -> ThermalParameters:
        """Swap the two boundaries, so that x=0 becomes x=L and vice versa."""
        return replace(self, h0=self.hL, hL=self.h0)

    def half_domain(self) -> ThermalParameters:
        """
        For symmetric cooling, the half-domain [0, L/2] with an insulated
        boundary at x=0 (the cell centre) and the original cooling at
        x=L/2 (the cell surface).

        A core-surface model built on these parameters has T(0) equal to
        the core (maximum) temperature. Use `.half_domain().mirrored()`
        to put the surface at x=0 instead.
        """
        if not self.is_symmetric:
            raise ValueError("half_domain() requires symmetric cooling (h0 == hL)")
        return replace(self, L=self.L / 2, h0=0.0, hL=self.hL)


# ---------------------------------------------------------------------------
# Standard parameter sets
# ---------------------------------------------------------------------------


def kokam_pouch(h0: float, hL: float | None = None) -> ThermalParameters:
    """
    Kokam power pouch cell ("cell A") used throughout the paper.
    If only h0 is given, cooling is symmetric.

    Sources:
    [1] Hales et al., The Cell Cooling Coefficient: A Standard to Define
        Heat Rejection from Lithium-Ion Batteries (thickness, density,
        specific heat capacity).
    [2] Hales et al., The Surface Cell Cooling Coefficient: A Standard to
        Define Heat Rejection from Lithium Ion Battery Pouch Cells
        (through-plane conductivity).
    """
    return ThermalParameters(
        L=KOKAM_POUCH_THICKNESS,
        k=0.916,  # [2]
        h0=h0,
        hL=h0 if hL is None else hL,
        rho=0.123 / KOKAM_POUCH_VOLUME,  # [1]
        cp=1030,  # [1]
    )


KOKAM_POUCH_THICKNESS = 0.0113  # m, [1]
KOKAM_POUCH_VOLUME = 0.113 * 0.04 * KOKAM_POUCH_THICKNESS  # m^3, [1]
