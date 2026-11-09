"""
Two-mode Galerkin projections of the 1D heat equation, transformed into
core-surface form.

The main entry point is `GalerkinProjection`, which takes a pair of
trial functions (`Mode`s) and produces a `CoreSurfaceModel`. Ready-made
trial functions are available from `EigenmodeFamily` (the eigenfunctions
of the heat operator) and `steadystate_mode` (the steady-state
temperature profile under uniform heating).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.optimize

from .model import CoreSurfaceModel
from .parameters import ThermalParameters

ModeFunction = Callable[[npt.ArrayLike], npt.ArrayLike]


@dataclass
class Mode:
    """
    A single Galerkin trial function, normalised so mode(0) = 1.

    Supply either `second_derivative`, or `eigenvalue_squared` if the
    mode is an eigenfunction of d2/dx2 (in which case
    second_derivative = -eigenvalue_squared * mode).
    """

    mode: ModeFunction
    second_derivative: Optional[ModeFunction] = None
    eigenvalue_squared: Optional[float] = None

    def __post_init__(self):
        mode0 = self.mode(0.0)
        if not np.isclose(mode0, 1.0):
            raise ValueError(f"mode(0) must equal 1, got {mode0}")
        if self.second_derivative is None:
            if self.eigenvalue_squared is None:
                raise ValueError(
                    "must supply second_derivative, or eigenvalue_squared "
                    "for an eigenfunction mode"
                )
            mu2 = self.eigenvalue_squared
            self.second_derivative = lambda x: -mu2 * self.mode(x)


class GalerkinProjection:
    """
    Core-surface transformation of a two-mode Galerkin projection of the
    1D heat equation onto span{mode1, mode2}, with both modes normalised
    to mode(0) = 1.

    The resulting model has states

        x = [T_mean, T_0]

    where T_mean is the volume-averaged temperature and T_0 = T(x=0).
    """

    def __init__(self, params: ThermalParameters, mode1: Mode, mode2: Mode):
        self.params = params
        self.mode1 = mode1
        self.mode2 = mode2
        self._projection_constants = None
        self._matrices = None

    def _inner(self, f, g) -> float:
        soln, _ = scipy.integrate.quad(lambda x: f(x) * g(x), 0, self.params.L)
        return soln

    def _average(self, f) -> float:
        soln, _ = scipy.integrate.quad(f, 0, self.params.L)
        return soln / self.params.L

    def _compute_constants(self):
        if self._projection_constants is not None:
            return self._projection_constants

        m1, m2 = self.mode1.mode, self.mode2.mode
        d1, d2 = self.mode1.second_derivative, self.mode2.second_derivative

        m1_m1 = self._inner(m1, m1)
        m2_m2 = self._inner(m2, m2)
        m1_m2 = self._inner(m1, m2)
        m1_d1 = self._inner(m1, d1)
        m2_d1 = self._inner(m2, d1)
        m1_d2 = self._inner(m1, d2)
        m2_d2 = self._inner(m2, d2)
        int_m1 = self._inner(m1, lambda x: 1)
        int_m2 = self._inner(m2, lambda x: 1)

        c1 = m1_m1 * m2_m2 - m1_m2**2
        c2 = m2_m2 * m1_d1 - m1_m2 * m2_d1
        c3 = m2_m2 * m1_d2 - m1_m2 * m2_d2
        c4 = m2_m2 * int_m1 - m1_m2 * int_m2
        c5 = m1_m1 * m2_d1 - m1_m2 * m1_d1
        c6 = m1_m1 * m2_d2 - m1_m2 * m1_d2
        c7 = m1_m1 * int_m2 - m1_m2 * int_m1

        self._projection_constants = (c1, c2, c3, c4, c5, c6, c7)
        return self._projection_constants

    def get_model_matrices(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Return (A, b) such that dx/dt = A (x - T_inf) + b q(t)."""
        if self._matrices is not None:
            return self._matrices

        c1, c2, c3, c4, c5, c6, c7 = self._compute_constants()
        m1av = self._average(self.mode1.mode)
        m2av = self._average(self.mode2.mode)
        D = self.params.diffusivity

        const = -D / (c1 * (m1av - m2av))
        A11 = const * (m1av * (c3 - c2) + m2av * (c6 - c5))
        A12 = const * (m2av * (m1av * c2 + m2av * c5) - m1av * (m1av * c3 + m2av * c6))
        A21 = const * (c3 + c6 - c2 - c5)
        A22 = const * (m2av * (c2 + c5) - m1av * (c3 + c6))
        A = np.array([[A11, A12], [A21, A22]])

        b_mean = (c4 * m1av + c7 * m2av) / (self.params.rho_cp * c1)
        b_0 = (c4 + c7) / (self.params.rho_cp * c1)
        b = np.array([b_mean, b_0])

        self._matrices = (A, b)
        return A, b

    def model(self) -> CoreSurfaceModel:
        A, b = self.get_model_matrices()
        return CoreSurfaceModel(A, b, state_names=("T_mean", "T_0"))


class EigenmodeFamily:
    """The eigenfunctions phi_n of the Robin-Robin heat operator on [0, L]."""

    def __init__(self, params: ThermalParameters):
        self.params = params
        self._eigenvalues: npt.NDArray = np.array([])

    def eigenvalues(self, n: int) -> npt.NDArray:
        """The first n eigenvalues mu_n (phi_n'' = -mu_n^2 phi_n)."""
        if n > len(self._eigenvalues):
            self._eigenvalues = self._solve_eigenvalues(n)
        return self._eigenvalues[:n]

    def _solve_eigenvalues(self, n_eigenvals: int) -> npt.NDArray:
        """
        Root-find the transcendental eigenvalue equation for the first n
        eigenvalues. Root n is bracketed in ((n-1) pi / L, n pi / L),
        widened if necessary.
        """
        p = self.params
        ret = []

        def objective(mu):
            return np.sin(mu * p.L) * (p.h0 * p.hL - (mu**2) * (p.k**2)) + mu * p.k * (
                p.h0 + p.hL
            ) * np.cos(mu * p.L)

        for n in range(1, n_eigenvals + 1):
            lo = (n - 1) * np.pi / p.L + 1e-5
            hi = n * np.pi / p.L - 1e-5
            if objective(lo) * objective(hi) > 0:
                hi = (n - 0.5) * np.pi / p.L
            if objective(lo) * objective(hi) > 0:
                hi = (n + 0.5) * np.pi / p.L

            soln = scipy.optimize.root_scalar(objective, bracket=[lo, hi])
            ret.append(soln.root)

        return np.array(ret)

    def mode(self, index: int) -> Mode:
        """The index'th eigenfunction as a Mode (1-indexed)."""
        mu_squared = self.eigenvalues(index)[index - 1] ** 2
        return Mode(
            mode=lambda x, i=index: self._eigenfunction(x, i),
            eigenvalue_squared=mu_squared,
        )

    def _eigenfunction(self, x, index: int) -> npt.ArrayLike:
        mu = self.eigenvalues(index)[index - 1]
        if mu == 0:
            return np.ones_like(np.asarray(x, dtype=float))
        h0, k = self.params.h0, self.params.k
        return np.cos(mu * x) + h0 * np.sin(mu * x) / (mu * k)


def steadystate_mode(params: ThermalParameters) -> Mode:
    """
    The steady-state temperature profile under uniform heat generation,
    normalised to mode(0) = 1. Including this as a trial function makes
    the core-surface model exact at steady state.
    """
    k, h0, hL, L = params.k, params.h0, params.hL, params.L
    numerator = k * (h0 + hL) + L * h0 * hL
    denominator = 2 * k**2 * L + hL * k * L**2
    frac = numerator / denominator  # coefficient of x^2

    def mode(x):
        return 1 + (h0 / k) * x - frac * x**2

    def second_derivative(x):
        return -2.0 * frac * np.ones_like(np.asarray(x, dtype=float))

    return Mode(mode=mode, second_derivative=second_derivative)
